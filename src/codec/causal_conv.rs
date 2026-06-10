//! # 因果卷積網路 + 環形緩衝區 (12Hz 核心)
//!
//! 實作 12Hz 即時解碼器所需的因果卷積層。
//!
//! ## 因果卷積特性
//! - 零前瞻：輸出僅依賴當前及之前的輸入
//! - 使用環形緩衝區管理隱藏狀態，禁止 push/pop 觸發 realloc
//!
//! ## 環形緩衝區設計
//! - 固定容量 `Vec<f32>` + 頭尾指標
//! - 寫入時覆蓋最舊資料，不分配新記憶體
//! - 索引計算使用 `idx = (head + offset) % capacity`

use candle_core::{Device, Tensor};
use std::fmt;

use crate::Error;

// ---------------------------------------------------------------------------
// 環形緩衝區
// ---------------------------------------------------------------------------

/// 固定容量環形緩衝區（f32）
///
/// 用於儲存因果卷積的歷史狀態。
/// - 所有記憶體在 `new()` 時預分配
/// - 寫入操作永不觸發 realloc
pub struct RingBuffer {
    /// 底層儲存
    data: Vec<f32>,
    /// 容量（元素數）
    capacity: usize,
    /// 寫入位置（下一個元素寫入處）
    head: usize,
    /// 當前有效元素數
    len: usize,
}

impl RingBuffer {
    /// 建立新的環形緩衝區
    ///
    /// # 參數
    /// - `capacity`: 最大元素數
    ///
    /// # 恐慌
    /// 若 `capacity == 0` 則 panic
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        Self {
            data: vec![0.0_f32; capacity],
            capacity,
            head: 0,
            len: 0,
        }
    }

    /// 推入一個元素（覆蓋最舊資料）
    #[inline]
    pub fn push(&mut self, value: f32) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// 取得偏移量處的元素（0 = 最新）
    ///
    /// # 參數
    /// - `offset`: 向後偏移量。0 為最新元素，1 為前一個，依此類推。
    #[allow(dead_code)]
    #[inline]
    pub fn get(&self, offset: usize) -> Option<f32> {
        if offset >= self.len {
            return None;
        }
        // 最新元素在 head - 1（考慮環繞）
        let idx = if self.head >= offset + 1 {
            self.head - offset - 1
        } else {
            self.capacity - (offset + 1 - self.head)
        };
        Some(self.data[idx])
    }

    /// 以切片形式取得最近 N 個元素（按時間順序：最舊到最新）
    ///
    /// 使用自定義暫存避免分配
    pub fn last_n(&self, n: usize, out: &mut [f32]) -> Option<usize> {
        let n = n.min(self.len);
        if out.len() < n {
            return None;
        }

        // 從最舊到最新填充
        for i in 0..n {
            // 最舊元素位置
            let oldest = if self.len < self.capacity {
                0_usize
            } else {
                self.head
            };
            let idx = (oldest + i) % self.capacity;
            out[i] = self.data[idx];
        }
        Some(n)
    }

    /// 重置緩衝區（零分配）
    #[inline]
    pub fn reset(&mut self) {
        self.data.fill(0.0_f32);
        self.head = 0;
        self.len = 0;
    }

    /// 當前元素數
    #[allow(dead_code)]
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// 是否為空
    #[allow(dead_code)]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 容量
    #[allow(dead_code)]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl fmt::Debug for RingBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len)
            .field("head", &self.head)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 因果卷積層
// ---------------------------------------------------------------------------

/// 因果卷積層配置
#[derive(Debug, Clone)]
pub struct CausalConvConfig {
    /// 輸入通道數
    pub in_channels: usize,
    /// 輸出通道數
    pub out_channels: usize,
    /// 卷積核大小
    pub kernel_size: usize,
    /// 擴張率
    pub dilation: usize,
}

impl Default for CausalConvConfig {
    fn default() -> Self {
        Self {
            in_channels: 512,
            out_channels: 512,
            kernel_size: 3,
            dilation: 1,
        }
    }
}

/// 因果卷積層狀態（環形緩衝區）
pub struct CausalConvState {
    /// 每通道的環形緩衝區
    buffers: Vec<RingBuffer>,
    /// 當前幀計數器
    frame_count: usize,
}

impl CausalConvState {
    /// 建立新狀態
    ///
    /// # 參數
    /// - `num_channels`: 通道數
    /// - `kernel_size`: 卷積核大小
    /// - `capacity`: 環形緩衝區容量（通常為 kernel_size 的 2-3 倍）
    pub fn new(num_channels: usize, kernel_size: usize, capacity: usize) -> Self {
        let actual_cap = capacity.max(kernel_size);
        let buffers = (0..num_channels)
            .map(|_| RingBuffer::new(actual_cap))
            .collect();
        Self {
            buffers,
            frame_count: 0,
        }
    }

    /// 推入一幀資料（所有通道）
    #[inline]
    pub fn push_frame(&mut self, frame: &[f32]) {
        for (ch, &val) in self.buffers.iter_mut().zip(frame.iter()) {
            ch.push(val);
        }
        self.frame_count += 1;
    }

    /// 重置狀態（零分配）
    #[inline]
    pub fn reset(&mut self) {
        for buf in &mut self.buffers {
            buf.reset();
        }
        self.frame_count = 0;
    }

    /// 將指定通道的最近 n 個歷史值寫入 `out` slice（零分配）
    ///
    /// `out` 的長度決定了請求的元素數。回傳實際寫入的元素數。
    ///
    /// # 恐慌
    /// 當 `channel` 超出範圍時 panic（caller 應保證索引有效）
    pub fn fill_history(&self, channel: usize, out: &mut [f32]) -> usize {
        let buf = &self.buffers[channel];
        let n = out.len().min(buf.len());
        // last_n 僅在 out 太小時回傳 None — 這裡保證 out.len() >= n
        let _ = buf.last_n(n, out);
        n
    }
}

// ---------------------------------------------------------------------------
// 因果卷積層實作
// ---------------------------------------------------------------------------

/// 因果卷積 1D 層
pub struct CausalConv1d {
    /// 權重: (out_channels, in_channels, kernel_size)
    weight: Tensor,
    /// 偏置: (out_channels,)
    bias: Option<Tensor>,
    /// 配置
    config: CausalConvConfig,
    /// 裝置
    device: Device,
    /// 狀態（環形緩衝區）
    state: CausalConvState,
    /// 預分配步進緩衝區 (in_channels × kernel_size)，熱路徑零分配
    step_scratch: Vec<f32>,
}

impl CausalConv1d {
    /// 建立因果卷積層
    ///
    /// # 參數
    /// - `weight`: 卷積權重，形狀 (out_channels, in_channels, kernel_size)
    /// - `bias`: 可選偏置，形狀 (out_channels,)
    /// - `config`: 配置
    /// - `state_capacity`: 環形緩衝區容量
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        config: CausalConvConfig,
        state_capacity: usize,
    ) -> crate::Result<Self> {
        let device = weight.device().clone();
        // 預分配步進緩衝區：最多 kernel_size 幀 × in_channels 通道
        let step_scratch = vec![0.0_f32; config.in_channels * config.kernel_size];

        Ok(Self {
            state: CausalConvState::new(config.out_channels, config.kernel_size, state_capacity),
            weight,
            bias,
            config,
            device,
            step_scratch,
        })
    }

    /// 從 safetensors 載入權重
    pub fn from_safetensors(
        tensors: &std::collections::HashMap<String, Tensor>,
        prefix: &str,
        config: CausalConvConfig,
        state_capacity: usize,
    ) -> crate::Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let weight = tensors
            .get(&weight_key)
            .ok_or_else(|| Error::Weight(format!("Missing weight: {weight_key}")))?
            .clone();

        let bias_key = format!("{prefix}.bias");
        let bias = tensors.get(&bias_key).cloned();

        Self::new(weight, bias, config, state_capacity)
    }

    /// 前向傳播
    ///
    /// # 參數
    /// - `input`: 輸入張量，形狀 (batch, in_channels, time)
    ///
    /// # 回傳值
    /// 形狀 (batch, out_channels, time) 的輸出張量
    pub fn forward(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        // 近期 Candle API：使用 conv1d 進行因果卷積
        // 因果卷積透過在左側填充 (kernel_size - 1) 個零實現
        let pad = self.config.kernel_size - 1;

        let output = input.conv1d(&self.weight, pad, 1, self.config.dilation, 1)?;
        let output = if let Some(ref bias) = self.bias {
            let bias = bias.unsqueeze(0)?.unsqueeze(2)?;
            output.broadcast_add(&bias)?
        } else {
            output
        };

        Ok(output)
    }

    /// 處理單幀（流式推理用）— O(1) per step
    ///
    /// 與 `forward()` 不同，此方法使用內部環形緩衝區管理歷史狀態。
    /// 只傳入最近 `kernel_size` 幀到 conv1d，避免歷史累積造成的 O(n) 增長。
    ///
    /// # 參數
    /// - `frame`: 當前幀，形狀 (in_channels,)
    pub fn step(&mut self, frame: &[f32]) -> crate::Result<Vec<f32>> {
        let in_channels = self.config.in_channels;
        if frame.len() != in_channels {
            return Err(Error::Config(format!(
                "Expected frame of size {in_channels}, got {}",
                frame.len()
            )));
        }

        // 推入環形緩衝區
        self.state.push_frame(frame);
        let k = self.config.kernel_size;
        let n_frames = self.state.frame_count.min(k); // 最多 kernel_size 幀

        // 使用預分配步進緩衝區，避免熱路徑分配
        let total_len = in_channels * n_frames;
        let conv_input = &mut self.step_scratch[..total_len];

        for ch in 0..in_channels {
            let start = ch * n_frames;
            let end = start + n_frames;
            self.state.fill_history(ch, &mut conv_input[start..end]);
        }

        // 構建輸入張量: (1, in_channels, n_frames) — Tensor::from_slice 會複製資料
        let input_tensor =
            Tensor::from_slice(conv_input, (1, in_channels, n_frames), &self.device)?;

        let output = self.forward(&input_tensor)?;
        // Output: (1, out_channels, L_out)，L_out = n_frames + kernel_size - 1
        // 因果卷積：最新的輸入幀 (index n_frames-1) 對應輸出中相同位置
        let frame_pos = n_frames.saturating_sub(1);
        let frame_out = output.narrow(2, frame_pos, 1)?;
        frame_out
            .squeeze(0)?
            .squeeze(1)?
            .to_vec1()
            .map_err(Into::into)
    }

    /// 重置內部狀態
    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    // ------------------------------------------------------------------
    // 存取子
    // ------------------------------------------------------------------

    pub fn config(&self) -> &CausalConvConfig {
        &self.config
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl fmt::Debug for CausalConv1d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalConv1d")
            .field("config", &self.config)
            .finish()
    }
}

impl fmt::Debug for CausalConvState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalConvState")
            .field("frame_count", &self.frame_count)
            .field("buffer_count", &self.buffers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// 單元測試
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use candle_core::Device;

    use super::*;

    fn test_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_ring_buffer_basic() {
        let mut buf = RingBuffer::new(4);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);

        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some(3.0)); // 最新
        assert_eq!(buf.get(1), Some(2.0));
        assert_eq!(buf.get(2), Some(1.0));
        assert_eq!(buf.get(3), None);
    }

    #[test]
    fn test_ring_buffer_wraparound() {
        let mut buf = RingBuffer::new(4);
        for i in 0..6 {
            buf.push(i as f32);
        }
        // 現在內容: 4, 5, 2, 3 (head=2)
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.get(0), Some(5.0)); // 最新
        assert_eq!(buf.get(3), Some(2.0)); // 最舊
    }

    #[test]
    fn test_ring_buffer_last_n() {
        let mut buf = RingBuffer::new(8);
        for i in 0..5 {
            buf.push(i as f32);
        }
        let mut out = [0.0_f32; 5];
        let n = buf.last_n(5, &mut out).unwrap();
        assert_eq!(n, 5);
        assert_eq!(out, [0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_ring_buffer_reset() {
        let mut buf = RingBuffer::new(4);
        buf.push(1.0);
        buf.push(2.0);
        buf.reset();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.get(0), None);
    }

    #[test]
    fn test_causal_conv1d_creation() {
        let device = test_device();
        let weight = Tensor::zeros((512, 512, 3), candle_core::DType::F32, &device).unwrap();
        let config = CausalConvConfig::default();
        let conv = CausalConv1d::new(weight, None, config, 16).unwrap();
        assert_eq!(conv.config().kernel_size, 3);
    }

    #[test]
    fn test_causal_conv_state() {
        let mut state = CausalConvState::new(512, 3, 16);
        assert_eq!(state.buffers.len(), 512);
        state.push_frame(&vec![1.0; 512]);
        let mut buf = [0.0f32; 1];
        let n = state.fill_history(0, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf[0], 1.0);
    }

    /// 數值對齊測試：使用已知輸入比對 CausalConv1d::step 與 PyTorch 參考
    #[test]
    fn test_causal_conv_alignment_with_pytorch() {
        let weight_path = std::path::Path::new("weights/tokenizer/lightweight.safetensors");
        if !weight_path.exists() {
            eprintln!("Skipping: real weights not found");
            return;
        }

        let device = Device::Cpu;
        let sd =
            candle_core::safetensors::load(weight_path, &device).expect("load lightweight weights");
        let weight = sd.get("pre_conv.weight").expect("pre_conv.weight").clone();
        let bias = sd.get("pre_conv.bias").cloned();

        let config = CausalConvConfig {
            in_channels: 512,
            out_channels: 1024,
            kernel_size: 3,
            dilation: 1,
        };
        let mut conv = CausalConv1d::new(weight, bias, config, 16).expect("create conv");

        // 輸入: 0..512 * 0.01
        let input: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();

        // Step 1: 1 frame
        let rust_step1 = conv.step(&input).expect("step1");
        assert_eq!(rust_step1.len(), 1024);

        // Step 2: 2 frames (same input = statefulness check)
        let rust_step2 = conv.step(&input).expect("step2");
        assert_eq!(rust_step2.len(), 1024);

        // 裝載 PyTorch 參考
        let ref_path1 = std::path::Path::new("weights/refs2/pre_conv_step1_ref.bin");
        let ref_path2 = std::path::Path::new("weights/refs2/pre_conv_step2_ref.bin");

        if ref_path1.exists() && ref_path2.exists() {
            let ref_bytes1 = std::fs::read(ref_path1).expect("read ref1");
            let ref_bytes2 = std::fs::read(ref_path2).expect("read ref2");
            let ref_step1: Vec<f32> = ref_bytes1
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let ref_step2: Vec<f32> = ref_bytes2
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            // 計算餘弦相似度
            let dot1: f32 = rust_step1
                .iter()
                .zip(ref_step1.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm1: f32 = rust_step1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_ref1: f32 = ref_step1.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos1 = dot1 / (norm1 * norm_ref1 + 1e-10);

            let dot2: f32 = rust_step2
                .iter()
                .zip(ref_step2.iter())
                .map(|(a, b)| a * b)
                .sum();
            let norm2: f32 = rust_step2.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_ref2: f32 = ref_step2.iter().map(|x| x * x).sum::<f32>().sqrt();
            let cos2 = dot2 / (norm2 * norm_ref2 + 1e-10);

            println!("Step1 cosine sim: {cos1:.8}");
            println!("Step2 cosine sim: {cos2:.8}");

            let mse1: f32 = rust_step1
                .iter()
                .zip(ref_step1.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / rust_step1.len() as f32;
            println!("Step1 MSE: {mse1:.10}");

            assert!(cos1 > 0.999, "Step1 cosine = {cos1} < 0.999");
            assert!(cos2 > 0.999, "Step2 cosine = {cos2} < 0.999");
            assert!(mse1 < 1e-4, "Step1 MSE = {mse1} >= 1e-4");
        } else {
            eprintln!("Skipping alignment check: reference files not found");
        }
    }
}
