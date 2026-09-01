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
///
/// **注意：** `CausalConvState` 現在使用扁平陣列替代此 struct；
/// `RingBuffer` 保留僅供獨立單元測試使用。
#[allow(dead_code)]
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

#[allow(dead_code)]
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
        let idx = if self.head > offset {
            self.head - offset - 1
        } else {
            self.capacity - (offset + 1 - self.head)
        };
        Some(self.data[idx])
    }

    /// 以切片形式取得最近 N 個元素（按時間順序：最舊到最新）
    ///
    /// # 重要
    /// 當 `n < self.len` 時，回傳 **最後** n 個元素（最接近最新的），
    /// 而不是前 n 個。這與因果卷積的需求一致：只需要最近 `kernel_size` 幀。
    ///
    /// 使用自定義暫存避免分配
    pub fn last_n(&self, n: usize, out: &mut [f32]) -> Option<usize> {
        let n = n.min(self.len);
        if out.len() < n {
            return None;
        }

        // 從最舊到最新填充（最近 n 個元素）
        let oldest = if self.len < self.capacity {
            // 緩衝區未滿：有效資料在 data[0..len]，
            // 最近 n 個從 data[len-n] 開始
            self.len - n
        } else {
            // 緩衝區已滿：有效資料環繞，最舊的頭部在 self.head，
            // 最近 n 個從 (head + capacity - n) 開始
            (self.head + self.capacity - n) % self.capacity
        };

        for i in 0..n {
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
    /// 分組數（1 = 標準卷積, out_channels = depthwise）
    pub groups: usize,
}

impl Default for CausalConvConfig {
    fn default() -> Self {
        Self {
            in_channels: 512,
            out_channels: 512,
            kernel_size: 3,
            dilation: 1,
            groups: 1,
        }
    }
}

impl CausalConvConfig {
    /// 從權重張量維度推導配置（便於 CausalConvNet → CausalConv1d 遷移）
    pub fn from_weight(weight: &Tensor, dilation: usize, groups: usize) -> Self {
        let d = weight.dims();
        Self {
            in_channels: d[1] * groups,
            out_channels: d[0],
            kernel_size: d[2],
            dilation,
            groups,
        }
    }
}

/// 扁平化環形緩衝區狀態（因果卷積用）
///
/// 所有通道共享同一個 head/len，儲存在連續的 `[num_channels × capacity]` 陣列中。
/// 與舊版 `Vec<RingBuffer>` 相比：
/// - 單次 heap 分配（而非 512 次）
/// - 通道間連續儲存，fill_history 時 cache-friendly
/// - reset 為單次 `fill(0.0)`（而非 512 次迴圈）
pub struct CausalConvState {
    /// 扁平資料: num_channels × capacity, row-major [channel][offset]
    data: Vec<f32>,
    /// 通道數
    num_channels: usize,
    /// 環形緩衝區容量
    capacity: usize,
    /// 下一個寫入位置 (0..capacity)
    head: usize,
    /// 當前有效幀數 (0..capacity)
    len: usize,
}

impl CausalConvState {
    /// 建立新狀態
    ///
    /// # 參數
    /// - `num_channels`: 通道數
    /// - `_kernel_size`: 卷積核大小（用於確保容量不小於核大小）
    /// - `capacity`: 環形緩衝區容量（通常為 kernel_size 的 2-3 倍）
    pub fn new(num_channels: usize, _kernel_size: usize, capacity: usize) -> Self {
        let cap = capacity.max(_kernel_size);
        Self {
            data: vec![0.0_f32; num_channels * cap],
            num_channels,
            capacity: cap,
            head: 0,
            len: 0,
        }
    }

    /// 推入一幀資料（所有通道）
    #[inline]
    pub fn push_frame(&mut self, frame: &[f32]) {
        let cap = self.capacity;
        let head = self.head;
        for (ch, &val) in frame.iter().enumerate() {
            self.data[ch * cap + head] = val;
        }
        self.head = (head + 1) % cap;
        if self.len < cap {
            self.len += 1;
        }
    }

    /// 重置狀態（零分配）
    #[inline]
    pub fn reset(&mut self) {
        self.data.fill(0.0_f32);
        self.head = 0;
        self.len = 0;
    }

    /// 將指定通道的最近 n 個歷史值寫入 `out` slice（零分配）
    ///
    /// `out` 的長度決定了請求的元素數。回傳實際寫入的元素數。
    ///
    /// # 恐慌
    /// 當 `channel` 超出範圍時 panic（caller 應保證索引有效）
    pub fn fill_history(&self, channel: usize, out: &mut [f32]) -> usize {
        let n = out.len().min(self.len);
        if n == 0 {
            return 0;
        }

        let oldest = if self.len < self.capacity {
            self.len - n
        } else {
            (self.head + self.capacity - n) % self.capacity
        };

        let base = channel * self.capacity;
        let end = oldest + n;

        if end <= self.capacity {
            // 連續情況：單次 memcpy
            out[..n].copy_from_slice(&self.data[base + oldest..base + end]);
        } else {
            // 環繞情況：尾部 + 頭部
            // 注意：務必限制到 base + self.capacity，而非到 data 尾部！
            let first_part = self.capacity - oldest;
            out[..first_part].copy_from_slice(&self.data[base + oldest..base + self.capacity]);
            let second_part = n - first_part;
            out[first_part..n].copy_from_slice(&self.data[base..base + second_part]);
        }
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
    /// 預先建立的 3D 偏置: (1, out_channels, 1)
    bias_3d: Option<Tensor>,
    /// 配置
    config: CausalConvConfig,
    /// 裝置
    device: Device,
    /// 純設備端歷史狀態張量: (1, in_channels, left_pad)
    state_tensor: Tensor,
    /// 左側填充大小 = (kernel_size - 1) * dilation
    left_pad: usize,
}

impl CausalConv1d {
    /// 建立因果卷積層
    ///
    /// # 參數
    /// - `weight`: 卷積權重，形狀 (out_channels, in_channels, kernel_size)
    /// - `bias`: 可選偏置，形狀 (out_channels,)
    /// - `config`: 配置
    /// - `_state_capacity`: 歷史容量（保留相容性）
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        config: CausalConvConfig,
        _state_capacity: usize,
    ) -> crate::Result<Self> {
        let device = weight.device().clone();
        let left_pad = (config.kernel_size - 1) * config.dilation;
        let state_tensor = Tensor::zeros(
            (1, config.in_channels, left_pad),
            weight.dtype(),
            &device,
        )?;
        let bias_3d = if let Some(ref b) = bias {
            Some(b.reshape((1, b.elem_count(), 1))?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            bias_3d,
            config,
            device,
            state_tensor,
            left_pad,
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

    /// 前向傳播（左側填充，輸出長度 = 輸入長度）
    ///
    /// 使用 `pad_with_zeros` + `conv1d(pad=0)` 實現純左側因果填充，
    /// 支援 dilation 與 groups。不同於 `step()` 的單幀模式，
    /// 此方法適用於完整序列的批次推理。
    ///
    /// # 參數
    /// - `input`: 輸入張量，形狀 (batch, in_channels, time)
    ///
    /// # 回傳值
    /// 形狀 (batch, out_channels, time) 的輸出張量
    pub fn forward(&self, input: &Tensor) -> crate::Result<Tensor> {
        let k = self.config.kernel_size;
        let d = self.config.dilation;
        let groups = self.config.groups;
        // 左側填充量 = (kernel_size - 1) * dilation，使 conv1d 輸出長度 = 輸入長度
        let left_pad = (k - 1) * d;
        let padded = input.pad_with_zeros(2, left_pad, 0)?;
        let output = padded.conv1d(&self.weight, 0, 1, d, groups)?;
        if let Some(ref bias_3d) = self.bias_3d {
            Ok(output.broadcast_add(bias_3d)?)
        } else if let Some(ref bias) = self.bias {
            let b = bias.unsqueeze(0)?.unsqueeze(2)?;
            Ok(output.broadcast_add(&b)?)
        } else {
            Ok(output)
        }
    }

    /// 處理單幀（流式推理用）— O(1) per step
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

        let input_tensor = Tensor::from_slice(frame, (1, in_channels, 1), &self.device)?;
        let output = self.step_tensor(&input_tensor)?;
        output
            .squeeze(0)?
            .squeeze(1)?
            .to_vec1()
            .map_err(Into::into)
    }

    /// **純設備端串流步進** — 零跨設備複製、零堆配置
    ///
    /// # 參數
    /// - `input`: 當前幀張量，形狀 `(1, in_channels, t_in)`, `(in_channels, t_in)` 或 `(in_channels,)`
    ///
    /// # 回傳值
    /// 形狀 `(1, out_channels, t_in)` 的輸出張量
    pub fn step_tensor(&mut self, input: &Tensor) -> crate::Result<Tensor> {
        let in_channels = self.config.in_channels;

        let x_3d = match input.dims() {
            &[1, c, _] if c == in_channels => input.clone(),
            &[b, c, _] if b == 1 && c == in_channels => input.clone(),
            &[c, _] if c == in_channels => input.unsqueeze(0)?,
            &[c] if c == in_channels => input.unsqueeze(0)?.unsqueeze(2)?,
            shape => {
                return Err(Error::Config(format!(
                    "Expected input with in_channels={in_channels}, got shape {shape:?}"
                )));
            }
        };

        let t_in = x_3d.dim(2)?;

        let output = if self.left_pad == 0 {
            x_3d.conv1d(&self.weight, 0, 1, self.config.dilation, self.config.groups)?
        } else {
            let full_x = Tensor::cat(&[&self.state_tensor, &x_3d], 2)?;
            self.state_tensor = full_x.narrow(2, t_in, self.left_pad)?.contiguous()?;
            full_x.conv1d(&self.weight, 0, 1, self.config.dilation, self.config.groups)?
        };

        if let Some(ref b3d) = self.bias_3d {
            Ok(output.broadcast_add(b3d)?)
        } else {
            Ok(output)
        }
    }

    /// 重置內部狀態（零分配）
    pub fn reset_state(&mut self) {
        if self.left_pad > 0 {
            if let Ok(zeros) = Tensor::zeros(
                (1, self.config.in_channels, self.left_pad),
                self.weight.dtype(),
                &self.device,
            ) {
                self.state_tensor = zeros;
            }
        }
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
            .field("num_channels", &self.num_channels)
            .field("capacity", &self.capacity)
            .field("head", &self.head)
            .field("len", &self.len)
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

    /// 關鍵回歸測試：last_n 在 n < len 時必須回傳最後 n 個元素，而非前 n 個。
    /// 先前 bug：`oldest` 始終為 0，導致回傳 data[0..n] 而非 data[len-n..len]。
    /// 這直接影響 step_tensor：當 frame_count > kernel_size 時餵錯歷史給 conv。
    #[test]
    fn test_ring_buffer_last_n_partial() {
        let mut buf = RingBuffer::new(8);
        // 推入 5 個元素
        for i in 0..5 {
            buf.push(i as f32);
        }
        // 請求最後 3 個：應為 [2.0, 3.0, 4.0]
        let mut out = [0.0_f32; 3];
        let n = buf.last_n(3, &mut out).unwrap();
        assert_eq!(n, 3);
        assert_eq!(
            out,
            [2.0, 3.0, 4.0],
            "last_n(3) should return the 3 most recent (indices 2..5), got {out:?}"
        );
    }

    /// 纏繞模式下 last_n 正確性（已滿緩衝區）
    #[test]
    fn test_ring_buffer_last_n_wrapped() {
        let mut buf = RingBuffer::new(4);
        for i in 0..6 {
            buf.push(i as f32);
        }
        // 已推入 6 個，capacity=4，所以僅保留 [2, 3, 4, 5]，head=2
        assert_eq!(buf.len, 4);
        // 最後 2 個：應為 [4.0, 5.0]
        let mut out = [0.0_f32; 2];
        let n = buf.last_n(2, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            out,
            [4.0, 5.0],
            "wrapped last_n(2) should return [4.0, 5.0], got {out:?}"
        );
        // 全部 4 個：應為 [2.0, 3.0, 4.0, 5.0]
        let mut out2 = [0.0_f32; 4];
        let n2 = buf.last_n(4, &mut out2).unwrap();
        assert_eq!(n2, 4);
        assert_eq!(out2, [2.0, 3.0, 4.0, 5.0], "wrapped last_n(4) failed");
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
        assert_eq!(state.num_channels, 512);
        assert_eq!(state.capacity, 16);
        state.push_frame(&vec![1.0; 512]);
        let mut buf = [0.0f32; 1];
        let n = state.fill_history(0, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf[0], 1.0);
        // 驗證 fill_history wraparound 正確性：推滿後再填應回傳最新值
        for _ in 0..20 {
            state.push_frame(&vec![2.0; 512]);
        }
        let mut buf2 = [0.0f32; 3];
        let n2 = state.fill_history(0, &mut buf2);
        assert_eq!(n2, 3);
        assert_eq!(buf2, [2.0, 2.0, 2.0]);
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
            groups: 1,
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

    #[test]
    fn test_causal_conv_step_tensor_matches_forward() {
        let device = test_device();
        let config = CausalConvConfig {
            in_channels: 8,
            out_channels: 16,
            kernel_size: 5,
            dilation: 2,
            groups: 1,
        };
        let weight = Tensor::randn(0.0f32, 1.0f32, (16, 8, 5), &device).unwrap();
        let bias = Some(Tensor::randn(0.0f32, 1.0f32, (16,), &device).unwrap());
        let mut conv = CausalConv1d::new(weight, bias, config, 16).unwrap();

        let num_frames = 10;
        let input_data: Vec<f32> = (0..8 * num_frames)
            .map(|i| ((i * 17) % 31) as f32 * 0.1)
            .collect();
        let full_input = Tensor::from_slice(&input_data, (1, 8, num_frames), &device).unwrap();

        // Batch forward
        let batch_out = conv.forward(&full_input).unwrap();
        let batch_flat: Vec<f32> = batch_out.flatten_all().unwrap().to_vec1().unwrap();

        // Streaming step_tensor frame by frame
        let mut stream_outputs = Vec::new();
        for t in 0..num_frames {
            let frame = full_input.narrow(2, t, 1).unwrap();
            let out_t = conv.step_tensor(&frame).unwrap();
            assert_eq!(out_t.shape().dims(), &[1, 16, 1]);
            let frame_flat: Vec<f32> = out_t.flatten_all().unwrap().to_vec1().unwrap();
            stream_outputs.push(frame_flat);
        }

        // Reconstruct channel-major [1, 16, 10]
        let mut stream_flat = Vec::with_capacity(16 * num_frames);
        for ch in 0..16 {
            for t in 0..num_frames {
                stream_flat.push(stream_outputs[t][ch]);
            }
        }

        let max_diff: f32 = batch_flat
            .iter()
            .zip(stream_flat.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-5,
            "CausalConv1d step_tensor must match forward: max_diff = {max_diff}"
        );

        // Test reset_state
        conv.reset_state();
        let first_frame = full_input.narrow(2, 0, 1).unwrap();
        let first_step_out = conv.step_tensor(&first_frame).unwrap();
        let first_flat: Vec<f32> = first_step_out.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(first_flat, stream_outputs[0], "State reset must reproduce first frame output");
    }
}
