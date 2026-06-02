# Qwen3-TTS Rust Rewrite Technical Specification

## 1. 项目概述
本项目旨在将 Qwen3-TTS 语音合成模型的核心 Codec Decode 模块从 Python/PyTorch 移植到纯 Rust 环境。重写后的模块需支持 12Hz（实时交互）与 25Hz（高质量合成）双模式解码，重点解决多码本并行处理、流式因果卷积推理及低延迟首包输出等核心工程挑战。

## 2. 技术栈约束
- **语言**: Rust (Edition 2021+)
- **张量后端**: Candle (优先 CUDA/Metal GPU 加速，CPU 作为 Fallback)
- **并发框架**: rayon (多码本并行), tokio (异步流式 I/O)
- **序列化**: serde / safetensors-rs (权重加载)
- **音频处理**: hound / symphonia (WAV/PCM 编解码)
- **GGUF 使用范围**: 核心解码器（Causal ConvNet、Flow Matching DiT、码本Embedding、声码器）**禁用** GGUF；仅文本前端 LLM 子模块（如 Qwen3 LLM 韵律预测器）可选用 GGUF Q4_K_M 加载
- **禁止依赖**: libtorch, PyTorch, Python FFI, ONNX Runtime

## 3. 核心功能规格

### 3.1 双模式 Tokenizer 解码
| 模式 | Tokenizer | 解码器架构 | 延迟目标 | 关键特性 |
| :--- | :--- | :--- | :--- | :--- |
| 实时交互 | 12Hz | Causal ConvNet + MTP | 首包 ≤97ms | 16层独立码本并行，零前瞻 |
| 高质量合成 | 25Hz | Block-wise Flow Matching DiT | 首包 ≤300ms | 单级Token预测，分块扩散 |

### 3.2 多码本处理规范 (12Hz 模式)
- **内存布局**: 采用扁平化一维连续内存 `Vec<u16>`，索引公式 `idx = layer * num_frames + frame`。
- **并行策略**: 16层量化器 Embedding Lookup 必须通过 `rayon::par_iter` 并行执行。
- **查表优化**: 预加载 2048 大小码本的权重矩阵，使用指针偏移直接读取，禁用条件分支。
- **状态管理**: 因果卷积隐藏状态使用固定大小环形缓冲区（Ring Buffer），禁止运行时动态扩容。

### 3.3 流式推理接口
```rust
pub trait TtsDecoder: Send + Sync {
    /// 初始化解码器，加载权重与声码器
    fn new(config: DecoderConfig) -> Result<Self>;

    /// 流式输入 Token，返回 PCM 音频片段
    fn decode_chunk(&mut self, tokens: &[u16]) -> Result<Vec<f32>>;

    /// 重置内部状态（用于会话切换）
    fn reset_state(&mut self);
}
```

### 3.5 权重量化规格

| 组件 | 推荐格式 | 量化策略 | 理由 |
| :--- | :--- | :--- | :--- |
| 12Hz Causal ConvNet | safetensors + Candle 原生 INT8/FP16 | 逐层手动校准 | 卷积核对量化敏感，需保留部分 FP16 锚点层 |
| 16层码本 Embedding | safetensors + 自定义 INT8 | 码本专属校准数据集 | 避免通用量化策略破坏离散语义空间 |
| 25Hz DiT 主干 | safetensors + W4A16 / W8A8 | 类似 LLM 的混合量化 | DiT 结构与 LLM 相似，可借鉴成熟方案 |
| 声码器 | FP32 / FP16（不量化） | — | 声码器对精度极度敏感，量化收益极低 |
| 文本前端 LLM（可选） | GGUF Q4_K_M | llama.cpp 标准量化 | 仅此子模块可用 GGUF，因社区已有成熟工具链 |

所有量化权重必须通过数值对齐测试（余弦相似度 ≥ 0.995）方可合入主线。

### 3.6 精度与鲁棒性
- 数值对齐: 与 PyTorch 原版逐层中间张量余弦相似度 ≥ 0.999。
- 容错降级: 单层码本解码异常时，自动使用上层均值填充，不中断整体推理。
- Tokenizer 配置: 必须从官方 tokenizer.json 完整加载预分词器规则，禁止硬编码。

## 4. 非功能性要求
- 二进制体积: 生产构建 ≤ 50MB (GPU) / ≤ 30MB (CPU)
- 冷启动时间: ≤ 200ms (不含权重加载)
- 内存占用: 12Hz 模式峰值 ≤ 1.5GB VRAM / 3GB RAM
- 平台支持: Linux x86_64 (CUDA), macOS ARM64 (Metal), Windows x86_64 (CUDA/CPU)

## 5. 验收标准
- 12Hz 模式下，连续生成 10 秒音频，首包延迟 P99 ≤ 97ms。
- 25Hz 模式下，生成音频 PESQ 评分与 Python 原版差异 ≤ 0.05。
- 通过 1000 条不同长度文本的端到端回归测试，无崩溃、无静音段。
- `cargo test --release` 全部通过，含数值对齐单元测试。
