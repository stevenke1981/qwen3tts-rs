# Qwen3-TTS Rust Rewrite Implementation Plan

## Phase 0.5: 文本前端 LLM 整合 (Week 1-2)
**目標**: 建立文字→語義 Token 的 Rust 整合管線，實現端到端 TTS
- [ ] 定義 `TextFrontend` trait、`TokenStream`、`SynthesisOptions` 等型別
- [ ] 實作 `TokenParser`：將 LLM 輸出解析為 16×u16 每幀的格式
- [ ] 實作 `PythonBridge` 後端：透過子行程調用 qwen-tts Python 套件
  - Python 腳本接收文字，呼叫 `model.generate()` 取得 `talker_codes`
  - 以二進位 stdout 輸出，Rust 端以 `read_exact` 高效讀取
- [ ] 實作 `CandleNative` 後端骨架（stub，未來 GGUF 推理）
- [ ] 重寫 `examples/synthesize.rs`：支援文字輸入 → 語音輸出
- [ ] 整合測試：文字 → 解碼器 → WAV 端到端驗證
- [ ] **里程碑**: `cargo run --example synthesize -- --text "你好世界"` 可聽懂

## Phase 0: 前置验证与基础设施 (Week 1)
**目标**: 确认技术可行性，搭建开发基线
- [ ] 导出 Qwen3-TTS 完整计算图，逐一核对 Candle 算子支持情况
- [ ] **GGUF 可行性评估**: 确认文本前端 LLM 是否复用 Qwen3 LLM；若是，评估 llama.cpp / candle-lm 加载 GGUF 的路由方案
- [ ] 编写 Python 权重转换脚本（safetensors → Candle 兼容格式）
- [ ] 搭建自动化数值对齐测试框架（PyTorch vs Candle 逐层比对）
- [ ] 提取并验证 `tokenizer.json` 完整性，编写 Rust 侧 Tokenizer 加载器
- [ ] **里程碑**: 产出《算子兼容性报告》与可运行的对齐测试脚手架

## Phase 1: 12Hz 实时解码器核心 (Week 2-3)
**目标**: 实现低延迟多码本解码管线
- [ ] 实现扁平化多码本 Tensor 数据结构与 Embedding Lookup 模块
- [ ] 集成 rayon 实现 16 层并行解码
- [ ] 实现因果卷积网络（Causal ConvNet）及环形缓冲区状态管理
- [ ] 实现 MTP 模块声学码本生成逻辑
- [ ] 完成 12Hz 解码器数值对齐测试（余弦相似度 ≥ 0.999）
- [ ] **里程碑**: 12Hz 解码器可离线生成正确音频，精度达标

## Phase 2: 25Hz 高质量解码器 (Week 4-5)
**目标**: 实现 Flow Matching DiT 解码
- [ ] 实现 Block-wise Flow Matching ODE Solver
- [ ] 实现 DiT 主干网络（RoPE, RMSNorm, FlashAttention）
- [ ] 实现分块上下文管理与前瞻缓冲逻辑
- [ ] 完成 25Hz 解码器数值对齐测试
- [ ] **里程碑**: 25Hz 解码器可离线生成正确音频，PESQ 差异 ≤ 0.05

## Phase 3: 声码器移植与端到端集成 (Week 6)
**目标**: 补全音频重建链路，实现完整 TTS 管线
- [ ] 分析原版声码器架构，在 Candle 中手写实现或适配现有 DSP 库
- [ ] 集成 Tokenizer → Decoder → Vocoder 完整管线
- [ ] 实现流式 `decode_chunk` 接口与异步 I/O 绑定
- [ ] 端到端音频质量主观听测与客观指标验证
- [ ] **里程碑**: 可流式输出可听语音，首包延迟初步达标

## Phase 4: 性能优化与生产加固 (Week 7-8)
**目标**: 达成所有非功能性指标
- [ ] GPU Kernel 调优（CUDA/Metal 自定义算子融合）
- [ ] 实现自定义量化校准管线：码本专属 INT8 校准、逐层卷积敏感度分析、量化对齐验证 (cosine ≥ 0.995)
- [ ] 内存 Profile 与分配热点消除（确保零动态扩容）
- [ ] 压力测试：1000 条文本回归 + 长时间运行稳定性测试
- [ ] 容错机制验证（模拟单层异常、非法 Token 输入）
- [ ] 编译优化（LTO, PGO, strip）与二进制体积压缩
- [ ] **里程碑**: 全部验收标准通过，发布 v0.1.0

## 风险登记册
| 风险项 | 影响等级 | 缓解措施 |
| :--- | :--- | :--- |
| Candle 缺失 Flow Matching 关键算子 | 高 | Phase 0 提前验证；预留手写 CUDA Kernel 时间 |
| 多码本并行 FFI 开销超预期 | 中 | 已选纯 Rust Candle；Phase 1 早期做微基准测试 |
| 声码器复刻音质劣化 | 高 | Phase 3 预留 1 周缓冲；必要时回退 tch-rs 仅用于声码器 |
| 数值对齐失败 | 高 | 每层独立对齐，定位问题粒度到单个算子；保留 PyTorch 调试钩子 |
| GGUF 格式错配导致工程浪费 | 中 | 核心解码器禁用 GGUF，仅限文本前端 LLM 子模块使用；量化方案已在 spec.md 3.5 明确 |

## 交付物清单
- [ ] `qwen3-tts-rs` Cargo Workspace 源码
- [ ] 权重转换 Python 工具集
- [ ] 数值对齐测试套件与报告
- [ ] 性能基准测试报告（含首包延迟、吞吐量、内存曲线）
- [ ] API 文档与集成示例
