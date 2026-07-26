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

---

# qwentts.cpp Full Alignment Roadmap

本 Roadmap 合併自 full-alignment devpack，作為目前 P00→P13 的主要執行順序。
上方既有計畫完整保留作為歷史技術脈絡；若順序或範圍衝突，以本 Roadmap、
`tasks/task-index.yaml` 與 Gate 文件為準。

## External-Agent Execution Model (P03-T05 → P13)

剩餘 Roadmap 採「外部代理實作、GPT-5.6 Sol 驗收」模式。外部代理可以是
DeepSeek V4 Flash 或主人另行指定的模型；模型名稱不改變任務契約。

| 階段 | External Implementer | GPT-5.6 Sol |
|---|---|---|
| Discover | 讀取指定 task card、契約、測試與既有證據 | 用 CBM 與真實 repo 建立 task card、限制範圍 |
| Implement | 僅修改 allowed files，test-first，執行局部驗證 | 不代替外部報告作出通過判定 |
| Report | 寫 `worker-report.md`、命令與原始結果 | 檢查實際 diff，不信任摘要或宣稱 |
| Review | 不可自我驗收或調低門檻 | 親自重跑必要測試、檢查 fixture、數值、錯誤與效能 |
| Gate | 不可更新 Gate、STATUS、TODOS、commit 或 push | 寫 `review.md`/`gate.json`，更新索引，接受後才 commit/push |

每次只能交付一個 `Pxx-Tyy` task card。Sol 必須先建立：

```text
artifacts/alignment/<phase>/<task>/assignment.md
```

外部代理完成後至少交付：

```text
worker-report.md
commands.txt
test-results.txt
```

外部代理回報「全部測試通過」只視為待驗證聲明。Sol 必須確認沒有編譯失敗、
ignored/skip、fixture 缺失、`continue-on-error`、門檻弱化、漏報 transfer、
API 相容性或 backend-specific 安全問題，才可建立：

```text
review.md
gate.json
```

工作循環固定為：

```text
SOL DISCOVER/SPECIFY → EXTERNAL TEST-FIRST/IMPLEMENT/LOCAL VERIFY
→ SOL DIFF REVIEW/INDEPENDENT VERIFY → SOL GATE/DOCUMENT/COMMIT/PUSH
```

各 Phase 的技術任務順序維持不變。P03-T05 至 P13-T05 預設由 External
Implementer 實作；每個 task Gate、每個 Phase Gate 與最終 `ALIGNED`
判定皆由 Sol 持有。通用契約與模板位於
`docs/alignment/agent-workflow/`。

## P00 — Baseline and parity harness

1. Pin target/reference/upstream revisions and write baseline delta report (`P00-T01`)
2. Create fixture manifest and fail-closed fixture resolver (`P00-T02`)
3. Add stage dump format and dump hooks behind a feature flag (`P00-T03`)
4. Build reference command adapters for Python and qwentts.cpp (`P00-T04`)
5. Create CPU F32 smoke corpus and baseline report (`P00-T05`)

## P01 — Metadata and prompt parity

1. Replace model-name inference with metadata/config parsing (`P01-T01`)
2. Load all special token, language, speaker and dialect tables from metadata (`P01-T02`)
3. Correct M-RoPE semantics and add exact position/rotation tests (`P01-T03`)
4. Match prompt assembly for Base, CustomVoice and VoiceDesign (`P01-T04`)
5. Gate exact prompt IDs across the model matrix (`P01-T05`)

## P02 — Sampling parity

1. Implement Philox RNG with known-vector tests (`P02-T01`)
2. Implement repetition penalty with exact operation ordering (`P02-T02`)
3. Separate Talker and Code Predictor sampling configs (`P02-T03`)
4. Match token suppression and EOS handling (`P02-T04`)
5. Gate deterministic token sequence parity (`P02-T05`)

## P03 — Talker and Code Predictor numerical parity

1. Instrument embeddings, norms, RoPE and layer outputs (`P03-T01`)
2. Verify Talker prefill and single-step KV cache (`P03-T02`)
3. Verify Code Predictor frame-local prefill and 14 decode steps (`P03-T03`)
4. Eliminate avoidable host transfers in acoustic prediction (`P03-T04`)
5. Gate stage cosine and logit ranking thresholds (`P03-T05`)

## P04 — Tokenizer decoder offline parity

1. Verify RVQ split projections and codebook policy (`P04-T01`)
2. Verify decoder transformer and sliding-window semantics (`P04-T02`)
3. Verify ConvNeXt upsample and DAC blocks (`P04-T03`)
4. Match offline waveform on short/medium/long corpora (`P04-T04`)
5. Create buffered chunk decode with left-context trimming (`P04-T05`)

## P05 — True stateful codec streaming

1. Define CodecStreamState and state ownership (`P05-T01`)
2. Implement persistent causal-convolution contexts (`P05-T02`)
3. Implement transposed-convolution overlap state (`P05-T03`)
4. Implement transformer KV ring and absolute RoPE position (`P05-T04`)
5. Implement one-frame graph/buffer reuse (`P05-T05`)
6. Implement reset, ICL prime and optional state snapshots (`P05-T06`)
7. Gate offline-equivalent output and bounded complexity (`P05-T07`)

## P06 — End-to-end generation streaming

1. Expose frame events from Talker generation (`P06-T01`)
2. Connect generated codes directly to codec stream session (`P06-T02`)
3. Add audio callback, backpressure and cancellation (`P06-T03`)
4. Update GUI to consume streaming events (`P06-T04`)
5. Gate TTFA-before-completion and long-form output (`P06-T05`)

## P07 — Tokenizer encoder and qwen-codec

1. Complete 24 kHz audio preprocessing/resampling contract (`P07-T01`)
2. Verify SEANet and encoder transformer (`P07-T02`)
3. Implement RVQ encode argmin path (`P07-T03`)
4. Define versioned RVQ code file format (`P07-T04`)
5. Implement qwen-codec encode/decode/stream CLI (`P07-T05`)
6. Gate round-trip and reference code parity (`P07-T06`)

## P08 — Native quantized runtime

1. Define quantized tensor types and protected tensor policy (`P08-T01`)
2. Implement direct Q8 linear and embedding operations (`P08-T02`)
3. Implement Q4_K_M-class block layout and kernels (`P08-T03`)
4. Add backend-resident packed weight loader (`P08-T04`)
5. Integrate quantized Talker and Code Predictor (`P08-T05`)
6. Gate memory, token and quality metrics (`P08-T06`)

## P09 — Model format and conversion

1. Define metadata-complete Rust model container strategy (`P09-T01`)
2. Implement GGUF reader compatibility or lossless converter (`P09-T02`)
3. Implement official checkpoint conversion with provenance (`P09-T03`)
4. Implement quantization command and protected tensor rules (`P09-T04`)
5. Gate five talker variants plus shared tokenizer (`P09-T05`)

## P10 — CLI and library product surface

1. Implement stable high-level Rust synthesis/session API (`P10-T01`)
2. Implement qwen-tts CLI with streaming and WAV output (`P10-T02`)
3. Add model discovery/download and cache policy (`P10-T03`)
4. Add structured logs, JSON metrics and exit codes (`P10-T04`)
5. Gate compatibility corpus and cancellation (`P10-T05`)

## P11 — OpenAI server and voice registry

1. Implement `/v1/audio/speech` non-streaming endpoint (`P11-T01`)
2. Implement chunked/streaming audio response (`P11-T02`)
3. Implement safe cloned-voice registry (`P11-T03`)
4. Implement request limits, cancellation and error mapping (`P11-T04`)
5. Gate API and concurrency tests (`P11-T05`)

## P12 — C ABI and continuous batching

1. Define stable opaque-handle C ABI (`P12-T01`)
2. Implement callback ownership and cancellation (`P12-T02`)
3. Implement per-session scheduler state (`P12-T03`)
4. Implement bounded multi-lane Talker/Predictor batching (`P12-T04`)
5. Implement per-slot codec streams and isolation (`P12-T05`)
6. Gate C lifecycle, 8-session isolation and throughput (`P12-T06`)

## P13 — Backends, CI and release

1. CPU/CUDA/Metal full matrix and performance report (`P13-T01`)
2. Vulkan/ROCm strategy and implementation gate (`P13-T02`)
3. Replace permissive CI with fail-closed parity workflow (`P13-T03`)
4. Generate SBOM, notices, model manifest and reproducible build notes (`P13-T04`)
5. Run final audit and produce alignment release report (`P13-T05`)

## Sequencing Rule

P00→P08 are core correctness/runtime prerequisites. P10–P12 must not mask incomplete core
parity. External agents may not skip dependencies or work around a failed Gate.
P13 is the only final release gate, and only Sol may set the overall status to
`ALIGNED`.
