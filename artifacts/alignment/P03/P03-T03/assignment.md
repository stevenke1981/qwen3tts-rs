# P03-T03 Assignment — Code Predictor Frame-Local Parity

## Outcome

Prove and, where necessary, correct the production Candle Code Predictor for
one complete audio frame: the two-token prefill followed by exactly fourteen
single-token decode calls that produce codebooks 1 through 15. Match the pinned
official/qwentts semantics, verify every layer's cache and every private
embedding/head selection, and prove that predictor state resets between audio
frames.

## Authoritative Context

- Official Qwen revision:
  `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- qwentts.cpp revision:
  `82cd05b9f3a175612dc89fd6943e610fab096ef5`
- Pinned model snapshot:
  `C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc`
- qwentts reference:
  `E:\qwentts.cpp-reference\src\code-predictor-forward.h`
- Production path:
  `Talker -> CodePredictor::{first_step_logits,generate_sampled_with_observer}
  -> forward_layers -> DecoderLayer`
- P02 real token fixture and P03 stage/Oracle evidence are inputs; do not
  regenerate them with Rust as the sole oracle.

## Required Semantics

For each audio frame:

1. Start with a fresh Code Predictor KV cache. No cache may carry across
   frames.
2. Prefill sequence length is exactly two:
   `[talker_last_hidden, codebook_0_embedding]`, followed by the optional
   `small_to_mtp` projection.
3. Prefill positions are `[0, 1]` with a causal mask. The first private
   `lm_head[0]` produces codebook 1.
4. Fourteen incremental calls use absolute positions `2..15`.
5. Incremental step `g=1..14` consumes the token sampled at the preceding
   group through private `codec_embeddings[g-1]` and predicts through private
   `lm_heads[g]`.
6. Every one of the configured predictor layers grows from cache length 2 to
   16 exactly. Cached prefixes remain bit-identical after append.
7. Greedy and sampled paths share tensor/cache semantics; only token selection
   differs.

## Allowed Files

- `src/talker/code_predictor.rs`
- `src/talker/decoder_layer.rs` only for a demonstrated shared cache defect
- `src/talker/talker_attention.rs` only for a demonstrated attention defect
- `src/talker/primitives.rs` only for a demonstrated RoPE/norm defect
- `src/alignment_stage_dump.rs` only when an additional predictor hook is
  strictly required
- `tests/code_predictor_frame_test.rs`
- `tests/code_predictor_frame_real_test.rs`
- `fixtures/alignment/p03_code_predictor_frame_real.json`
- `tools/generate_code_predictor_frame_fixture.py`
- `docs/alignment/p03-code-predictor-frame.md`
- `config/fixtures.json`
- `artifacts/alignment/P03/P03-T03/*`

Do not modify Talker sampling policy, prompt assembly, tokenizer/codec code,
P03-T01/P03-T02 evidence, Git state, or remote state. If an independent
official oracle requires a narrowly scoped adapter change, stop and request
main-agent approval before editing outside this list.

## Required Work

1. Trace the exact production and qwentts/official execution order. Record the
   embedding/head/cache/position mapping for all 15 acoustic codebooks.
2. Add a deterministic tiny production test with at least two predictor
   layers and distinct private embedding/head weights. Compare:
   - one prefill plus fourteen cached calls;
   - full recomputation of each growing prefix using the same explicit
     positions and causal masks;
   - every layer's K/V prefix and appended position;
   - normalized hidden state and logits for all 15 groups.
3. Add mutation tests that fail for an off-by-one private embedding, private
   head, or position, and for cache reuse from a previous frame.
4. Fail closed, without partial cache mutation or panic, for wrong cache
   count, mixed presence, incompatible K/V shape/dtype/device/sequence, or
   more layers than the fixed production contract. Avoid per-token host
   container allocation on the cache path.
5. Prove frame-local reset by running two distinct frames both separately and
   consecutively. Frame 2 must equal a fresh run and must differ when frame 1
   cache is deliberately reused.
6. Build an independent real-weight oracle from the pinned official Python
   implementation or qwentts tensors. It must capture, at minimum, the
   two-token projected prefill input, positions, 15 logits tensors, final
   codes, and enough intermediate/cache data to detect shared Rust errors.
   Record source revision and SHA256. Do not lower the `0.999` cosine or
   `0.001` max-absolute limits.
7. Add an ignored real-weight test for the exact 0.6B Base snapshot. Verify
   five predictor layers, cache lengths 2 through 16, all 15 private
   embedding/head selections, logits/codes against the independent oracle,
   and deterministic JSON metrics. Missing model/oracle/config assets must
   fail with `FIXTURE_MISSING`.
8. Preserve P03-T01 stage names/capture-off behavior and P02 token sampling
   semantics. Do not add unconditional device-to-host transfers.
9. Register all required checked-in fixtures and metrics in
   `config/fixtures.json`, including generator command, license, revision, and
   SHA256.

## Required Commands

```powershell
$env:PATH='C:\Users\steven\.cargo\bin;' + $env:PATH
$env:QWEN3_TTS_REAL_MODEL_DIR='C:\Users\steven\.cache\huggingface\hub\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc'

cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo check --no-default-features --features cpu,stage-dump
cargo test --test code_predictor_frame_test --no-default-features
cargo test --test code_predictor_frame_real_test -- --ignored --nocapture
cargo test --test stage_instrumentation_test --features stage-dump
cargo test --test stage_instrumentation_real_test --features stage-dump -- --ignored --nocapture
cargo test --lib code_predictor
cargo test --test deterministic_token_sequence_test
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
python tools/fixture_manifest.py config/fixtures.json
python tools/generate_code_predictor_frame_fixture.py --check fixtures/alignment/p03_code_predictor_frame_real.json
git diff --check -- src/talker/code_predictor.rs src/talker/decoder_layer.rs src/talker/talker_attention.rs src/talker/primitives.rs src/alignment_stage_dump.rs tests/code_predictor_frame_test.rs tests/code_predictor_frame_real_test.rs fixtures/alignment/p03_code_predictor_frame_real.json tools/generate_code_predictor_frame_fixture.py docs/alignment/p03-code-predictor-frame.md config/fixtures.json artifacts/alignment/P03/P03-T03
```

## Evidence and Completion

Write `commands.txt`, `test-results.txt`, `worker-report.md`, deterministic
metrics/oracle files, and a draft `gate.json` under
`artifacts/alignment/P03/P03-T03/`. Do not write `review.md`; the independent
reviewer owns it. Do not mark task/status indexes complete and do not commit or
push.
