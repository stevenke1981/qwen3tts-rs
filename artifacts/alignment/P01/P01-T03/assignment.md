# GPT-5.3 Codex Spark — Implementation Assignment

## Task

- Task ID: `P01-T03`
- Goal: match official Qwen3-TTS M-RoPE channel selection, prefill position
  IDs, RoPE deltas, and cached-decode positions exactly.
- Dependency: `P01-T02 GATE_PASSED`.

## Allowed files

- `src/text_frontend/model_catalog.rs`
- `src/text_frontend/candle_backend.rs`
- `src/talker/primitives.rs`
- `src/talker/talker.rs`
- `tests/mrope_reference_test.rs` (new)
- `docs/alignment/mrope-parity.md` (new)

No other files may be modified. Do not change prompt assembly, sampling,
weights, attention masks, codec generation, or public product APIs.

## Official semantics

Use the pinned official implementation
`qwen_tts/core/models/modeling_qwen3_tts.py` as the oracle:

- `get_rope_index`: `cumsum(mask)-1`, padded positions filled with `1`,
  result expanded to `[3,batch,seq]`, and delta shaped `[batch,1]` as
  `max_position + 1 - valid_length`;
- cached decode: `arange(query_len) + cache_position_start + rope_delta`,
  expanded to all three axes;
- interleaved M-RoPE: start from axis 0 over half-head channels, then replace
  axis `i` at `i .. section[i]*3` with stride 3, finally duplicate the
  half-head cos/sin to the full head;
- non-interleaved M-RoPE: split the full head with
  `mrope_section * 2` (list repetition, not numeric doubling) and select
  chunks by `chunk_index % 3`;
- `rotate_half` and q/k application must match the official formula exactly.

## Metadata/runtime contract

- Parse and validate `talker_config.rope_theta` in `ModelMetadata`.
- Install `rope_theta`, `rope_scaling.interleaved`, and `mrope_section` into
  every loaded Candle `TalkerConfig` before building the model.
- Reject non-finite/non-positive theta and structurally inconsistent section
  data.
- Do not retain a runtime path where loaded model RoPE behavior comes only
  from hard-coded defaults.

## Tests

- Exact prefill position IDs and `[batch,1]` deltas for all-valid,
  left-padded, right-padded, and two-batch masks.
- Exact cached-decode positions with non-zero cache start and per-batch delta.
- Exact interleaved axis/channel selection for official `[24,20,20]`,
  `head_dim=128`; assert boundary channels including 0,1,2,57,58,59,60,63
  and their duplicated full-head partners.
- Exact non-interleaved selection proving list repetition semantics.
- Small hand-computed cos/sin and q/k rotation vectors with tolerance
  `<= 1e-6`; expected values must not be produced by the Rust function under
  test.
- Invalid rank/axis/section/theta inputs fail with errors, not panic.
- The ignored reference test may invoke the pinned local official Python
  implementation to compare a deterministic fixture, but when explicitly
  run it must require its prerequisites and cannot silently return.

## Required commands

```text
cargo fmt --all -- --check
cargo check --no-default-features --features cpu
cargo test --lib talker::primitives
cargo test --lib talker::talker
cargo test --test mrope_reference_test -- --nocapture
cargo check --example synthesize --features candle-llm
cargo check --example synthesize_batch --features candle-llm
git diff --check -- src/text_frontend/model_catalog.rs src/text_frontend/candle_backend.rs src/talker/primitives.rs src/talker/talker.rs tests/mrope_reference_test.rs docs/alignment/mrope-parity.md
```

## Acceptance

- Position IDs, delta shape/value, cached positions, channel selection, and
  q/k rotation match the official formulas.
- Both interleaved and non-interleaved paths are exact and boundary-tested.
- Loaded runtime RoPE configuration is metadata-driven.
- Tests are deterministic and do not silently skip.
- Diff stays inside the six allowed files.
