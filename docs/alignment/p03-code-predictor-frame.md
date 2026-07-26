# P03 Code Predictor frame-local parity

The Code Predictor consumes exactly two prefill positions,
`[talker_last_hidden, codec_embedding(codebook_0)]`, at positions `0, 1`.
The prefill selects private `lm_head[0]`; each subsequent group `g=1..14`
gathers the previous token from `codec_embeddings[g-1]`, appends at absolute
position `g+1`, and selects `lm_heads[g]`.

Production `forward_layers` stages every layer's cache and commits only after
the final norm succeeds. Cache count, layer limit, shape, dtype, device,
mixed-presence, and cross-layer sequence length are checked before mutation.
A prefill rejects a non-empty cache so an audio frame cannot inherit state
from its predecessor.

The deterministic tiny test uses distinct nonzero weights for two layers and
all fifteen private head/embedding mappings. It compares every cached step
against a fresh growing-prefix recomputation, including normalized hidden,
logits, and every layer's K/V prefix and appended position. Separate mutations
prove that embedding, head, or absolute-position off-by-one errors are detected.
Malformed cache inputs fail before state mutation, and frame prefill rejects
cache reuse.

The pinned real fixture records the official and qwentts revisions and the
required cosine/max-absolute thresholds. Run
`python tools/generate_code_predictor_frame_fixture.py --export` with
`QWEN3_TTS_REAL_MODEL_DIR` set to the pinned snapshot to produce the
CPU-F32 independent oracle under `artifacts/alignment/P03/P03-T03/official-oracle`.
The exporter consumes the T02 direct-oracle hidden dump's final row and writes
the projected prefill/private inputs, normalized hidden states, fifteen logits,
greedy codes, and actual five-layer K/V tensors. Its manual loop explicitly
uses positions `0..15` and is bit-for-bit cross-checked against official
`GenerationMixin`.

The ignored Rust gate loads the real Candle weights and compares every private
input, normalized hidden, logit, and K/V prefix for cache lengths `2..16`.
The accepted run recorded minimum cosine `0.999999999939492` and maximum
absolute error `0.000438690185546875`. `--check` verifies every tensor SHA256
and fails closed when an asset is absent or altered.
