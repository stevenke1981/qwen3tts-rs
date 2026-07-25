# P03-T01 Worker Report

Production Talker and Code Predictor paths now expose capture-aware stage hooks
for embeddings, positions, RoPE, every decoder-layer substage, final norms,
logits, codes, and the post-codebook next embedding. Existing public forward
APIs delegate through the no-op observer.

Capture-only formatting, host conversion, hashing, and file writes are guarded
by `wants_capture()`. The writer enforces an exact structured-name grammar,
canonical layouts, little-endian F32 bytes, unique names, and SHA-256 entries.

The synthetic gate runs a real tiny Talker plus Code Predictor production path
with one layer in each component. The pinned 0.6B Base gate verified 723 exact
stages, qwentts-compatible `talker-hidden-step1`, complete per-layer coverage,
manifest provenance, layout/shape byte lengths, file hashes, and no-op token
parity.
