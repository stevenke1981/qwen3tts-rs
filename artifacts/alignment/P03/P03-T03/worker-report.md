# P03-T03 worker report

Production Code Predictor prefill now validates the fixed frame contract and
requires an empty frame-local cache. `forward_layers` validates layer count,
input/RoPE/mask geometry, K/V shape, dtype, device, presence, shared sequence
length, and maximum length before staging all layer cache updates. Greedy
argmax no longer allocates a one-element host Vec.

The tiny two-layer test uses distinct nonzero weights. Its production private
embedding/head loop is compared at every group against a fresh growing-prefix
recompute for normalized hidden, logits, and every K/V tensor. Every Rust K/V
prefix is also checked bit-for-bit after append. Embedding, head, and position
mutations are independently detected. Malformed caches and a failing final
observer callback do not partially mutate state, and frame reuse is rejected.

The official exporter consumes the T02 qwentts
`talker-hidden-prefill-final.bin` final row, loads the pinned 0.6B Base model
on CPU F32, and executes an explicit prefill plus fourteen incremental calls.
It captures projected inputs, normalized hidden, all fifteen logits, actual
five-layer K/V tensors, cache lengths, and codes. The manual loop is bit-equal
to official `GenerationMixin`; all assets have SHA256 metadata.

The ignored Rust gate loads the same pinned safetensors through
`TalkerWeightLoader` and compares all groups/layers. Result: minimum cosine
`0.999999999939492`, maximum absolute error `0.000438690185546875`, and exact
codes. The real gate resolves every tensor through the hashed oracle manifest,
verifies byte sizes and model/config hashes, and fails closed with
`FIXTURE_MISSING`. The existing Windows `LNK4098` warning remains unrelated.
