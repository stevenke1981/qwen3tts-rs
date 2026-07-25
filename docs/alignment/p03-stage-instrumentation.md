# P03 Stage Instrumentation

`StageDumpObserver::on_stage` is a generic, capture-gated hook. Existing
forward methods remain unchanged; observer-aware variants delegate to them and
the default observer performs no host copy, formatting, or file allocation.

Stage names are deterministic and include component, phase/frame/step, layer,
and substage. Talker families include `talker-input-embed`,
`talker-hidden-prefill-l{n}`, `talker-hidden-prefill-final`,
`talker-hidden-step{n}`, and codec/RoPE/position stages. Code Predictor names
use `code-predictor-prefill-frame{f}-...` and
`code-predictor-step{s}-frame{f}-...`.

Mapping to qwentts.cpp is direct: `talker-input-embed` maps to
`talker-input-embed`; prefill final hidden maps to
`talker-hidden-prefill-final`; codebook-0 logits map to
`talker-logits-prefill`/step logits; `next-emb-step0` maps to the first
Talker codec embedding; and `talker-hidden-step1` maps to the incremental
Talker final hidden stage. Manifest entries are F32, little-endian, contiguous
(`BTH`, `BBTH`, or `BT` layouts) and SHA-256 hashed by the writer.
