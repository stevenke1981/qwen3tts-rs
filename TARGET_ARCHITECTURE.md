# Target Architecture

```text
                 Model Metadata / Config
                          │
            ┌─────────────┴─────────────┐
            │                           │
       Text Tokenizer              Audio Frontend
            │                           │
       Prompt Builder       Ref WAV → Tokenizer Encoder
            │                           │
            ├────────── Speaker Encoder / ICL Codes
            │
      Talker Session Runtime
  28 layers + persistent KV cache
            │ codebook-0 frame
            ▼
   Code Predictor Frame Runtime
  5 layers, frame-local KV cache
            │ codes[16]
            ├──────────────→ token/code event
            ▼
 Stateful Codec Decode Session
  RVQ lookup/project
  decoder transformer KV ring
  causal conv state
  transposed-conv overlap state
            │ 1920 PCM samples/frame
            ├──────────────→ callback / HTTP stream / player
            ▼
      buffered output / WAV
```

## Module Boundaries

Recommended target modules:

```text
src/
  model/
    metadata.rs
    format.rs
  generation/
    prompt.rs
    sampling.rs
    philox.rs
    talker_session.rs
    predictor_session.rs
  tokenizer/
    encoder/
    decoder/
    rvq.rs
    stream_state.rs
  runtime/
    session.rs
    scheduler.rs
    callbacks.rs
    cancellation.rs
  quant/
    types.rs
    loader.rs
    matmul.rs
    embedding.rs
    policy.rs
    backends/
  api/
    library.rs
    ffi.rs
  server/
    openai.rs
    voice_registry.rs
  bin/
    qwen-tts.rs
    qwen-codec.rs
    tts-server.rs
```

The final names may differ, but the ownership boundaries must remain.

## Streaming State Contract

A `CodecStreamState` owns all mutable state required to decode the next frame without reading
earlier input frames from host memory:

- current absolute frame position
- transformer KV ring and valid mask
- per-causal-conv left context
- per-transposed-conv overlap tail
- primed ICL state
- cancellation/session identity
- output frame counter

`decode_frame(&[u16; 16])` must not:

- append the complete history to a growing Vec
- rebuild a tensor containing all earlier frames
- call the offline full-sequence decoder
- allocate model-sized buffers per frame

## Quantized Runtime Contract

A quantized tensor stores:

- format and shape
- block/group metadata
- packed data
- scales/minimums as required
- backend resident allocation

Quantized linear and embedding operations consume this representation directly.
Temporary tile dequantization inside a kernel is allowed; permanent full-weight F32 expansion is not.

## Scheduler Contract

- Each session owns logical KV and codec states.
- Batching may group sessions at the same model/runtime stage.
- A stalled client cannot block unrelated sessions.
- Session cancellation removes it from future batches and releases state.
- Deterministic single-session mode bypasses scheduling variability for parity tests.
