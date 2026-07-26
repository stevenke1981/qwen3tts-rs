# P03 Acoustic Host Transfers

## Scope

P03-T04 removes avoidable host synchronization from Talker codebook-0
selection, Code Predictor generation, and final acoustic-frame assembly. It
does not change Philox probability math, token order, cache semantics, or stage
names.

The pinned 0.6B model uses a Talker vocabulary of 3072 and a Code Predictor
vocabulary of 2048 with 15 private codebook draws per completed frame.

## Greedy path

Codebook-0 suppression and argmax run on the device. The selected scalar crosses
to the host only for EOS/control flow and fail-closed sentinel validation. The
terminal draw also reads one scalar: validating before embedding is required
because CUDA gather kernels cannot safely validate an out-of-range sentinel.

Code Predictor argmax tensors remain on device and feed the next private
embedding directly. Its 15 output codes and the final 16-code frame are
assembled on device.

Per ordinary completed frame:

| Direction | Before | After |
|---|---:|---:|
| Device to host | 3102 elements | 1 element |
| Host to device | 46 elements | 0 elements |

The before count comprises one 3072-element Talker logit vector, 15 Code
Predictor scalar reads, and a 15-code Talker readback. The old host-to-device
path rebuilt codebook-0, private tokens, final predictor codes, and the final
frame.

The terminal-cap draw adds one device-to-host scalar and produces no frame.
This preserves the pinned generation draw semantics while preventing an
all-invalid sentinel from reaching an embedding kernel. It is an explicit
exception to the ordinary completed-frame budget.

## Sampled path

The CPU Philox sampler still requires exactly one full-logit download per draw.
Telemetry is emitted inside `Sampler` immediately after the real `to_vec1`
boundary. Each selected host scalar is uploaded once for the following device
embedding. No complete code vector is downloaded or uploaded.

Per completed frame:

| Direction | Before | After |
|---|---:|---:|
| Device to host | 33807 elements | 33792 elements |
| Host to device | 46 elements | 16 elements |

The after count is one 3072-element Talker logit vector plus fifteen
2048-element Code Predictor logit vectors, followed by one codebook-0 and
fifteen private scalar uploads.

## Telemetry contract

`TransferObserver` receives fixed-size event values:

- `Codebook0Scalar`
- `SubCodebookScalar`
- `FullLogitVector(vocab_size)`

`NoopStageDumpObserver` and `StageDumpWriter` use the allocation-free default
implementation. All production callback sites first check
`wants_transfer_capture()`, so capture-off observers receive no callback. Test
collectors may allocate outside the production hot path. Stage callbacks never
infer transfer events, and existing `StageDumpObserver` implementations do not
need to implement transfer telemetry.

## Greedy non-finite handling

Non-finite logits are masked on device. A finite sentinel at index
`vocab_size` represents the all-invalid case without an extra synchronization.
Talker converts that sentinel into the same fail-closed error before any
embedding lookup. Batch sizes other than one and empty vocabularies fail
explicitly.
