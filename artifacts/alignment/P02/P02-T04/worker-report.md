# P02-T04 Worker Report

- Talker suppression now derives the reserved suffix from `vocab_size - 1024`
  and fails closed when vocabulary or EOS metadata violates the contract.
- Greedy and sampled c0 selection use the same suppression rules.
- EOS is suppressed for generated c0 steps 0 and 1 and becomes the only allowed
  reserved-suffix token from step 2 onward.
- A sampled EOS terminates before Code Predictor execution, frame append,
  history append, or any additional RNG draw.
- Code Predictor sampling remains unsuppressed; its production test uses a
  2048-token vocabulary and selects reserved token 1500.
- Production tests cover sampled and greedy Talker EOS termination, literal
  sampled suppression vectors, output matrices, history, and Philox ordering.
- Fixture provenance pins official Qwen and qwentts.cpp revisions and is guarded
  by its manifest SHA-256.
- No commit, push, or task-status mutation was performed by the implementation
  worker.
