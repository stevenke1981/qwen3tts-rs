# P02-T05 Worker Report

- Replaced the prior self-fulfilling token fixture with an independent
  qwentts.cpp-compatible F32/Philox oracle over explicit logits, options,
  histories, masks and subsequences.
- Covered all four Talker/Code Predictor sampling routes, multiple seeds,
  top-k threshold ties, top-p crossing and relative cutoff boundaries,
  positive and negative repetition-penalty inputs, shared Philox ordering,
  Talker suppression, unsuppressed Code Predictor tokens, and terminal EOS.
- The EOS sequence contains two complete 16-codebook frames followed by a
  terminal c0-only call; the incomplete terminal frame is excluded.
- Corrected production sampling to use original-vocabulary F32 accumulation,
  preserve kth-score ties, apply the qwentts top-p cutoff, keep the crossing
  token, fail closed on no finite candidates, and skip repetition penalty in
  greedy mode.
- Matched Hugging Face generation-cap semantics in both greedy and sampled
  Talker paths: the terminal c0 draw is counted but its incomplete frame is
  not returned.
- The pinned real 0.6B Base production route now matches the official
  patched-PyTorch oracle exactly: 3 x 16 tokens and 49 Philox draws.
- Offline checks validate synthetic and real fixture revisions, structure,
  unique manifest entries, SHA-256 values and mutation rejection.
- No commit, push, or task-status mutation was performed by implementation
  workers.
