# P02-T05 deterministic token sequences

The committed corpus replays explicit captured logits, options, histories and
Philox subsequences through an independent qwentts.cpp-compatible oracle. It
covers all four Talker/subtalker mode routes, top-k/top-p boundaries, signed
repetition penalties, shared subsequences, c0 history and empty Code Predictor
history. Expected token matrices are literal fixture data; the generator never
constructs logits from expected tokens. P02 gates sampling decisions only; P03
separately gates Rust model logits and hidden states against the reference
implementation.

The sampler follows qwentts.cpp's original-vocabulary `float` accumulation:
temperature and top-k mutate the vocabulary-ordered logits, top-p computes its
full-vocabulary denominator before applying the `max_logit - 16` cutoff and
sorted nucleus mask, and the final Philox draw scans the masked exponentials in
vocabulary order. The corpus includes adversarial sum-order and cutoff-boundary
vectors so a descending-probability reduction cannot silently replace this
contract.

The real-weight 0.6B oracle remains an independently pinned integration gate;
missing snapshot or oracle fixture must fail closed rather than silently skip.
