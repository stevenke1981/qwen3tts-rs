# P02-T04 suppression and EOS parity

Talker codebook 0 masks the final 1024 vocabulary entries dynamically. EOS is
forbidden for generated steps 0 and 1 (`min_new_tokens=2`) and is the only
reserved token admitted from step 2 onward. EOS terminates before Code
Predictor invocation and is not emitted as a frame or repetition-history item.
Code Predictor codebooks remain unsuppressed.

The checked-in vectors are generated offline by
`tools/generate_suppression_eos_vectors.py` and hash-verified in
`config/fixtures.json`.
