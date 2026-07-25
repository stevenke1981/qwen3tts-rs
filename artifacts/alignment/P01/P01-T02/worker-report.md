# P01-T02 Worker Report

Status: `IMPLEMENTED`

The GPT-5.3 Codex Spark implementer extended official metadata parsing with
every prompt-affecting top-level and Talker token, language table, speaker
table, and heterogeneous dialect map. The loaded Candle runtime now overwrites
shape-inferred defaults with validated metadata in both model constructors.

Input construction rejects unknown explicit languages and speakers, applies
official Chinese/Auto dialect substitution, and leaves non-Chinese languages
unchanged. Synthetic CustomVoice coverage uses the official nine speakers and
IDs rather than invented fixtures.
