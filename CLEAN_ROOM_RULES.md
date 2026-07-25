# Clean-room and License Rules

- qwentts.cpp is MIT licensed; Qwen3-TTS upstream is Apache-2.0.
- Behavioral facts, tensor layouts, metadata names and algorithms may be reproduced.
- When non-trivial source code is translated or adapted, preserve required copyright and license
  notices and document origin in `NOTICE.md`.
- Prefer writing tests and contracts from observed behavior, then implementing idiomatic Rust.
- Do not paste whole C++ files into Rust comments or task reports.
- Keep third-party submodule/kernel licenses in release artifacts.
- Voice samples and reference audio require explicit permission and must not be committed by default.
