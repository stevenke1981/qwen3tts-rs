# TEAM.md

## Core Roles

| Role | Model | Responsibility |
|---|---|---|
| Alignment Lead | GPT-5.6 Sol | architecture, sequencing, gate ownership |
| Numerical Investigator | GPT-5.3 Codex Spark | stage dumps, token/audio comparisons |
| Runtime Implementer | GPT-5.3 Codex Spark | KV cache, streaming state, scheduler |
| Kernel Implementer | GPT-5.3 Codex Spark | quantized ops and backend kernels |
| Product Implementer | GPT-5.3 Codex Spark | CLI, server, FFI |
| Independent Reviewer | separate Spark invocation or Sol | adversarial review |
| Release Auditor | GPT-5.6 Sol | full matrix and release evidence |

## Assignment Constraints

- One Spark task should normally touch no more than 6 production files.
- Target effort per Spark task: one coherent algorithm or one testable interface.
- Architecture changes require a Sol-authored decision record in `docs/alignment/decisions/`.
- The implementer and reviewer must be separate invocations.
- A failed task is narrowed, not enlarged.
