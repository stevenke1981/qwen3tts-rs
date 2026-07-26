# Findings and Decisions

## Requirements

- Push the already accepted P03-T04 state first.
- Rewrite the remaining plan for implementation by external agents.
- Keep GPT-5.6 Sol as the independent reviewer and final Gate owner.
- Commit and push the rewritten workflow after validation.

## Research Findings

- The branch push to `origin/alignment/full-qwentts-parity` succeeded at
  commit `8fdcb2b`.
- `plan.md` already contains the authoritative P00-P13 roadmap.
- `AGENTS.md` contains a Spark-specific controlled workflow that should be
  generalized to external agents without weakening technical constraints.
- `STATUS.md` currently points to P03-T05; P04-P13 remain pending.
- P13 is the only final alignment/release Gate.
- `tasks/task-index.yaml` and per-task Gate evidence are co-authoritative with
  the P00-P13 section of `plan.md`.
- The existing `AGENTS.md` workflow hard-codes GPT-5.3 Codex Spark even though
  its underlying restrictions already fit provider-neutral external workers.
- Every remaining phase can retain its technical task order while using a
  common ownership rule: external implementation, Sol acceptance, and Sol-only
  phase Gate.
- P03-T04 proved external reports cannot be accepted solely from claimed test
  counts; compilation, ignored tests, transfer accounting, API compatibility,
  CUDA safety, and evidence freshness all required independent Sol correction.
- `tasks/task-index.yaml` is stale: it still marks P03-T03 as `READY` and
  P03-T04 as `NOT_STARTED`, despite accepted Gate evidence. The workflow rewrite
  must synchronize these statuses and set P03-T05 to `READY`.
- P03-T05 can build on `tools/compare_stage_dumps.py`, the P03-T01 stage schema,
  P03-T02 Talker metrics, and P03-T03 Code Predictor real metrics. Existing
  numerical limits are cosine `>= 0.999` and max absolute error `<= 0.001`.
- YAML parsing confirms P03-T01 through P03-T04 are `GATE_PASSED` and P03-T05
  is `READY` under the new external-agent execution model.
- Repository-wide formatting can include unrelated worktree drift, so the
  P03-T05 card requires reporting the global check plus a scoped rustfmt check
  for its exact Rust files; the external agent may not format unrelated files.
- Independent review found that the first P03-T05 card could false-green: it
  ran only comparator `--help`, omitted the new gate test, and did not pin full
  manifests. The corrected design requires a 723-stage official-Python
  reference, a persistent 723-stage Candle candidate, an explicit qwentts.cpp
  anchor mapping, and two reproducible comparator invocations.
- `READY` is now part of the AGENTS status vocabulary with Sol-owned
  `NOT_STARTED → READY → IN_PROGRESS` transitions.
- Final independent review accepted the workflow. Its non-blocking provenance
  suggestion was incorporated by pinning the official checkout, Python
  executable, official source revision, and model snapshot revision in the
  P03-T05 export command and manifest contract.

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Use provider-neutral term `External Implementer` | Supports DeepSeek V4 Flash or another explicitly assigned external model |
| Keep task cards under `artifacts/alignment/<phase>/<task>/` | Matches existing evidence layout |
| Add reusable templates under `docs/alignment/agent-workflow/` | Gives external agents copyable, durable contracts |
| Sol must rerun required commands and inspect the actual diff | Prevents self-reported false positives |
| External agents never update Gate/status or push | Separates implementation from acceptance authority |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| Worktree contains unrelated user changes and generated databases | Preserve them and stage only workflow-owned files |
| Previously pushed history contains generated artifacts | User explicitly authorized the push after being informed |

## Resources

- `plan.md`
- `AGENTS.md`
- `STATUS.md`
- `TODOS.md`
- `artifacts/alignment/P03/P03-T04/`
