# P03-T05 External-Agent Prompt

You are the bounded External Implementer for `P03-T05`.

Read these files completely before editing:

1. `AGENTS.md`
2. `spec.md`
3. `artifacts/alignment/P03/P03-T05/assignment.md`
4. Every contract, test, and prior Gate referenced by the assignment

Implement exactly one task card. Modify only its Allowed Files. Use test-first
development and run every Required Command with the pinned real fixtures.

Your output is not a Gate decision. Write only:

- implementation and tests inside Allowed Files;
- `worker-report.md`;
- `commands.txt`;
- `test-results.txt`;
- the raw metric/mapping/manifests required by the assignment.

Do not write `review.md` or `gate.json`. Do not update `STATUS.md`, `TODOS.md`,
`tasks/task-index.yaml`, thresholds, Git history, or remote state. Do not commit
or push.

If the complete 723-stage official reference, complete 723-stage Candle
candidate, exact qwentts anchor mapping, pinned model, or required backend is
missing or cannot be generated, stop and report F8 `BLOCKED`. Never substitute
partial dumps, random weights, synthetic-only evidence, silent skips, or
relaxed thresholds.

Report exact commands, exit codes, passed/failed/ignored counts, numerical
metrics, changed files, design choices, and remaining risks. GPT-5.6 Sol will
inspect the actual diff, rerun the full Gate, and decide acceptance.

## Sol Rejection Addendum — Required Rework

The first Qwen 3.8 Preview submission was rejected as `F8 BLOCKED`. Do not
resubmit the existing partial result.

Fix all of the following within the task card's Allowed Files:

1. Implement the official Python generation/capture path needed to emit exactly
   the same 723 required stage names as the Candle candidate. A 678-stage
   forward-hook-only dump is not acceptable.
2. Use the pinned fixture seed `12345` consistently for runtime generation and
   both manifests. Reject any CLI seed that differs from the fixture.
3. Preserve and validate both official source revision
   `022e286b98fbec7e1e916cb940cdf532cd9f488e` and model snapshot revision
   `5d83992436eae1d760afd27aff78a71d676296fc`, plus source, model, case, and seed.
4. Make `--require-all-reference-stages` work in mapping mode. All 16
   qwentts.cpp anchors must be mapped and compared, or the command must fail.
5. Do not bypass layout validation merely because a mapping is present. Any
   reshape/axis transform must be explicit, deterministic, validated, and
   covered by mutation tests.
6. Add provenance, mapping-completeness, layout-transform, empty-logit, and
   rank-margin mutation tests.
7. Make `P03_STAGE_DUMP_DIR` cleanup safe: reject paths outside the task-owned
   artifact output root and never recursively delete an arbitrary environment
   path.
8. Resolve the real qwentts.cpp fixture mismatch. The rejected run compared a
   21-token qwentts prefill against an 11-token Candle prefill and failed with
   cosine values as low as `0.807507106672`.

Before reporting completion, run every Required Command and provide successful
`stage-metrics.json` and `qwentts-anchor-metrics.json`. If an exact 723-stage
official capture or exact 16-anchor pairing remains unavailable, report
`F8 BLOCKED`; do not say complete.
