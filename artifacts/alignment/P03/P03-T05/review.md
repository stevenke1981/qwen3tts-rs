# P03-T05 Independent Review

## Result

`REJECT` — `F8 BLOCKED`. P03-T05 is not complete and must not advance the
P03 Phase Gate.

## Gate-blocking findings

1. The official Python reference contains 678 stages, not the required 723.
   The Candle real-model run independently reproduced 723 stages.
2. The full official-Python-to-Candle comparator therefore fails before any
   numerical comparison: `reference stage count 678 != expected 723`.
3. The qwentts.cpp anchor comparison independently failed. The mapping compares
   only 10 of 16 anchors, several mapped prefill tensors have incompatible
   21-token versus 11-token shapes, and observed cosine values include
   `0.841567582635`, `0.807507106672`, and `0.964468462082`.
4. Mapping mode bypasses `--require-all-reference-stages`, so six unmapped
   anchors are silently excluded instead of failing closed.
5. Comparator provenance is report-only: source, revision, model, case, and seed
   are not validated against pinned expectations.
6. The exporter records seed 42 but actually generates with fixture seed 12345,
   and records the model snapshot revision where the official source revision
   must also be retained.
7. Mapped comparisons bypass layout equality without declaring or validating an
   explicit axis/layout transform.
8. `P03_STAGE_DUMP_DIR` is recursively deleted without constraining it to a safe
   task-owned artifact root.

## Sol verification

- `cargo check --no-default-features --features cpu,stage-dump`: pass.
- `stage_dump_test`: 5 passed.
- `stage_instrumentation_test`: 1 passed.
- `stage_threshold_gate_test`: 12 passed.
- Real Candle stage instrumentation: 1 passed, exactly 723 stages, 101.99 s.
- Full 723-stage comparator: failed because reference count is 678.
- qwentts.cpp anchor comparator: failed structurally and numerically after
  comparing only 10/16 anchors.
- Independent Luna review: `REJECT / F8 BLOCKED`.

## Required rework

Use the corrected external-agent prompt in this evidence directory. A new
submission must produce both complete 723-stage manifests, compare all required
stages and all 16 anchors fail-closed, use seed 12345 consistently, validate
pinned provenance, and pass both real comparator commands at the fixed
thresholds.
