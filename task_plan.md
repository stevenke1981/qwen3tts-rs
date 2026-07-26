# Task Plan: External-Agent Alignment Workflow

## Goal

Rewrite the remaining qwentts.cpp alignment roadmap so bounded external agents
can implement one task card at a time while GPT-5.6 Sol independently reviews,
tests, gates, documents, commits, and publishes accepted work.

## Current Phase

Phase 6

## Phases

### Phase 1: Requirements and repository discovery

- [x] Push the accepted P03-T04 branch state.
- [x] Confirm current roadmap and Gate status.
- [x] Read the planning-with-files workflow.
- **Status:** complete

### Phase 2: Workflow architecture

- [x] Define external-agent task-card contract.
- [x] Define Sol-only review and Gate responsibilities.
- [x] Define evidence, failure, Git, and handoff rules.
- **Status:** complete

### Phase 3: Rewrite durable project documents

- [x] Update `plan.md` for P03-T05 through P13.
- [x] Align `AGENTS.md`, `STATUS.md`, and `TODOS.md`.
- [x] Add reusable external-agent task-card and report templates.
- **Status:** complete

### Phase 4: Verification and independent review

- [x] Check document consistency and stale workflow claims.
- [x] Run formatting/diff validation.
- [x] Obtain independent read-only review if required by project Gate rules.
- **Status:** complete

### Phase 5: Commit and publish

- [x] Stage only workflow-document changes.
- [x] Commit with a scoped message.
- [x] Push the current branch.
- **Status:** complete

### Phase 6: Audit Qwen 3.8 Preview delivery

- [ ] Inventory actual code, tests, reports, manifests, and Git changes.
- [ ] Compare every change against P03-T05 allowed files and thresholds.
- [ ] Reject skipped, partial, synthetic-only, or stale evidence.
- **Status:** completed

### Phase 7: Independent Sol verification

- [ ] Run focused compile, comparator, mutation, and real-fixture commands.
- [ ] Validate 723-stage official/Candle manifests and qwentts anchors.
- [ ] Obtain independent read-only review of the final diff.
- **Status:** completed

### Phase 8: Gate and publish

- [ ] Write Sol review and Gate decision.
- [ ] Update status/index only if accepted.
- [ ] Commit and push only accepted, scoped work.
- **Status:** completed

## Key Questions

1. Which files are authoritative for execution order and agent behavior?
2. How can external agents work without receiving merge, Gate, or push authority?
3. What evidence must Sol rerun rather than trust from an external report?

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| External agents receive one bounded task card | Prevents phase-wide scope drift |
| Sol owns review, Gate, status, commit, and push | User explicitly requires final Sol acceptance |
| External reports are claims until independently rerun | P03-T04 demonstrated false-green reports are possible |
| Preserve P00-P13 ordering | Existing roadmap remains the authoritative dependency chain |
| Treat Qwen 3.8 Preview completion as unverified | External reports require actual diff and Sol reruns |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Combined memory/skill lookup returned exit 1 without output | 1 | Read the skill and catch-up script separately; no relevant memory registry hit was found |
