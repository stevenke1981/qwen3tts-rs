# TASK-ID Assignment — TASK-TITLE

## Outcome

State one measurable outcome. This task must not contain an entire Phase.

## External Implementer

- Assigned model: `MODEL`
- Implementation authority only; no Gate, status, commit, merge, or push.

## Authoritative Context

- Dependency Gate:
- Reference revisions:
- Required contracts:
- Required fixtures:
- Production entry points:

## Allowed Files

- List every file or bounded directory the implementer may change.

Everything else is read-only. Preserve user changes and generated databases.

## Required Work

1. Read the referenced contracts and existing tests.
2. Add a failing behavioral/parity test before implementation.
3. Implement the smallest change that satisfies the task.
4. Preserve public API, numerical, state, cancellation, and backend semantics
   unless this card explicitly changes them.
5. Write raw evidence without editing Gate-owned files.

## Acceptance Thresholds

- List exact numerical, behavioral, performance, and compatibility limits.
- Missing or untrusted fixtures are F8 and must fail closed.

## Required Commands

```powershell
# Exact commands, environment variables, fixtures, and feature flags.
```

## External Deliverables

- `worker-report.md`
- `commands.txt`
- `test-results.txt`
- Task-specific raw metric artifacts

Do not write `review.md` or `gate.json`. Do not update `STATUS.md`,
`TODOS.md`, or `tasks/task-index.yaml`. Do not commit or push.
