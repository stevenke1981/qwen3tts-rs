# External-Agent Alignment Workflow

This directory defines the provider-neutral workflow for P03-T05 through P13.
External agents implement bounded task cards; GPT-5.6 Sol independently accepts
or rejects the result.

## Ownership

| Artifact or action | External Implementer | GPT-5.6 Sol |
|---|---:|---:|
| Read task card and referenced contracts | Yes | Yes |
| Modify allowed implementation/test files | Yes | If needed after rejection |
| Write worker report and raw results | Yes | May correct factual records |
| Change scope, architecture, or thresholds | No | Yes |
| Write independent review and Gate | No | Yes |
| Update STATUS/TODOS/task index | No | Yes, after acceptance |
| Commit, merge, or push | No | Yes, after acceptance |

## Required lifecycle

1. Sol inspects the real repository and creates one `assignment.md`.
2. The external agent reads every referenced contract and test before editing.
3. The external agent implements only the allowed scope and writes raw evidence.
4. Sol reviews the actual diff and classifies failures F1-F8.
5. Sol reruns every required command, including real fixtures and backend checks.
6. Sol writes `review.md` and `gate.json`.
7. Only an accepted Gate permits status updates, commit, and push.

The external agent must stop with `BLOCKED` when a fixture, model, reference
binary, backend, or contract is missing. It must never replace missing evidence
with random weights, silent skips, mocks, or relaxed thresholds.

## Files

- `assignment-template.md`: copied by Sol to a task evidence directory.
- `worker-report-template.md`: completed by the external implementer.
- `sol-review-template.md`: completed independently by Sol.
