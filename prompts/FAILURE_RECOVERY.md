# Failure Recovery Prompt for Sol

A task has failed. Do not enlarge the task.

1. Classify the failure F1-F8 using AGENTS.md.
2. Preserve the failing command, logs and artifacts.
3. Identify the smallest falsifiable hypothesis.
4. Create one repair task touching fewer files than the failed task where possible.
5. Ask Spark to add a regression test reproducing the failure before the fix.
6. Re-run the task gate and the immediately preceding phase smoke gate.
7. Document the root cause in `docs/alignment/lessons.md`.
