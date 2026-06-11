# ADR: Repository layout and artifact consolidation

Date: 2026-06-11

Decision: keep the repository root for Rust build metadata, AGENTS.md, plan/spec contracts, and primary entrypoints only. Checked-in non-source validation material belongs under artifacts/, ad-hoc Windows run wrappers under scripts/local/, temporary upstream comparison probes under tools/prototypes/, and durable engineering notes under docs/.

Context: the root directory had accumulated tracked local run products such as token streams, logs, prototype scripts, temporary inspection dumps, and story prompts. These were useful for traceability but made code discovery harder for humans and agents.

Consequences: future generated WAVs, root-level tokens/logs/webm files, temporary scripts, and prompt scratch files should remain ignored unless intentionally promoted into artifacts/ with a clear README-backed reason. The codebase-memory graph artifact remains under .codebase-memory/ for agent discovery.