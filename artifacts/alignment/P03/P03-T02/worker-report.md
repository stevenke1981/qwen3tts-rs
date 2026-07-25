# P03-T02 Worker Report

- Root cause: non-ICL InputBuilder omitted full text/eos/pad-bos prefill rows.
- Fix: align Base and xvector paths with pinned qwentts layout; preserve ICL.
- Oracle: qwentts.cpp revision `82cd05b9f3a175612dc89fd6943e610fab096ef5`.
- Real gate: PASS, sequence 11, final hidden cosine `0.999999999999012`.
- Tiny gate: 2/2 PASS; generator metrics check PASS.
- Cache staging uses fixed 28-slot stack storage and atomic `take()` commit; no hot-path Vec clone.
- Stage instrumentation real gate: PASS, manifest_count=723.
- Metrics artifact SHA256: `598486218384408FC5F8BEC6706F4876DB1D9D00F0135640F088C79261E5A34C`.
