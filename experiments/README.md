# Experiment results

Results of the joint prefix-cache / speculative-decoding study on an RTX 5090 32GB.
The original benchmarks use SGLang v0.5.22 (`baa9befa381511cd4d56f83980951e5a45fffc6d`,
no runtime patches). Later HiCache experiments use the local Mamba layer-first patch
`00fd51eab3`; each report identifies its source version and scope.

Only result documents and summary tables are kept in git. Harness scripts, input fixtures and raw
per-run artifacts (server logs, per-request results, metric samples) stay in each machine's
gitignored `work_dirs/<same directory name>/`; links inside these documents to `runs/`, `inputs/`,
or `*.json` evidence files point there.

Latest state-management update: [DSpark findings](dspark-state-management/FINDINGS.md)
and [CPU offload validation](dspark-state-management/CPU_OFFLOAD.md). The 27B transport
and byte checks pass, but end-to-end output equivalence does not; diagnostic completion
must not be read as correctness validation.

| Directory | Run by | What it contains |
|---|---|---|
| `qwen3-8b-prefix-baseline-v0.5.22/` | teammate (blackmagic) | Qwen3-8B plain / EAGLE3 / DFlash / DSpark, cache on/off, 4 workloads, c1/c8. |
| `qwen3-8b-prefix-baseline-v0.5.22-repro/` | nuo, 2026-10-27 | Unmodified re-run of the row above on a second machine. `REPRO.md` lists environment differences; `comparison-vs-teammate.txt` compares all 96 phases (median ratio 1.00 for tok/s, TTFT, tokens/verify). |
| `qwen35-4b-hybrid-prefix-baseline-v0.5.22/` | teammate (blackmagic) | Qwen3.5-4B (24 GDN + 8 FA) plain / DFlash, cache on/off, Mamba state pool observations. |
| `dspark-state-management/` | teammate (blackmagic) | GPU capacity sweeps, 164 checkpoint probes, 8B CPU load-back, CUDA transfer isolation, and 27B transport/output controls. Start with `FINDINGS.md`, `PROGRESS.md`, and `CPU_OFFLOAD.md`; JSON/CSV summaries preserve pass/fail distinctions. |
| `stress-test-summary.md` | nuo, 2026-10-27 | **Phase-1 conclusion**: where speculative decoding loses under memory pressure and the candidate phase-2 directions (opportunity map). Start here. |
| `stress-qwen3-8b-v0.5.22/` | nuo, 2026-10-27 | Qwen3-8B stress test with cookbook-style config (CUDA graph + overlap on): memory anatomy, concurrency sweeps 1-64 on decode / shared-prefix / natural-chat workloads, equal-KV-capacity plain control, `--speculative-adaptive`. `e2-v1` first pass, `e2-rep2` repeat (median ratio 1.000). Narrative in `FINDINGS.md`. |
| `stress-qwen38-27b-v0.5.22/` | nuo, 2026-10-27 | Qwen3.8-27B NVFP4 hybrid stress test: memory anatomy (`e1-probe`), capacity tuning (`e1-capacity`), concurrency sweeps 1-16 for plain / MTP / DSpark / DFlash2 (`e2-v1`, single pass). Narrative in `FINDINGS.md`. |
| `phase1-followups-v0.5.22/` | nuo, 2026-10-27 | Follow-ups before policy design: 27B checkpoint-reuse output divergence (numerical path noise, no wrong state) and HiCache restore mismatch (post-prefill KV dedup), MTP init-order memory fix (27B MTP concurrency 1 -> 9-13), 27B equal-pool plain control, 8B decode profile at c64 (verify ~80%), 8B prefix hit cost (host restore 3-20x cheaper than recompute). Narrative in `FINDINGS.md`. |
| `policy-p0p1-v0.5.22/` | nuo, 2026-10-27 | Policy layer P0 (scheduler-side sensor / policy / actuator skeleton, observe mode, overhead within noise) and P1 (DSpark compact ragged-verify confidence budget on FA4: 8B w1-natural c64 1.21x plain vs 0.99x for static DSpark; small loss at low concurrency, worse p99 TPOT at c16-32). Branch `lvelf/policy-layer`. Narrative in `FINDINGS.md`. |
| `policy-p1b-v0.5.22/` | nuo, 2026-10-27 | Policy P1b: load-aware switch between full verify and DSpark's confidence planner (no gain: compact's low-concurrency gap is path overhead, not budget trimming); P1 tail latency not reproduced; planner beats every fixed verify-budget fraction at c64 by 7%+. |
| `policy-p2-v0.5.22/` | nuo, 2026-10-27 | Policy P2 memory partition: SGLang's default-off post-capture KV sizing gives 8B DSpark 60.8K KV tokens (default 38.1K, plain 58.7K); W2 c64 0.68x -> 1.08x plain, c16 0.89x -> 1.31x. Runtime VMM growth left as later work (no headroom remains on 8B). |
| `policy-p3-v0.5.22/` | nuo, 2026-10-27 | Policy P3 prefix tiering on 8B W2 with post-capture sizing everywhere: plain also grows to 90.2K KV, so DSpark stays 0.70x at c64 (corrects P2); HiCache brings DSpark's hit rate back to plain's (0.79 -> 0.875) and throughput to 0.85x; the remaining gap is concurrency (93 vs 127 running); 1.27x at c16; LFU = LRU. |
