# Experiment results

Results of the joint prefix-cache / speculative-decoding study, all on SGLang v0.5.22
(`baa9befa381511cd4d56f83980951e5a45fffc6d`, no runtime patches) and an RTX 5090 32GB.

Only result documents and summary tables are kept in git. Harness scripts, input fixtures and raw
per-run artifacts (server logs, per-request results, metric samples) stay in each machine's
gitignored `work_dirs/<same directory name>/`; links inside these documents to `runs/`, `inputs/`,
or `*.json` evidence files point there.

| Directory | Run by | What it contains |
|---|---|---|
| `qwen3-8b-prefix-baseline-v0.5.22/` | teammate (blackmagic) | Qwen3-8B plain / EAGLE3 / DFlash / DSpark, cache on/off, 4 workloads, c1/c8. |
| `qwen3-8b-prefix-baseline-v0.5.22-repro/` | nuo, 2026-10-27 | Unmodified re-run of the row above on a second machine. `REPRO.md` lists environment differences; `comparison-vs-teammate.txt` compares all 96 phases (median ratio 1.00 for tok/s, TTFT, tokens/verify). |
| `qwen35-4b-hybrid-prefix-baseline-v0.5.22/` | teammate (blackmagic) | Qwen3.5-4B (24 GDN + 8 FA) plain / DFlash, cache on/off, Mamba state pool observations. |
| `dspark-state-management/` | teammate (blackmagic) | DSpark state audit, GPU capacity pilot, Qwen3-8B 8K vs 16K KV and Qwen3.8-27B NVFP4 KV/state-slot sweeps, failed CPU-offload probes. Start with `PROGRESS.md`, `RESULTS.md`, `gpu-workloads/ANALYSIS.md`. |
| `stress-test-summary.md` | nuo, 2026-10-27 | **Phase-1 conclusion**: where speculative decoding loses under memory pressure and the candidate phase-2 directions (opportunity map). Start here. |
| `stress-qwen3-8b-v0.5.22/` | nuo, 2026-10-27 | Qwen3-8B stress test with cookbook-style config (CUDA graph + overlap on): memory anatomy, concurrency sweeps 1-64 on decode / shared-prefix / natural-chat workloads, equal-KV-capacity plain control, `--speculative-adaptive`. `e2-v1` first pass, `e2-rep2` repeat (median ratio 1.000). Narrative in `FINDINGS.md`. |
| `stress-qwen38-27b-v0.5.22/` | nuo, 2026-10-27 | Qwen3.8-27B NVFP4 hybrid stress test: memory anatomy (`e1-probe`), capacity tuning (`e1-capacity`), concurrency sweeps 1-16 for plain / MTP / DSpark / DFlash2 (`e2-v1`, single pass). Narrative in `FINDINGS.md`. |
