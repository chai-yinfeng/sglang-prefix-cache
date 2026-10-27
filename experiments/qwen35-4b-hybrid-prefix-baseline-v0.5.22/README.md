# Qwen3.5-4B hybrid prefix-cache baseline, v0.5.22

Separate from both earlier work_dirs campaigns. Runtime source is unchanged v0.5.22.
Reuses the previous pinned environment without modifying it. Fixed snapshot hashes in models.json.

Target: Qwen3.5-4B, 24 GDN + 8 FA layers; draft: native z-lab DFlash, 5 SWA (4096 window) + 1 FA, block 16. Plain and DFlash, each cache on/off. This is a native hybrid resource baseline, not a controlled replacement of attention layers and not an isolated attention performance comparison against Qwen3-8B.

Four workloads, exact lengths, c1/c8, seed, warmup, cache flush/replay, BF16 weights/KV, graph/overlap disabled and 16K KV slot capacity follow the Qwen3-8B experiment. Token IDs are regenerated with this target tokenizer, and fixed thereafter. Synthetic histories use supplied reference replies rather than model outputs. Long prompts are 4224 tokens, only 128 beyond the draft window; the run does not characterize very long windows.

Explicit Mamba state pool: 64 slots, FP32 state, extra_buffer, track interval 256. This is a second independently constrained resource, not equivalent to 16K token slots. Report native allocation and all available resource gauges. Native draft layer SWA does not imply independent eviction or a compact SWA-only allocator: resolved allocations and actual Unified Radix components must be checked. No forced draft-window override is applied to the mixed-attention checkpoint.

FlashInfer for FA; Triton for GDN prefill/decode/verify and Mamba kernels, explicitly fixed across plain/spec. Startup/JIT and pilot excluded from timed runs. Exact launch and resolved args, server logs, response token IDs, cache metrics, memory and timings retained. Output ID differences are observations, not automatically accuracy success or corruption. Request/slot audits do not replace accuracy evaluation.

Run prepare.py then run_campaign.py --pilot --variants plain-cache-on dflash-cache-on; if successful run run_campaign.py. Pilot artifacts remain separate. No runtime fixes are authorized by this benchmark request; failures are retained and diagnosed before changing framework behavior.

Initial startup-only attempt rejected --language-model-only for Qwen3.5 in v0.5.22; retained in pilot-unsupported-language-model-only/. Removed that optional flag, kept the native conditional-generation model including its vision tower; all benchmark requests are text token IDs. No model configuration or runtime patch. GPU totals therefore include native loaded model components.

Pilot confirmed native runtime behavior: FULL+MAMBA components only (hybrid_swa=False), draft uses a dense full-capacity 16384-row KV pool despite its per-layer sliding attention mask. Mamba state pool logs ~3.12 GiB; DFlash adds ~6.75 GiB intermediate SSM state cache and ~0.06 GiB intermediate conv-window cache, in addition to draft weights (~1.36 GiB) and KV (~0.375 GiB). These temporary verify states must not be classified as persistent prefix storage. Plain replay of 4224-token prompts matched 4096-token GDN checkpoints. This checkpoint granularity differs from FA-only prefix matching.

## Completed campaign

4 variants, 48 phases, 984 formal requests, 10.58 minutes including launch/exit; 8.54 minutes client measurements. Request/artifact integrity passed; 115/656 full-output comparisons differ (zero first-token divergences), strict output parity did not pass. No claim of full accuracy validation. Both cache-on variants reached zero available Mamba state slots in sampled metrics. All server parents exited. Final GPU total ~4067 MiB belongs to the overall device observation and is not asserted equal to the pre-run desktop baseline; compute-app query returned empty.

Regenerate summaries with summarize.py, audit.py and resource_audit.py (run summarize.py again after resource_audit.py to include resource observations). Exact launch commands reside in runs/*/launch.json. All experiment artifacts remain locally ignored by git; there is no tracked source change or new commit.
