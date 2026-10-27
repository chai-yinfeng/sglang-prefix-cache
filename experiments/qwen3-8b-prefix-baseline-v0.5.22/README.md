# Qwen3-8B prefix-cache baseline, v0.5.22

This directory is independent of `../spec-prefix-cache/`. The old experiment,
its environments, and the PR worktree are not modified by this campaign.

Source: v0.5.22, baa9befa381511cd4d56f83980951e5a45fffc6d, no runtime patch.
The known bigram raw-token-boundary limitation is part of this baseline.

The first group compares plain, EAGLE3, DFlash, and DSpark with cache on/off,
all using full-attention drafts. Window experiments and Qwen3.5 hybrid models
are outside this run. Model revisions are frozen in models.json.

The target/EAGLE3 checkpoints reuse the existing HF snapshots. The
DSpark checkpoint is DeepSeek Qwen3-8B block7. DFlash uses z-lab's native
Qwen3-8B DFlash-b16 checkpoint after the DeepSeek markov_rank=0 variant failed
the unmodified v0.5.22 loader. Failed pilot artifacts and model-selection-attempts.json
retain that evidence. Resolved verify widths are recorded, not assumed equal
across algorithms; algorithm, checkpoint and width effects are not isolated.
The serving environment is an independent copy of the PR environment with v0.5.22 and its
pinned dependencies installed into the copy. environment.json records actual
packages and source identity. Installation and first kernel compilation are
not included in client timing.

## Workloads and controls

prepare.py freezes exact token IDs from the three public repository documents
listed in config.json. This is synthetic natural-text workload construction,
not a quality benchmark or a representative production acceptance estimate.
Each group is distinguished near the start of its prompt. Chat delimiters and
the same non-thinking assistant prefix are included in all algorithms' input.

* shared-long: 2 document groups, 8 suffixes per group, input 4096+128,
  output 32; interleaved groups.
* multiturn-branch: 8 conversations, 3 progressive stages plus a sibling
  continuation; each stage completes before the next. Supplied reference
  answers are fixed across algorithms; generated text does not feed back.
* capacity-pressure: 6 groups, each with a 4096-token prefix and 128-token
  suffix; 3 round-robin cycles; output 32. Prefix working set exceeds the
  configured 16384-slot capacity. This is finite churn, not proof of steady state.
* unique-decode: 16 distinct inputs of 256+128 tokens, output 256.

Concurrency 1/8 means maximum client requests in flight. Server batch sizes
remain scheduler decisions. A single small, mixed-shape warmup is excluded.
Before every workload/concurrency pair, the cache is flushed once; the first
pass starts empty but may reuse earlier requests in that pass. Cache-on then
replays the identical saved input sequence; cache-off has no redundant replay.
One seed (42) is used for calibration. Graphs and overlap are disabled for this
controlled baseline; it does not claim fully tuned serving performance.

This experiment holds the slot cap fixed across algorithms and reports actual
memory cost. It is not an equal-memory-budget comparison. A freed logical
draft resource is not assumed to have freed independently allocatable capacity.
SGLang's default Rust UnifiedTreeCore is explicitly selected for the campaign.
The slot limit also constrains admission: eight distinct 4224-token requests
cannot all reside fully in a 16384-slot pool. Shared prefixes can change the
number of requests that fit, so high-concurrency throughput includes admission
and batching effects, not just time saved in the prefix lookup or prefill.

## Execution and artifacts

Run from the repository root, using this directory's .venv/bin/python:

    python work_dirs/qwen3-8b-prefix-baseline-v0.5.22/prepare.py
    python work_dirs/qwen3-8b-prefix-baseline-v0.5.22/run_campaign.py --pilot --variants plain-cache-on
    python work_dirs/qwen3-8b-prefix-baseline-v0.5.22/run_campaign.py

Pilot outputs are separate from timed full-fixture runs. Each variant has its
exact command, resolved server_info, server log, per-request results, metrics
snapshots, GPU observations, and completion/failure marker. Successful variants
are skipped on restart; an incomplete variant restarts from a new server.

Public cached_tokens reports joint coverage, not independent target/draft
coverage or actual recomputed rows. Prometheus snapshots retain the available,
evictable, used-slot and retraction counters exposed by the baseline. Values
may lag short phases; per-request response fields and raw server logs are kept.
GPU used bytes include other processes and preallocated pools. Stream-amortized
TPOT uses first-to-last streamed-token delivery and is not GPU iteration time.
Acceptance counters are final per-request counters, not differences of a
server-lifetime acceptance ratio. Default BF16 output mismatches are reported
as observations rather than silently treated as proof of KV corruption.

The streaming client is copied from the old experiment's
pr-local-review/client-sources/token_workload.py; provenance and hashes are
retained. Only this local harness and its files are added; SGLang is unchanged.
