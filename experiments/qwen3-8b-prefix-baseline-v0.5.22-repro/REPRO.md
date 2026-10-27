# Reproduction of the teammate's Qwen3-8B v0.5.22 baseline (2026-10-27)

Source: the teammate's `work_dirs/qwen3-8b-prefix-baseline-v0.5.22/` harness (prepare.py, run_campaign.py,
client.py, summarize.py, audit.py), copied unchanged. Results of the original run stay in
`/home/nuo/projects/sglang-prefix-cache/work_dirs/qwen3-8b-prefix-baseline-v0.5.22/runs`.

## Environment differences from the original run
- Runs from a detached worktree `/home/nuo/projects/sglang-bench-v0.5.22` at baa9befa38, because
  run_campaign.py asserts HEAD == baa9befa38 and an empty `git status --porcelain`.
- `.venv` is a symlink to the copied venv; its script shebangs were rewritten from /home/blackmagic paths.
  sglang itself is imported via PYTHONPATH=<worktree>/python (set by run_campaign.py).
- Rust 1.92 installed via rustup in ~/.cargo; the radix-tree extension is built on first import and
  cached in ~/.cache/sglang/rust_extensions. run_campaign.py must be launched with ~/.cargo/bin on PATH
  (the loader fingerprints `cargo --version` on every start).
- Models downloaded at the pinned revisions in models.json into ~/.cache/huggingface; models/ links there.
- Same GPU class (RTX 5090 32GB) and driver 610.88.

## Outcome
- prepare.py regenerated all 4 input fixtures byte-identically (sha256 match).
- 8/8 variants, 96/96 phases, 1968 requests, audit complete, 0 slot-gauge violations.
- Cell-by-cell vs original (`comparison-vs-teammate.txt`, mine/theirs median ratio):
  output tok/s 1.002, TTFT 0.999, tokens/verify 1.000. Concurrency-1 cells agree within about 1%;
  outliers are concurrency-8 cells (e.g. dspark shared-long c8 replay 1035 vs 770 tok/s,
  capacity-pressure c8 hit 3-16%), i.e. scheduling nondeterminism with 16-18 requests per cell.
- Default BF16 cache-on/off output mismatches: 498/1312 (original 489/1312), spread evenly across
  plain/eagle3/dflash/dspark (112-129 of 328 each), so not speculative-decoding specific.

## Commands
    cd /home/nuo/projects/sglang-bench-v0.5.22
    PATH=$HOME/.cargo/bin:$PATH work_dirs/qwen3-8b-prefix-baseline-v0.5.22/.venv/bin/python \
        work_dirs/qwen3-8b-prefix-baseline-v0.5.22/run_campaign.py
    .venv/bin/python summarize.py; .venv/bin/python audit.py   # from the experiment dir
