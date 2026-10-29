# 报告素材整理

为写报告和做 PPT 整理的结果。每份文档每节大致对应一页 PPT：现象 → 怎么发现的 → 有多严重 → 对 policy 的含义。所有数字都来自仓库里 `experiments/` 下已归档的结果，每份文档末尾列了来源。

| 文件 | 内容 | 状态 |
|---|---|---|
| [`01-stress-test.md`](01-stress-test.md) | 第一部分：压力测试。推测解码在显存压力下哪里输、输多少，以及因此 policy 该往哪改 | 完成 |
| [`01-stress-test-figures.md`](01-stress-test-figures.md) | 压力测试四个场景（长前缀、多轮对话、显存容量有限、短输入长输出）的图，覆盖 4B / 8B / 27B | 完成 |
| [`02-optimizations.md`](02-optimizations.md) | 第二部分：三项优化（动态 verify 宽度、hybrid 显存划分、HiCache）做了什么，以及在 EAGLE3 / DFlash / DSpark 上的结果图 | 完成 |
| [`make_figures.py`](make_figures.py) | 从 `experiments/` 的归档 CSV 生成 `pics/` 下所有图：`uv run --no-project --with matplotlib python report/make_figures.py`。`latex/` 存在时同时输出 PDF 到 `latex/figures/` | |
| `latex/` | LaTeX 报告，**独立的 git 仓库**，和 Overleaf 互相 pull/push；不在本仓库里（本地 `.git/info/exclude` 排除） | |

## 分支检查（2026-10-28）

`git fetch` 之后检查了 main 以外的所有分支：

| 分支 | 是否有当前分支没有的实验结果 | 说明 |
|---|---|---|
| `origin/fix/hicache-mamba-layer-first`（队友） | 没有，已合入 | 队友的 8B / 4B 基线和 DSpark 状态管理实验（`experiments/qwen3-8b-prefix-baseline-v0.5.22/`、`qwen35-4b-hybrid-prefix-baseline-v0.5.22/`、`dspark-state-management/`）都已经在 `lvelf/policy-layer` 里。其中属于压力测试的部分已经写进 `01-stress-test.md` |
| `origin/fix/bigram-prefix-raw-token-limit`（队友） | 没有 | bigram 前缀匹配的 bug 修复，加上一次 upstream 合并，不含实验 |
| `origin/research/joint-prefix-cache` | 没有 | 停在基线 commit `baa9befa38` |
| `lvelf/benchmark`、`lvelf/dflash-width` | 没有 | 都已包含在 `lvelf/policy-layer` 里 |
