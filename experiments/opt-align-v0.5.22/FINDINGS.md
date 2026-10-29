# Optimization 对齐：三项优化在 EAGLE3 / DFlash / DSpark 上（2026-10-29）

报告第 3 节的三项优化，之前只在部分 drafter 上测过：
- 动态 verify 宽度只有 DFlash。
- hybrid 显存划分只有 MTP 和 DFlash。
- HiCache 只有 plain 和 DSpark。

这一轮把三项优化都补到三组 drafter 上，计划见 `MyDocs/Plan/08-optimization-alignment.md`。

共同设置：
- RTX 5090 32 GB；policy-layer checkout `78934cec2a`。
- capture 后定池（`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`）全部打开。
- 同一张表内 attention backend 和 page size 一致。
- drafter 用 cookbook 配置：EAGLE3 3/1/4、DFlash b16、DSpark block7。
- 除注明外单次运行。原始数据在 benchmark worktree 的 `work_dirs/stress-qwen3-8b/runs/o*`。

## O3 HiCache（Qwen3-8B）

设置：
- 沿用 P3：FA4 + page 64。
- HiCache 用 `direct` + `layer_first`、`write_through`，主机池 12 GB。
- 负载 W2 共享前缀，以及新的多轮对话（回填输出，见 `stress-gapfill-v0.5.22/`）；并发 16 / 64。
- 数据：`o3-hicache.csv`。

各配置的 KV 容量（token）：plain 90,240，EAGLE3 76,224（−16%），DSpark 61,184（−32%），DFlash 53,376（−41%）。

并发 64，单位 tok/s，括号里是相对不开 HiCache 的 plain 的倍数：

| 负载 | | plain | EAGLE3 | DFlash | DSpark |
|---|---|---:|---:|---:|---:|
| W2 | 不开 HiCache | 1522 | 1493（0.98） | 661（0.43） | 1077（0.71） |
| | 开 HiCache | 1508（0.99） | 1553（1.02） | 820（0.54） | 1291（0.85） |
| | HiCache 带来的提升 | −1% | **+4%** | **+24%** | **+20%** |
| | 命中率（不开 → 开） | 0.886 → 0.889 | 0.859 → 0.878 | 0.746 → 0.887 | 0.790 → 0.875 |
| 多轮对话 | 不开 HiCache | 2852 | 3161（1.11） | 1533（0.54） | 2840（1.00） |
| | 开 HiCache | 2795（0.98） | 3367（1.18） | 1697（0.60） | 3270（1.15） |
| | HiCache 带来的提升 | −2% | **+7%** | **+11%** | **+15%** |
| | 命中率（不开 → 开） | 0.73 → 0.79 | 0.60 → 0.81 | 0.18 → 0.87 | 0.34 → 0.85 |

- **四个配置都测过之后**：HiCache 的收益和该配置被推测挤掉的 KV 成正比。
  - KV 少 16% 的 EAGLE3 提升 4–7%，少 32–41% 的 DSpark / DFlash 提升 11–24%。
  - plain 的池子本来就装得下，HiCache 没有收益（−1% 到 −2%）。
- **命中率都补回来了**：开 HiCache 后，四个配置的命中率都回到 0.79–0.89，和 plain 接近。被挤出 GPU 的前缀和对话历史从 CPU 取回，不用重算。
- **补不回的部分是计算和并发**：
  - DFlash 开了 HiCache 仍只有 plain 的 0.54–0.60 倍。原因是宽度 16 的 verify，以及 KV 只够同时跑 34 个 W2 请求（plain 64 个）。
  - DSpark 在 W2 上是 0.85 倍：同时运行的请求 42 个，plain 64 个。
- **多轮对话上，开了 HiCache 的推测配置全部回到 plain 之上或附近**：EAGLE3 1.18 倍，DSpark 1.15 倍。这是最典型的 HiCache 场景，两轮之间的历史被挤出 GPU，下一轮再取回。
- **并发 16 没有提升**（±3%），因为什么都没被挤出去。但 W2 上推测配置开 HiCache 后 TTFT 中位数下降：DFlash 372 → 88 ms，EAGLE3 340 → 188 ms。
- **复现**：DSpark 在 W2 并发 64 不开 / 开 HiCache 是 1077 / 1291 tok/s，P3 是 1065 / 1286，差在 1% 以内。
- 限制：主机池 12 GB（81K token），小于 plain 的设备池（90K），这对 plain 不利，SGLang 也打印了警告；推测配置的设备池更小，不受影响。单次运行。

## O1 动态 verify 宽度

（进行中：EAGLE3 静态 steps 1 / 2 / 3 / 5、SGLang adaptive 默认和精简；DSpark 满 verify、planner、load-aware policy；放在第四、五轮的 phase bench 和开环负载上。）

## O2 hybrid 显存划分

（排队中：Qwen3.5-4B EAGLE3 / DSpark，27B MTP / DSpark / DFlash2。）
