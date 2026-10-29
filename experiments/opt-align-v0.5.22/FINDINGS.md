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

设置：
- 负载用第四、五轮的时变 phase bench，加上开环对话：
  - 对话：并发 4 / 64 交替。
  - 校对：同样的四段，换成校对数据。
  - 混合时变：对话 / 校对、并发 64 / 16 / 4 共六段。
  - 开环：泊松到达，1 / 7 个每秒交替，约 980 个请求。
- 两组 backend：
  - EAGLE3 和 DFlash 用 flashinfer，和第四、五轮一致。DFlash 的数字直接取第四、五轮（同一 checkout 逻辑、同样的 capture 后定池），policy 是最终版。
  - DSpark 的 planner 要 FA4 才能 capture compact ragged verify，所以 DSpark 和它的 plain 对照都用 FA4。两个 plain 的总时间差 1–3%。
- 每个 drafter 比四类配置：
  - 默认静态宽度；
  - 各静态宽度里最好的；
  - SGLang 自带的动态机制：EAGLE3 是 `--speculative-adaptive`，DSpark 是置信度 planner；
  - 我们的 policy：DFlash 是宽度 policy，DSpark 是 load-aware。
- EAGLE3 的静态 steps 1 / 2 / 3 / 5，对应 verify 宽度 2 / 3 / 4 / 6。
- SGLang adaptive 跑两份：
  - 默认候选 {1, 3, 5, 7}；
  - 精简版 `inputs/adaptive-lite.json`：批 < 8 用 3，8–31 在 {1, 3} 里选，32–63 在 {0, 1} 里选，≥ 64 用 0（不推测）。
- 数据：`o1-phase-benches.csv`、`o1-openloop-natural.csv`。

### 闭环 phase bench

总时间（s），括号里是相对同 backend plain 的加速：

| drafter | 配置 | 对话 | 校对 | 混合时变 |
|---|---|---:|---:|---:|
| plain（flashinfer） | | 203.2 | 214.7 | 227.9 |
| DFlash | 默认 b16 | 299.1（0.68） | 84.9（2.53） | 197.1（1.16） |
| | 最好的静态 | b3 171.4（1.19） | b16 84.9（2.53） | b4 149.8（1.52） |
| | **宽度 policy** | **162.1（1.25）** | **84.4（2.54）** | **131.1（1.74）** |
| EAGLE3 | 默认 s3 | 158.0（1.29） | 119.3（1.80） | 145.5（1.57） |
| | 最好的静态 | s2 157.2（1.29） | s5 112.1（1.92） | s3 145.5（1.57） |
| | 逐阶段最优静态（上限） | 153.1 | 112.1 | 139.3 |
| | SGLang adaptive 默认 | 600.5（0.34） | 416.8（0.52） | 545.7（0.42） |
| | SGLang adaptive 精简 | 160.4（1.27） | 142.9（1.50） | 154.8（1.47） |
| plain（FA4） | | 200.5 | 208.9 | 224.0 |
| DSpark | 满 verify（block 7，宽度 8） | 159.4（1.26） | **82.2（2.54）** | 123.6（1.81） |
| | **置信度 planner** | **133.8（1.50）** | 82.8（2.52） | **113.7（1.97）** |
| | load-aware policy | 133.7（1.50） | 82.8（2.52） | 114.3（1.96） |

"逐阶段最优静态"是每一段分别取 s1 / s2 / s3 / s5 里最快的再相加，是任何 EAGLE3 宽度切换方法在这组候选里的上限。

- **DFlash：动态宽度收益最大。**
  - 默认 b16 在对话并发 64 时只有 plain 的 0.4 倍，policy 把它变成三个负载都最好的配置。
  - 相对最好的静态宽度，对话快 5%，混合快 12%。混合时变的每一段最优宽度不同，任何单一静态宽度都会在某几段吃亏。
  - 校对上 b16 本来就最好，policy 一直停在 16，持平。
- **EAGLE3：最优 steps 也随负载和负载类型变，但可赚的空间小。**
  - 对话并发 4 时 s3 / s5 最好，并发 64 时 s2 最好；校对上每段都是 s5 最好；混合里校对段是 s5，对话并发 16 / 4 是 s3，对话并发 64 是 s2。
  - 逐阶段最优只比最好的单一静态快 0–4%，比默认 s3 快 3–6%。EAGLE3 的候选宽度只在 2–6 之间（DFlash 是 3–16），默认 s3 离每一段的最优都不远。
- **SGLang 自带的 adaptive 在 32 GB 卡上不可用。**
  - 默认候选 {1, 3, 5, 7} 要为每个候选建一套 runtime state 和 CUDA graph。capture 之后 KV 池只剩 2,273 token（静态 s3 是 63,055），同时只能跑几个请求，三个负载都只有 plain 的 0.34–0.52 倍。
  - 精简版 KV 池 45,800（−27%）。它的规则是"批越大 steps 越少"，校对上因此比静态 s5 慢 27%，对话也比 s2 / s3 略慢；对话负载里高并发时在 step 0 和 1 之间来回切了 834 次。
- **DSpark：SGLang 的置信度 planner 已经是动态宽度，我们的 load-aware policy 不需要。**
  - planner 相对满 verify，对话快 16%、混合快 8%，校对持平（−0.7%）。
  - load-aware 在 planner 和满 verify 之间切换，结果和 planner 一样（差 ≤ 0.5%），与 P1b 的结论一致。

### 开环对话

E2E p50 / p95（s），TPOT p50（ms）：

| drafter | 配置 | E2E p50 / p95 | TPOT p50 |
|---|---|---:|---:|
| plain（flashinfer） | | 3.44 / 9.42 | 14.2 |
| DFlash | 默认 b16 | 22.2 / 46.9（积压） | 46.7 |
| | 静态 b4 | **2.60 / 6.91** | 10.7 |
| | 宽度 policy | 2.99 / 8.77 | 12.2 |
| EAGLE3 | s1 / s2 | 2.57 / 7.01，2.18 / 6.04 | 10.4，8.7 |
| | 默认 s3 | **2.04 / 5.78** | 8.3 |
| | s5 | 2.23 / 6.44 | 8.9 |
| | SGLang adaptive 默认 | 166 / 280（积压） | 6.3 |
| | SGLang adaptive 精简 | 2.07 / 5.82 | 8.4 |
| plain（FA4） | | 3.53 / 9.84 | 14.7 |
| DSpark | 满 verify | 2.02 / 6.04 | 8.2 |
| | **置信度 planner** | **1.83 / 5.05** | 7.4 |
| | load-aware policy | 2.10 / 6.38 | 8.7 |

- 7 个 / 秒约为 1950 tok/s 的需求，接近这些配置的容量（DFlash 这时同时在跑约 25–31 个请求，见第五轮）。
- 三个 drafter 里只有 DSpark 的动态机制在开环下也是最好的：planner 比满 verify 的 p95 低 16%。它按每个请求的置信度决定 verify 多少，不需要运行中试探。
- DFlash 宽度 policy 的弱点仍在：比静态 b4 的 p95 高 27%（第五轮已记录，原因是接近饱和时的试探把队列推进积压）。
- EAGLE3 默认 s3 已经是开环最优，精简 adaptive 持平。
- load-aware 在开环下比 planner 和满 verify 都差（p95 比 planner 高 26%）。只跑了一次，第五轮发现开环对单次运行很敏感，原因没查。

### EAGLE3 上的宽度 policy

（排队中：policy-layer `d70c35be29` 让宽度 policy 通过 `--speculative-adaptive` 的 runtime state 切换 EAGLE3 的 steps，候选 {1, 2, 3} 和 {1, 2, 3, 5} 两组，跑同样的 phase bench 和开环。）

## O2 hybrid 显存划分

（排队中：Qwen3.5-4B EAGLE3 / DSpark，27B MTP / DSpark / DFlash2。）
