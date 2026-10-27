# Qwen3-8B v0.5.22 首轮 baseline

完成配置：8/8；测量阶段：96。

固定 tag/commit、模型 revision、seed=42、16K slot 上限；graphs/overlap 关闭，使用 Rust Unified Radix Cache。输入为固定 token IDs 的自然文本合成负载，不能视为真实业务的 acceptance 或准确率评估。这里比较的是各算法、drafter checkpoint 和原生 verify 宽度构成的整体配置。

正式 campaign 约 20.83 分钟（含每个服务启动及退出），请求计时合计 15.67 分钟。模型下载、环境安装、首次编译、pilot 和单独诊断均不计入正式性能数据。

## 低共享长输出对照

以下取 cache-off，避免把 replay 的大量命中混入 decode 对照。所有请求的输入长度 384、输出长度 256。

| 算法 | 并发 1 输出 tok/s | 并发 8 输出 tok/s | 原生 verify 宽度 | 观测 GPU 峰值 GiB* |
|---|---:|---:|---:|---:|
| plain | 83.6 | 395.2 | n/a | 21.16 |
| eagle3 | 129.3 | 642.7 | 6 | 22.19 |
| dflash | 163.2 | 791.2 | 16 | 23.74 |
| dspark | 190.4 | 903.1 | 8 | 23.67 |

## 缓存与容量

| 算法 | 长共享前缀并发 1 TTFT：cache-off → replay (ms) | 并发 1 压力 replay 命中率 |
|---|---:|---:|
| plain | 332.6 → 17.0 | 0.166% |
| eagle3 | 348.9 → 19.3 | 0.142% |
| dflash | 338.9 → 17.6 | 0.166% |
| dspark | 340.2 → 17.2 | 0.166% |

压力负载的前缀工作集约 24K，大于 16K slot；当前单点有限轮次显示缓存淘汰后命中消失，不构成完整容量曲线或稳态实验。高并发结果同时包含准入、batching 和缓存复用的影响。固定 slot 上限不等于固定总显存预算；native allocation 日志保留在 allocation-summary.json。当前仍无法独立观测 target/draft 的 readiness、重算行数及独立 draft slot 回收，不能声称已验证独立生命周期策略。

## 输出一致性与结果边界

默认 BF16 的 cache-on/off 共 1312 对比较中，498 对完整输出 IDs 不同。请求完成、长度和 slot 检查通过，不代表严格输出对齐通过。差异和首次分叉位置分别保存在 output-parity-observations.json 与 output-divergence-positions.json，尚未逐项归因。

独立 deterministic 诊断的状态和结果保存在 diagnostic-deterministic-plain/，与正式计时分开。

## 完整分项数据

## shared-long, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 44.9 | 332.6 | 0.0 | n/a | 21.10 |
| plain-cache-on | initial-empty | 72.6 | 21.0 | 84.9 | n/a | 21.10 |
| plain-cache-on | replay | 80.6 | 17.0 | 100.0 | n/a | 21.10 |
| eagle3-cache-off | initial-empty | 47.4 | 348.9 | 0.0 | 1.93 | 22.19 |
| eagle3-cache-on | initial-empty | 82.6 | 22.3 | 84.9 | 1.93 | 22.03 |
| eagle3-cache-on | replay | 92.9 | 19.3 | 100.0 | 1.92 | 22.03 |
| dflash-cache-off | initial-empty | 53.6 | 338.9 | 0.0 | 2.17 | 23.56 |
| dflash-cache-on | initial-empty | 100.9 | 21.6 | 84.9 | 2.19 | 23.61 |
| dflash-cache-on | replay | 117.4 | 17.6 | 100.0 | 2.19 | 23.61 |
| dspark-cache-off | initial-empty | 62.1 | 340.2 | 0.0 | 3.24 | 23.62 |
| dspark-cache-on | initial-empty | 136.6 | 21.4 | 84.9 | 3.30 | 23.67 |
| dspark-cache-on | replay | 168.8 | 17.2 | 100.0 | 3.30 | 23.67 |

## shared-long, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 65.2 | 2436.6 | 0.0 | n/a | 21.10 |
| plain-cache-on | initial-empty | 243.0 | 294.5 | 84.6 | n/a | 21.10 |
| plain-cache-on | replay | 543.7 | 39.3 | 100.0 | n/a | 21.10 |
| eagle3-cache-off | initial-empty | 63.8 | 2748.0 | 0.0 | 1.98 | 22.16 |
| eagle3-cache-on | initial-empty | 256.2 | 295.4 | 84.3 | 1.97 | 22.01 |
| eagle3-cache-on | replay | 548.3 | 44.5 | 100.0 | 1.97 | 22.01 |
| dflash-cache-off | initial-empty | 71.3 | 2431.3 | 0.0 | 2.25 | 23.56 |
| dflash-cache-on | initial-empty | 290.4 | 299.7 | 84.6 | 2.22 | 23.62 |
| dflash-cache-on | replay | 518.6 | 42.7 | 100.0 | 2.24 | 23.69 |
| dspark-cache-off | initial-empty | 75.0 | 2425.6 | 0.0 | 3.32 | 23.67 |
| dspark-cache-on | initial-empty | 335.7 | 333.6 | 84.4 | 3.28 | 23.66 |
| dspark-cache-on | replay | 1034.9 | 40.8 | 100.0 | 3.37 | 23.66 |

## multiturn-branch, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 66.4 | 115.2 | 0.0 | n/a | 21.15 |
| plain-cache-on | initial-empty | 77.1 | 31.1 | 69.7 | n/a | 21.05 |
| plain-cache-on | replay | 82.2 | 16.5 | 99.9 | n/a | 21.05 |
| eagle3-cache-off | initial-empty | 94.3 | 121.1 | 0.0 | 2.75 | 22.18 |
| eagle3-cache-on | initial-empty | 115.6 | 32.8 | 69.6 | 2.65 | 21.85 |
| eagle3-cache-on | replay | 126.7 | 18.7 | 99.9 | 2.64 | 21.85 |
| dflash-cache-off | initial-empty | 118.9 | 116.3 | 0.0 | 3.45 | 23.55 |
| dflash-cache-on | initial-empty | 151.8 | 31.6 | 69.7 | 3.25 | 23.41 |
| dflash-cache-on | replay | 173.8 | 17.0 | 99.9 | 3.25 | 23.41 |
| dspark-cache-off | initial-empty | 120.9 | 116.2 | 0.0 | 3.64 | 23.62 |
| dspark-cache-on | initial-empty | 160.6 | 31.5 | 69.7 | 3.58 | 23.51 |
| dspark-cache-on | replay | 185.1 | 16.8 | 99.9 | 3.58 | 23.51 |

## multiturn-branch, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 151.3 | 572.3 | 0.0 | n/a | 21.14 |
| plain-cache-on | initial-empty | 317.5 | 165.9 | 69.7 | n/a | 21.16 |
| plain-cache-on | replay | 544.5 | 33.3 | 99.9 | n/a | 21.16 |
| eagle3-cache-off | initial-empty | 190.1 | 582.4 | 0.0 | 2.71 | 22.09 |
| eagle3-cache-on | initial-empty | 399.2 | 173.4 | 69.6 | 2.72 | 22.09 |
| eagle3-cache-on | replay | 680.5 | 37.1 | 99.9 | 2.65 | 22.09 |
| dflash-cache-off | initial-empty | 206.1 | 583.5 | 0.0 | 3.21 | 23.67 |
| dflash-cache-on | initial-empty | 418.9 | 169.3 | 69.7 | 3.25 | 23.74 |
| dflash-cache-on | replay | 744.3 | 34.3 | 99.9 | 3.26 | 23.74 |
| dspark-cache-off | initial-empty | 215.1 | 567.6 | 0.0 | 3.57 | 23.67 |
| dspark-cache-on | initial-empty | 475.4 | 169.6 | 69.7 | 3.58 | 23.67 |
| dspark-cache-on | replay | 929.0 | 35.1 | 99.9 | 3.53 | 23.67 |

## capacity-pressure, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 44.6 | 334.9 | 0.0 | n/a | 21.10 |
| plain-cache-on | initial-empty | 44.6 | 338.2 | 0.2 | n/a | 21.12 |
| plain-cache-on | replay | 44.6 | 338.3 | 0.2 | n/a | 21.12 |
| eagle3-cache-off | initial-empty | 55.6 | 350.3 | 0.0 | 2.80 | 22.04 |
| eagle3-cache-on | initial-empty | 55.8 | 352.5 | 0.1 | 2.84 | 22.04 |
| eagle3-cache-on | replay | 55.9 | 353.0 | 0.1 | 2.85 | 22.04 |
| dflash-cache-off | initial-empty | 60.5 | 341.4 | 0.0 | 2.98 | 23.56 |
| dflash-cache-on | initial-empty | 60.2 | 344.5 | 0.2 | 3.00 | 23.61 |
| dflash-cache-on | replay | 60.1 | 344.8 | 0.2 | 2.98 | 23.61 |
| dspark-cache-off | initial-empty | 61.9 | 341.6 | 0.0 | 3.25 | 23.66 |
| dspark-cache-on | initial-empty | 61.7 | 343.4 | 0.2 | 3.25 | 23.67 |
| dspark-cache-on | replay | 61.7 | 343.8 | 0.2 | 3.25 | 23.67 |

## capacity-pressure, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 67.1 | 2442.8 | 0.0 | n/a | 21.10 |
| plain-cache-on | initial-empty | 73.4 | 2207.4 | 13.5 | n/a | 21.10 |
| plain-cache-on | replay | 70.4 | 2462.6 | 8.1 | n/a | 21.10 |
| eagle3-cache-off | initial-empty | 72.6 | 2494.6 | 0.0 | 2.84 | 22.04 |
| eagle3-cache-on | initial-empty | 84.1 | 1845.6 | 16.3 | 2.84 | 22.01 |
| eagle3-cache-on | replay | 78.7 | 2352.3 | 10.9 | 2.84 | 22.01 |
| dflash-cache-off | initial-empty | 76.7 | 2378.5 | 0.0 | 3.10 | 23.60 |
| dflash-cache-on | initial-empty | 81.9 | 2133.1 | 8.2 | 3.06 | 23.60 |
| dflash-cache-on | replay | 75.8 | 2393.2 | 0.2 | 3.06 | 23.60 |
| dspark-cache-off | initial-empty | 77.4 | 2364.4 | 0.0 | 3.27 | 23.67 |
| dspark-cache-on | initial-empty | 81.4 | 2314.0 | 5.5 | 3.29 | 23.67 |
| dspark-cache-on | replay | 78.9 | 2383.8 | 2.9 | 3.31 | 23.67 |

## unique-decode, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 83.6 | 33.6 | 0.0 | n/a | 20.92 |
| plain-cache-on | initial-empty | 83.5 | 34.6 | 1.8 | n/a | 20.92 |
| plain-cache-on | replay | 84.1 | 16.4 | 99.7 | n/a | 20.92 |
| eagle3-cache-off | initial-empty | 129.3 | 35.5 | 0.0 | 2.53 | 21.72 |
| eagle3-cache-on | initial-empty | 129.2 | 36.5 | 1.6 | 2.51 | 21.72 |
| eagle3-cache-on | replay | 131.8 | 18.6 | 99.5 | 2.53 | 21.72 |
| dflash-cache-off | initial-empty | 163.2 | 34.3 | 0.0 | 2.81 | 23.28 |
| dflash-cache-on | initial-empty | 162.3 | 35.3 | 1.8 | 2.80 | 23.28 |
| dflash-cache-on | replay | 164.2 | 17.1 | 99.7 | 2.80 | 23.28 |
| dspark-cache-off | initial-empty | 190.4 | 34.5 | 0.0 | 3.38 | 23.34 |
| dspark-cache-on | initial-empty | 188.6 | 35.3 | 1.8 | 3.35 | 23.35 |
| dspark-cache-on | replay | 191.1 | 17.0 | 99.7 | 3.35 | 23.35 |

## unique-decode, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 395.2 | 184.6 | 0.0 | n/a | 21.11 |
| plain-cache-on | initial-empty | 394.0 | 188.9 | 1.8 | n/a | 21.14 |
| plain-cache-on | replay | 601.6 | 35.7 | 99.7 | n/a | 21.14 |
| eagle3-cache-off | initial-empty | 642.7 | 78.7 | 0.0 | 2.49 | 21.95 |
| eagle3-cache-on | initial-empty | 656.2 | 115.1 | 1.6 | 2.54 | 21.98 |
| eagle3-cache-on | replay | 874.4 | 37.7 | 99.5 | 2.52 | 21.98 |
| dflash-cache-off | initial-empty | 791.2 | 171.2 | 0.0 | 2.86 | 23.65 |
| dflash-cache-on | initial-empty | 764.4 | 189.9 | 1.8 | 2.80 | 23.59 |
| dflash-cache-on | replay | 1019.5 | 41.8 | 99.7 | 2.87 | 23.59 |
| dspark-cache-off | initial-empty | 903.1 | 146.1 | 0.0 | 3.29 | 23.61 |
| dspark-cache-on | initial-empty | 923.8 | 148.2 | 1.8 | 3.33 | 23.64 |
| dspark-cache-on | replay | 1288.8 | 40.4 | 99.7 | 3.37 | 23.64 |

*GPU peak includes other processes, preallocated pools and sampling limitations. Public cached_tokens is joint coverage; it does not measure independent target/draft readiness or recompute rows.

Default BF16 cache-on/off output differences, where output IDs are available, are retained in output-parity-observations.json. They are observations requiring diagnosis, not a pass/fail accuracy claim.
