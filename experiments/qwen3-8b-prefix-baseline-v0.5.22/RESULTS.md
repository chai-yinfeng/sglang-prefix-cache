# Qwen3-8B v0.5.22 首轮 baseline

完成配置：8/8；测量阶段：96。

固定 tag/commit、模型 revision、seed=42、16K slot 上限；graphs/overlap 关闭，使用 Rust Unified Radix Cache。输入为固定 token IDs 的自然文本合成负载，不能视为真实业务的 acceptance 或准确率评估。这里比较的是各算法、drafter checkpoint 和原生 verify 宽度构成的整体配置。

正式 campaign 约 19.64 分钟（含每个服务启动及退出），请求计时合计 15.71 分钟。模型下载、环境安装、首次编译、pilot 和单独诊断均不计入正式性能数据。

## 低共享长输出对照

以下取 cache-off，避免把 replay 的大量命中混入 decode 对照。所有请求的输入长度 384、输出长度 256。

| 算法 | 并发 1 输出 tok/s | 并发 8 输出 tok/s | 原生 verify 宽度 | 观测 GPU 峰值 GiB* |
|---|---:|---:|---:|---:|
| plain | 83.4 | 394.8 | n/a | 21.26 |
| eagle3 | 130.1 | 640.6 | 6 | 22.34 |
| dflash | 163.4 | 748.6 | 16 | 23.99 |
| dspark | 189.9 | 905.5 | 8 | 23.92 |

## 缓存与容量

| 算法 | 长共享前缀并发 1 TTFT：cache-off → replay (ms) | 并发 1 压力 replay 命中率 |
|---|---:|---:|
| plain | 333.0 → 16.7 | 0.166% |
| eagle3 | 349.2 → 19.2 | 0.142% |
| dflash | 341.3 → 17.7 | 0.166% |
| dspark | 340.8 → 17.4 | 0.166% |

压力负载的前缀工作集约 24K，大于 16K slot；当前单点有限轮次显示缓存淘汰后命中消失，不构成完整容量曲线或稳态实验。高并发结果同时包含准入、batching 和缓存复用的影响。固定 slot 上限不等于固定总显存预算；native allocation 日志保留在 allocation-summary.json。当前仍无法独立观测 target/draft 的 readiness、重算行数及独立 draft slot 回收，不能声称已验证独立生命周期策略。

## 输出一致性与结果边界

默认 BF16 的 cache-on/off 共 1312 对比较中，489 对完整输出 IDs 不同。请求完成、长度和 slot 检查通过，不代表严格输出对齐通过。差异和首次分叉位置分别保存在 output-parity-observations.json 与 output-divergence-positions.json，尚未逐项归因。

单独的 Triton deterministic 对照使用两个此前发生分叉的 plain 输入，共 6 请求、每请求输出 256 tokens；真实 replay 命中为 [383, 383]，相对 cache-off 的差异数为 {'initial-empty': 0, 'replay': 0}。它仅覆盖这两个 plain 输入；切换后端的对照不能排除 FlashInfer 特有问题或 speculative 路径缺陷，也不能替代完整准确率验证。诊断不进入上面的性能数据。

## 完整分项数据

## shared-long, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 44.8 | 333.0 | 0.0 | n/a | 21.20 |
| plain-cache-on | initial-empty | 72.5 | 21.0 | 84.9 | n/a | 21.20 |
| plain-cache-on | replay | 80.6 | 16.7 | 100.0 | n/a | 21.20 |
| eagle3-cache-off | initial-empty | 47.6 | 349.2 | 0.0 | 1.93 | 22.25 |
| eagle3-cache-on | initial-empty | 81.8 | 22.5 | 84.9 | 1.93 | 22.26 |
| eagle3-cache-on | replay | 92.4 | 19.2 | 100.0 | 1.92 | 22.26 |
| dflash-cache-off | initial-empty | 53.5 | 341.3 | 0.0 | 2.17 | 23.81 |
| dflash-cache-on | initial-empty | 100.6 | 21.6 | 84.9 | 2.19 | 23.86 |
| dflash-cache-on | replay | 116.9 | 17.7 | 100.0 | 2.19 | 23.86 |
| dspark-cache-off | initial-empty | 61.9 | 340.8 | 0.0 | 3.24 | 23.87 |
| dspark-cache-on | initial-empty | 136.5 | 21.4 | 84.9 | 3.30 | 23.92 |
| dspark-cache-on | replay | 167.9 | 17.4 | 100.0 | 3.30 | 23.92 |

## shared-long, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 65.4 | 2424.0 | 0.0 | n/a | 21.19 |
| plain-cache-on | initial-empty | 242.3 | 295.3 | 84.6 | n/a | 21.19 |
| plain-cache-on | replay | 544.7 | 37.8 | 100.0 | n/a | 21.19 |
| eagle3-cache-off | initial-empty | 63.6 | 2751.7 | 0.0 | 1.98 | 22.26 |
| eagle3-cache-on | initial-empty | 252.6 | 293.8 | 84.1 | 1.96 | 22.26 |
| eagle3-cache-on | replay | 543.5 | 45.3 | 100.0 | 1.95 | 22.26 |
| dflash-cache-off | initial-empty | 71.0 | 2427.4 | 0.0 | 2.25 | 23.81 |
| dflash-cache-on | initial-empty | 290.0 | 295.5 | 84.4 | 2.27 | 23.87 |
| dflash-cache-on | replay | 546.5 | 39.8 | 100.0 | 2.26 | 23.95 |
| dspark-cache-off | initial-empty | 76.4 | 2352.6 | 0.0 | 3.30 | 23.92 |
| dspark-cache-on | initial-empty | 349.0 | 332.0 | 84.6 | 3.24 | 23.91 |
| dspark-cache-on | replay | 770.3 | 39.2 | 100.0 | 3.28 | 23.91 |

## multiturn-branch, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 66.4 | 113.6 | 0.0 | n/a | 21.24 |
| plain-cache-on | initial-empty | 77.0 | 31.1 | 69.7 | n/a | 21.15 |
| plain-cache-on | replay | 82.1 | 16.4 | 99.9 | n/a | 21.15 |
| eagle3-cache-off | initial-empty | 94.5 | 119.7 | 0.0 | 2.75 | 22.28 |
| eagle3-cache-on | initial-empty | 115.4 | 32.9 | 69.6 | 2.65 | 22.10 |
| eagle3-cache-on | replay | 126.6 | 18.8 | 99.9 | 2.64 | 22.10 |
| dflash-cache-off | initial-empty | 118.5 | 116.5 | 0.0 | 3.45 | 23.80 |
| dflash-cache-on | initial-empty | 151.5 | 31.7 | 69.7 | 3.25 | 23.66 |
| dflash-cache-on | replay | 172.7 | 17.2 | 99.9 | 3.25 | 23.66 |
| dspark-cache-off | initial-empty | 120.8 | 116.5 | 0.0 | 3.64 | 23.87 |
| dspark-cache-on | initial-empty | 160.2 | 31.7 | 69.7 | 3.58 | 23.76 |
| dspark-cache-on | replay | 184.1 | 17.0 | 99.9 | 3.58 | 23.76 |

## multiturn-branch, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 150.8 | 572.4 | 0.0 | n/a | 21.24 |
| plain-cache-on | initial-empty | 317.2 | 166.2 | 69.7 | n/a | 21.26 |
| plain-cache-on | replay | 544.5 | 33.6 | 99.9 | n/a | 21.26 |
| eagle3-cache-off | initial-empty | 186.9 | 584.6 | 0.0 | 2.72 | 22.34 |
| eagle3-cache-on | initial-empty | 390.6 | 173.3 | 69.6 | 2.70 | 22.34 |
| eagle3-cache-on | replay | 678.8 | 37.8 | 99.9 | 2.72 | 22.34 |
| dflash-cache-off | initial-empty | 204.7 | 571.9 | 0.0 | 3.30 | 23.92 |
| dflash-cache-on | initial-empty | 432.8 | 169.6 | 69.7 | 3.39 | 23.99 |
| dflash-cache-on | replay | 782.4 | 35.6 | 99.9 | 3.38 | 23.99 |
| dspark-cache-off | initial-empty | 217.0 | 569.8 | 0.0 | 3.62 | 23.92 |
| dspark-cache-on | initial-empty | 464.8 | 169.2 | 69.7 | 3.54 | 23.92 |
| dspark-cache-on | replay | 912.6 | 34.2 | 99.9 | 3.53 | 23.92 |

## capacity-pressure, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 44.7 | 335.1 | 0.0 | n/a | 21.20 |
| plain-cache-on | initial-empty | 44.5 | 338.6 | 0.2 | n/a | 21.21 |
| plain-cache-on | replay | 44.4 | 338.8 | 0.2 | n/a | 21.21 |
| eagle3-cache-off | initial-empty | 55.7 | 350.6 | 0.0 | 2.80 | 22.29 |
| eagle3-cache-on | initial-empty | 55.8 | 353.0 | 0.1 | 2.84 | 22.30 |
| eagle3-cache-on | replay | 55.8 | 353.1 | 0.1 | 2.85 | 22.30 |
| dflash-cache-off | initial-empty | 60.4 | 342.3 | 0.0 | 2.98 | 23.81 |
| dflash-cache-on | initial-empty | 60.3 | 344.3 | 0.2 | 3.00 | 23.85 |
| dflash-cache-on | replay | 60.1 | 344.4 | 0.2 | 2.98 | 23.85 |
| dspark-cache-off | initial-empty | 61.9 | 341.8 | 0.0 | 3.25 | 23.91 |
| dspark-cache-on | initial-empty | 61.6 | 343.8 | 0.2 | 3.25 | 23.92 |
| dspark-cache-on | replay | 61.6 | 344.4 | 0.2 | 3.25 | 23.92 |

## capacity-pressure, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 67.1 | 2441.4 | 0.0 | n/a | 21.19 |
| plain-cache-on | initial-empty | 70.5 | 2379.0 | 8.2 | n/a | 21.19 |
| plain-cache-on | replay | 67.8 | 2457.7 | 2.9 | n/a | 21.19 |
| eagle3-cache-off | initial-empty | 72.6 | 2493.7 | 0.0 | 2.84 | 22.29 |
| eagle3-cache-on | initial-empty | 77.3 | 2327.0 | 8.2 | 2.87 | 22.26 |
| eagle3-cache-on | replay | 83.6 | 2257.7 | 16.3 | 2.84 | 22.26 |
| dflash-cache-off | initial-empty | 76.7 | 2390.0 | 0.0 | 3.10 | 23.81 |
| dflash-cache-on | initial-empty | 77.8 | 2391.4 | 2.9 | 3.11 | 23.85 |
| dflash-cache-on | replay | 77.3 | 2396.7 | 2.9 | 3.06 | 23.85 |
| dspark-cache-off | initial-empty | 77.5 | 2331.3 | 0.0 | 3.25 | 23.92 |
| dspark-cache-on | initial-empty | 82.4 | 2121.4 | 8.2 | 3.33 | 23.92 |
| dspark-cache-on | replay | 85.0 | 2245.7 | 10.9 | 3.31 | 23.92 |

## unique-decode, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 83.4 | 33.6 | 0.0 | n/a | 21.02 |
| plain-cache-on | initial-empty | 83.4 | 34.8 | 1.8 | n/a | 21.02 |
| plain-cache-on | replay | 83.9 | 16.3 | 99.7 | n/a | 21.02 |
| eagle3-cache-off | initial-empty | 130.1 | 35.5 | 0.0 | 2.53 | 21.98 |
| eagle3-cache-on | initial-empty | 128.8 | 36.5 | 1.6 | 2.51 | 21.97 |
| eagle3-cache-on | replay | 131.2 | 18.7 | 99.5 | 2.53 | 21.97 |
| dflash-cache-off | initial-empty | 163.4 | 34.3 | 0.0 | 2.81 | 23.53 |
| dflash-cache-on | initial-empty | 161.9 | 35.4 | 1.8 | 2.80 | 23.53 |
| dflash-cache-on | replay | 164.0 | 16.8 | 99.7 | 2.80 | 23.53 |
| dspark-cache-off | initial-empty | 189.9 | 34.3 | 0.0 | 3.38 | 23.60 |
| dspark-cache-on | initial-empty | 188.1 | 35.3 | 1.8 | 3.35 | 23.60 |
| dspark-cache-on | replay | 190.7 | 16.8 | 99.7 | 3.35 | 23.60 |

## unique-decode, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 394.8 | 178.8 | 0.0 | n/a | 21.20 |
| plain-cache-on | initial-empty | 394.2 | 188.0 | 1.8 | n/a | 21.23 |
| plain-cache-on | replay | 602.1 | 34.1 | 99.7 | n/a | 21.23 |
| eagle3-cache-off | initial-empty | 640.6 | 96.1 | 0.0 | 2.47 | 22.21 |
| eagle3-cache-on | initial-empty | 636.1 | 84.6 | 1.6 | 2.55 | 22.23 |
| eagle3-cache-on | replay | 868.1 | 44.0 | 99.5 | 2.53 | 22.23 |
| dflash-cache-off | initial-empty | 748.6 | 168.1 | 0.0 | 2.81 | 23.90 |
| dflash-cache-on | initial-empty | 760.6 | 154.6 | 1.8 | 2.85 | 23.84 |
| dflash-cache-on | replay | 1018.5 | 41.5 | 99.7 | 2.88 | 23.84 |
| dspark-cache-off | initial-empty | 905.5 | 131.9 | 0.0 | 3.30 | 23.86 |
| dspark-cache-on | initial-empty | 921.8 | 149.8 | 1.8 | 3.33 | 23.89 |
| dspark-cache-on | replay | 1286.6 | 42.1 | 99.7 | 3.37 | 23.89 |

*GPU peak includes other processes, preallocated pools and sampling limitations. Public cached_tokens is joint coverage; it does not measure independent target/draft readiness or recompute rows.

Default BF16 cache-on/off output differences, where output IDs are available, are retained in output-parity-observations.json. They are observations requiring diagnosis, not a pass/fail accuracy claim.
