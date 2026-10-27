# Qwen3.5-4B v0.5.22 hybrid baseline

完成配置：4/4；测量阶段：48。

固定 tag/commit、模型 revision、seed=42、16K slot 上限；graphs/overlap 关闭，使用 Rust Unified Radix Cache。输入为固定 token IDs 的自然文本合成负载，不能视为真实业务的 acceptance 或准确率评估。target 为 24 GDN + 8 FA，DFlash draft 为 5 SWA + 1 FA、窗口 4096。固定 Mamba pool 64 状态、FP32、extra_buffer、track interval 256。这是原生 hybrid 组合的基线，不是与 Qwen3-8B 的 attention 单变量比较。SWA 层的存在不代表 Unified Radix Cache 启用了独立 SWA 组件，以实际初始化日志为准。

正式 campaign 约 10.58 分钟（含每个服务启动及退出），请求计时合计 8.54 分钟。模型下载、环境安装、首次编译、pilot 和单独诊断均不计入正式性能数据。

## 低共享长输出对照

以下取 cache-off，避免把 replay 的大量命中混入 decode 对照。所有请求的输入长度 384、输出长度 256。

| 算法 | 并发 1 输出 tok/s | 并发 8 输出 tok/s | 原生 verify 宽度 | 观测 GPU 峰值 GiB* |
|---|---:|---:|---:|---:|
| plain | 75.8 | 383.6 | n/a | 19.26 |
| dflash | 152.2 | 711.9 | 16 | 28.03 |

## 缓存与容量

| 算法 | 长共享前缀并发 1 TTFT：cache-off → replay (ms) | 并发 1 压力 replay 命中率 |
|---|---:|---:|
| plain | 200.4 → 28.8 | 0.000% |
| dflash | 244.3 → 31.6 | 0.000% |

压力负载的前缀工作集约 24K，大于 16K slot；当前单点有限轮次显示低命中和资源周转，不构成完整容量曲线或稳态实验。高并发结果同时包含准入、batching 和缓存复用的影响。固定 slot 上限不等于固定总显存预算；native allocation 日志保留在 allocation-summary.json。当前仍无法独立观测 target/draft 的 readiness、重算行数及独立 draft slot 回收，不能声称已验证独立生命周期策略。

## 资源生命周期观察

原版 v0.5.22 这条路径的 Unified Radix Cache 实际只启用 FULL + MAMBA；DFlash 的 5 个 SWA 层使用窗口掩码，但 6 层 draft KV 仍整体分配 16384 行（约 0.375 GiB），未启用独立 SWA pool/component。不能据此声称已验证 SWA slot 回收或三组件同时管理。
target FA KV pool 约 0.5 GiB；64-slot FP32 Mamba pool 的 conv + SSM 约 3.12 GiB。DFlash 另有约 6.75 GiB intermediate SSM state cache、0.06 GiB intermediate conv-window cache，以及约 1.36 GiB draft 权重。verify 临时 state 与持久 prefix state 必须区分。server_info.memory_usage.kvcache 仅报告部分池，不能当作全部内存。
GDN decode track interval 固定为 256，prefill 的缓存对齐另由内核 chunk/grid 控制，不应把所有 checkpoint 位置都视为 256 的整数倍。长输入只有 128 tokens 超过 SWA 窗口，这不是长窗口容量曲线。
resource-audit.json 保留 KV、Mamba、SWA 的 available/evictable/used 观测范围；其中 Mamba 的单位是 state slots。采样不是分配器的原子事务，不代替运行时严格空闲检查。
本轮无独立 target/draft 重算行数或 hidden-state 存储开销计数，无法直接量化未来 target-only 保留后重建 draft 的收益。

观测到 Mamba pool 空闲 state slots 降到 0 的配置：plain-cache-on, dflash-cache-on。KV 与 state 是两个容量约束。采样三元组均未超过各自 16384-token / 64-state 容量。

## 输出一致性与结果边界

默认 BF16 的 cache-on/off 共 656 对比较中，115 对完整输出 IDs 不同。请求完成、长度和 slot 检查通过，不代表严格输出对齐通过。差异保存在 output-parity-observations.json，尚未逐项归因。

首次分叉位于首 token 的比较数：0。该观察本身不能确定差异原因。

本轮未完成独立 deterministic 准确率诊断；输出不同不能仅凭 BF16 解释为安全，也不能仅凭完整输出差异断定 cache 损坏。

## 完整分项数据

## shared-long, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 52.7 | 200.4 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 70.7 | 29.1 | 84.8 | n/a | 18.83 |
| plain-cache-on | replay | 74.7 | 28.8 | 97.0 | n/a | 18.83 |
| dflash-cache-off | initial-empty | 88.3 | 244.3 | 0.0 | 5.89 | 27.50 |
| dflash-cache-on | initial-empty | 186.1 | 30.9 | 84.8 | 5.89 | 27.50 |
| dflash-cache-on | replay | 215.9 | 31.6 | 97.0 | 5.89 | 27.50 |

## shared-long, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 93.8 | 1520.8 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 270.3 | 219.2 | 84.3 | n/a | 19.26 |
| plain-cache-on | replay | 507.9 | 94.0 | 97.0 | n/a | 19.26 |
| dflash-cache-off | initial-empty | 99.9 | 1761.5 | 0.0 | 5.89 | 27.60 |
| dflash-cache-on | initial-empty | 462.3 | 223.8 | 84.1 | 5.69 | 27.84 |
| dflash-cache-on | replay | 906.3 | 81.7 | 97.0 | 5.89 | 27.84 |

## multiturn-branch, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 65.5 | 75.3 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 72.8 | 29.9 | 67.1 | n/a | 18.65 |
| plain-cache-on | replay | 74.3 | 27.2 | 77.9 | n/a | 18.65 |
| dflash-cache-off | initial-empty | 129.8 | 92.9 | 0.0 | 4.32 | 27.42 |
| dflash-cache-on | initial-empty | 162.5 | 31.5 | 67.1 | 4.36 | 27.32 |
| dflash-cache-on | replay | 162.5 | 29.1 | 77.9 | 4.30 | 27.33 |

## multiturn-branch, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 172.8 | 387.8 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 324.9 | 137.5 | 67.1 | n/a | 19.04 |
| plain-cache-on | replay | 361.3 | 57.7 | 82.0 | n/a | 19.04 |
| dflash-cache-off | initial-empty | 261.8 | 398.8 | 0.0 | 4.36 | 27.70 |
| dflash-cache-on | initial-empty | 498.8 | 141.6 | 67.1 | 4.30 | 28.03 |
| dflash-cache-on | replay | 623.4 | 56.3 | 82.0 | 4.36 | 28.03 |

## capacity-pressure, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 52.6 | 200.9 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 48.9 | 244.7 | 0.0 | n/a | 18.71 |
| plain-cache-on | replay | 49.0 | 243.7 | 0.0 | n/a | 18.71 |
| dflash-cache-off | initial-empty | 80.2 | 244.0 | 0.0 | 4.47 | 27.51 |
| dflash-cache-on | initial-empty | 80.5 | 249.2 | 0.0 | 4.47 | 27.53 |
| dflash-cache-on | replay | 79.0 | 250.0 | 0.0 | 4.47 | 27.53 |

## capacity-pressure, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 98.4 | 1540.2 | 0.0 | n/a | 18.79 |
| plain-cache-on | initial-empty | 88.5 | 1788.0 | 2.4 | n/a | 18.70 |
| plain-cache-on | replay | 98.8 | 1519.7 | 19.9 | n/a | 18.70 |
| dflash-cache-off | initial-empty | 109.8 | 1758.1 | 0.0 | 4.47 | 27.54 |
| dflash-cache-on | initial-empty | 114.7 | 1672.1 | 7.9 | 4.47 | 27.54 |
| dflash-cache-on | replay | 125.5 | 1487.3 | 15.7 | 4.47 | 27.54 |

## unique-decode, concurrency 1

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 75.8 | 27.5 | 0.0 | n/a | 18.55 |
| plain-cache-on | initial-empty | 76.1 | 32.9 | 0.0 | n/a | 18.44 |
| plain-cache-on | replay | 76.4 | 32.9 | 0.0 | n/a | 18.40 |
| dflash-cache-off | initial-empty | 152.2 | 31.6 | 0.0 | 3.25 | 27.32 |
| dflash-cache-on | initial-empty | 155.8 | 34.0 | 0.0 | 3.25 | 27.32 |
| dflash-cache-on | replay | 156.0 | 33.8 | 0.0 | 3.25 | 27.32 |

## unique-decode, concurrency 8

| Variant | Phase | Output tok/s | Median TTFT ms | Joint hit % | Tokens/verify | GPU peak GiB* |
|---|---|---:|---:|---:|---:|---:|
| plain-cache-off | initial-empty | 383.6 | 117.3 | 0.0 | n/a | 18.80 |
| plain-cache-on | initial-empty | 381.0 | 148.2 | 0.0 | n/a | 18.67 |
| plain-cache-on | replay | 386.1 | 143.6 | 4.2 | n/a | 18.68 |
| dflash-cache-off | initial-empty | 711.9 | 130.6 | 0.0 | 3.37 | 27.92 |
| dflash-cache-on | initial-empty | 717.6 | 148.7 | 0.0 | 3.32 | 27.92 |
| dflash-cache-on | replay | 736.6 | 148.4 | 2.1 | 3.32 | 27.92 |

*GPU peak includes other processes, preallocated pools and sampling limitations. Public cached_tokens is joint coverage; it does not measure independent target/draft readiness or recompute rows.

Default BF16 cache-on/off output differences, where output IDs are available, are retained in output-parity-observations.json. They are observations requiring diagnosis, not a pass/fail accuracy claim.
