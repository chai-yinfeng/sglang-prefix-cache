# DSpark GPU 四类负载结果

基线：v0.5.22 / baa9befa381511cd4d56f83980951e5a45fffc6d；GPU-only、graph/overlap 关闭，page_size=64，固定 synthetic natural-text token traces。每个完整配置三次重复；先完成短/长 shape warmup。

两个模型的 checkpoint、KV/SSM dtype、实际池和 workspace 不同；只在同模型固定 trace 的配置间比较，不能将跨模型差异归因于某种 attention。

## 容量压力与性能

TTFT 列为三次 repeat 的阶段中位数的中位数。session 汇总 cycle 1–3；burst throughput 为三次 repeat 中位数，p95 只描述本次小样本。

| 配置 | 请求数 | 共享前缀 TTFT：热 → 驱逐后 ms | 会话稳态命中率 | 会话输出 tok/s | 6K 长生成后 inactive 命中率 | Burst tok/s | Burst p95 TTFT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-k16-baseline-v1 | 273 | 21.4 → 251.3 | 96.1% | 218.4 | 95.8% | 866.2 | 3605.3 |
| qwen3-k8-baseline-controlled | 255 | 24.1 → 252.0 | 0.0% | 143.5 | 0.0% | 496.2 | 6324.1 |
| qwen38-k16-s12-r2-baseline-snapshot | 273 | 88.4 → 592.9 | 0.0% | 40.4 | 0.0% | 78.4 | 40921.1 |
| qwen38-k16-s8-r2-baseline-snapshot | 273 | 602.5 → 606.4 | 0.0% | 40.3 | 0.0% | 76.4 | 41218.9 |
| qwen38-k8-s12-r2-baseline-snapshot | 255 | 88.5 → 614.3 | 0.0% | 39.7 | 0.0% | 77.5 | 41656.3 |

## 资源与生命周期

GPU 峰值为 200ms NVML 采样的整卡占用，含外部常驻占用和预分配/缓存，既非严格进程峰值，也非全部活跃状态量。Mamba 指标的单位是 state slots，不能与 token slots 相加。完整分配日志和 flush 后快照见 summary.json。Prometheus 容量 gauge 可能滞后于 flush；以日志核对 flush 成功，不把快照当作原子池审计。

| 配置 | 采样整卡峰值 GiB | Burst 最大采样队列 | Burst retract 次数 | 已确认取消请求 | 冷/热输出分叉对数 |
|---|---:|---:|---:|---:|---:|
| qwen3-k16-baseline-v1 | 24.26 | 5 | 0 | 3 | 0/6 |
| qwen3-k8-baseline-controlled | 22.92 | 5 | 0 | 3 | 0/6 |
| qwen38-k16-s12-r2-baseline-snapshot | 31.54 | 7 | 0 | 3 | 0/6 |
| qwen38-k16-s8-r2-baseline-snapshot | 31.09 | 7 | 0 | 3 | 0/6 |
| qwen38-k8-s12-r2-baseline-snapshot | 31.37 | 7 | 0 | 3 | 0/6 |

所有完整配置的请求错误和 host load-back bytes 均为 0；取消请求单独以 finish_reason=abort 判定。输出分叉保留为观察值，未归因为浮点或缓存缺陷。严格空闲检查开启，但监控采样不是 allocator 原子审计。

## 失败与未完成配置

- `qwen38-replay-smoke-calibration-replay`: failed，RuntimeError('scheduler exception during startup')；原始日志和启动参数保留。
- `qwen38-smoke-calibration-budget-v2`: failed，RuntimeError('server exited -15')；原始日志和启动参数保留。
- `qwen38-smoke-calibration-extra-buffer`: failed，RuntimeError('server exited -9')；原始日志和启动参数保留。
- `qwen38-smoke-calibration-limited-jit`: failed，RuntimeError('request errors in warmup')；原始日志和启动参数保留。
- `qwen38-smoke-calibration-v1`: failed，RuntimeError('server exited -9')；原始日志和启动参数保留。

CPU offload 不进入本轮测量；它的错误与修复另行调查。未实现新的状态管理策略，因此容量对照只能证明压力和成本，不能证明 hidden-only 或 offload 的实际收益。
