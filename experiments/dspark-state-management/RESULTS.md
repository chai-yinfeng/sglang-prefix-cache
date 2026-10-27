# 调查与 pilot 结果

本页8B历史机制pilot与原四负载固定为 v0.5.22 / baa9befa381511cd4d56f83980951e5a45fffc6d，当时未改runtime。27B新增LF补丁和输出检查失败详情见 CPU_OFFLOAD.md，不混入本页历史统计。

## 已确认能力

- GPU 前缀命中直接复用；只重算未命中的区间，并非每次都完整 prefill。
- CPU HiCache 源码支持 target KV + Qwen DSpark draft KV sidecar 的 backup/load-back；SSM component 另有 checkpoint host 回载。
- 未找到 context-hidden-only 的持久 pool、radix readiness 和恢复调度；projected hidden -> context KV 的计算原语已经存在。
- 原生 Qwen draft sidecar 跟随 target KV 转移，不是独立 draft-only CPU 生命周期。

## GPU 容量边界实测

以下是机制 pilot，非稳态吞吐/尾延迟 benchmark。图和 overlap 均关闭，page_size=64，BF16 target/draft KV。

| GPU slots | 两侧 KV 理论 payload GiB | 压力输入 tokens | 压力阶段 evicted tokens | anchor 热命中 → 压力后命中 | anchor 冷/热/重算 IDs |
|---:|---:|---:|---:|---|---|
| 4096 | 0.640625 | 6144 | 6144 | 1984 → 0 | 一致 |
| 8192 | 1.281250 | 12288 | 8192 | 1984 → 0 | 一致 |
| 16384 | 2.562500 | 24576 | 12288 | 1984 → 0 | 一致 |

已完成 48 个测量请求；warmup 单独保存。相同输出仅覆盖这些 anchor，不替代完整正确性验证。

payload 为每个有效 slot 的理论字节数乘以配置容量；实际池含 padding，原始分配日志和 server_info 已保留。未测量进程严格峰值显存，不把 mem-fraction-static 当作硬上限。

## CPU offload 运行状态

- `qwen3-8b-4096-cpu-small-direct`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-direct-blocking`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-kernel-page-first-memcheck-v1`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-kernel-page-first-memcheck-v2`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-kernel-page-first-sync-diag-v1`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-kernel-page-first-v2`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-cpu-path-fixed`：failed；CUDA illegal memory access before pressure/load-back measurement。日志保留。
- `qwen3-8b-8192-gpu`：failed；missing ninja in launch PATH。日志保留。
- `qwen3-8b-8192-cpu-direct-layer-first-async-v1`：passed；回载 1984 KV tokens / 333185024 bytes，即 167936 bytes/token，核对两侧合计 167936 bytes/token。
- `qwen3-8b-8192-cpu-direct-layer-first-v1`：passed；回载 1984 KV tokens / 333185024 bytes，即 167936 bytes/token，核对两侧合计 167936 bytes/token。

当前通过配置为原生 direct/layer_first（blocking及正常异步）；默认 kernel/page_first 的registered host memory + batch-copy故障已独立复现，batch路径未改。27B Mamba LF支持已在本地00fd51eab3扩展，联合回载的传输与字节检查通过、端到端输出检查失败；GPU-only对照也有checkpoint复用分叉。详情见 [CPU_OFFLOAD.md](CPU_OFFLOAD.md)，不能以8B结论替代27B正确性。

## 四类 GPU 负载

独立 campaign 见 [GPU 观测表](gpu-workloads/RESULTS.md) 和 [分析](gpu-workloads/ANALYSIS.md)。RadixArk Qwen3.8-27B target/draft 已下载并固定 revision；每个 run 以 COMPLETE/FAILED 文件判断状态。此处的机制 pilot 不混入四类负载统计。

已汇总完整配置 5 个；具体容量、重复次数和请求数见 GPU 观测表。

## 未完成范围

- Qwen3.5-4B SWA DSpark：not-run，Speculators 适配尚未实施。
- hidden-only 与独立 draft CPU 策略：not implemented；需要先决定数据所有权和恢复边界。

原始实验目录 work_dirs/dspark-state-management 被 .gitignore 排除；本目录归档结果摘要。完整能力分析见 README.md，机器可读机制摘要见 pilot-summary.json；原始audit/logs仍在work_dirs。
