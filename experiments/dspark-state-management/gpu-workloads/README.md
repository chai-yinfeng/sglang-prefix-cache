# GPU-only DSpark 四类负载 campaign

源码固定：`baa9befa381511cd4d56f83980951e5a45fffc6d`。复用已安装隔离环境，未改 runtime。模型 revision、完整启动命令、依赖、输入 hash、runner 源码、原始 metrics 和请求结果均按 run 冻结。

模型：Qwen3-8B BF16 + deepseek-ai 官方 DSpark；RadixArk/Qwen3.8-27B-NVFP4 + RadixArk/Qwen3.8-27B-DSpark。后者 target revision=`319f741cce68d7914884900c138a1fbb70a42f30`，draft revision=`b9a5dbdf03bc999c6c73c426b19c2d9041cea393`。

共同设置：page_size=64；graph/overlap 关闭；原生 radix cache，未开启 CPU HiCache；temperature=0、ignore_eos=true。使用此前冻结的技术文档语料和各模型自己的 tokenizer，固定自然文本 token trace；不是模型准确率或真实业务 acceptance 评估。

## 四类负载

1. 共享前缀/分叉：两组 3072-token 前缀，128-token suffix，64-token 输出。先各一次冷/热请求，再做 8 个并发分叉；不同文档填入超过 1.5 倍池容量后重访。冷/热/压力后探针输入完全相同。
2. 会话轮转：六组 2048-token 前缀、128-token suffix，64-token 输出，串行轮转四轮；cycle 0 为初始状态，cycle 1–3 汇总为稳态段。约 13–15K 的前缀/分支工作集用于跨越 8K 与 16K 容量边界。
3. 长上下文生成：先存四组 inactive 前缀（每请求 1536 输入、16 输出），再运行 2048/4096/6144 输入 + 1024 输出的独立 active 请求，随后重新探测 inactive 前缀。逐 shape 清 cache 后重新播种，记录生成阶段驱逐与探测命中。探测自身也会改变 cache，probe 低命中包含扫描放大，不能解释成生成阶段独自驱逐全部前缀；生成阶段 evicted tokens 是独立证据。
4. 突发与生命周期：12 个不同前缀、1536 输入 + 512 输出、客户端并发 8。记录 server queue、实际运行请求和 retractions；随后两个请求并发，其中一个在首 token 后经 `/abort_request` 取消，另一个完成。每轮保存清理后的空闲指标。

每个配置三次重复，完整配置的 COMPLETE 必须四类全完成。warmup 单独保存，不入结果表。取消是正常控制操作，以 finish_reason=abort 验证，不将其视为请求错误。没有强制制造 retract；如果 admission 提前排队，必须按实际情况报告。

## 预算和解释

Qwen3-8B 的 8K/16K slots 对应两侧 KV 理论 payload 1.28125/2.5625 GiB，不含页 padding 和 workspace。
Qwen3.8 固定 target FP8 KV、BF16 SSM、extra_buffer，分别控制 KV slots 与 Mamba state slots；按启动日志核对 draft 的实际 dtype、persistent state 与 intermediate verify state。首次 smoke 使用 extra_buffer_lazy 被配置校验拒绝：该策略要求 overlap；此次保持 overlap 关闭，使用已有 extra_buffer 路径。
ReplaySSM 不是此处的可用替代：当前版本对 DSPARK/DFLASH 明确要求 KDA target，拒绝 Qwen3.8 的 GDN target。`calibration-replay` 是失败的兼容性探针，不进入正式表。实际正式配置继续使用 snapshot verify。
首次 autotune 导致主机内存压力，后续 27B 统一关闭广泛 autotuning，并限制 MAX_JOBS=2、FLASHINFER_NVCC_THREADS=1、TORCHINDUCTOR_COMPILE_THREADS=1。首次短请求编译可达约两分钟，因此先完成单请求校准和长短/batch warmup，编译不纳入正式阶段。
`max-running-requests`、池容量、有效共享前缀和 admission 共同决定能同时运行多少请求；不能由客户端并发数直接推断。
`mem-fraction-static` 不是进程硬显存上限。200ms NVML 为整卡采样占用，包含外部常驻与预分配；metrics 的 state slots 与 token slots 分开计量。

正式对照预算如下。state slots 是 active、tracking 和 cached state 共用的原生池容量；不是可全部交给 inactive checkpoint 的数量。

| 模型 | KV slots | target/draft KV dtype | Mamba slots | 请求并发上限 |
|---|---:|---|---:|---:|
| Qwen3-8B | 8192 / 16384 | BF16 / BF16 | 不适用 | 8 |
| Qwen3.8-27B | 8192 / 16384 | FP8 / FP8 | 12 | 2 |
| Qwen3.8-27B | 16384 | FP8 / FP8 | 8 | 2 |

27B 的 burst 会受到显式并发上限影响；状态优化能否提高可运行并发，需要另行提高上限并在同字节预算下验证，本轮不据排队直接推断。

## 运行与汇总

```bash
work_dirs/qwen3-8b-prefix-baseline-v0.5.22/.venv/bin/python \
  work_dirs/dspark-state-management/gpu_workloads.py \
  --profiles qwen3-k8 qwen3-k16 --repeats 3 --attempt baseline-v1

python3 work_dirs/dspark-state-management/summarize_gpu.py
```

27B 正式命令：

```bash
work_dirs/qwen3-8b-prefix-baseline-v0.5.22/.venv/bin/python \
  work_dirs/dspark-state-management/gpu_workloads.py \
  --profiles qwen38-k8-s12-r2 qwen38-k16-s12-r2 qwen38-k16-s8-r2 \
  --repeats 3 --attempt baseline-snapshot
```

原始目录不会被覆盖；失败尝试保留 FAILED.json 和日志。Qwen3-k8-baseline-v1 的首次 repeat 与下载尾段重叠，只用于校准；另行复测用于正式容量对照。其余运行必须在下载结束后启动。
RESULTS.md 为可重复生成的观测表；因果分析与范围限制另存 ANALYSIS.md。
