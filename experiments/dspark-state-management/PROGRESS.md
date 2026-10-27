# 续接记录（2026-10-27）

## 用户最后确认的范围

只完成当前 benchmark。CPU offload 问题排查和 checkpoint 边界探针均延后；用户的 5h 额度恢复后会主动要求继续，不自动追加实验。

## 已完成

- 仓库 `/home/blackmagic/sglang`，分支 `research/joint-prefix-cache`；实验源码固定 `baa9befa381511cd4d56f83980951e5a45fffc6d`（v0.5.22）。未修改 runtime，实验和文档在 gitignored work_dirs 中，未强制加入 commit、未推送。
- RTX 5090 32GB；实验环境 `work_dirs/qwen3-8b-prefix-baseline-v0.5.22/.venv`。GPU/localhost 操作需沙箱外权限；PATH 需包含该环境的 bin（ninja）。
- Qwen3-8B BF16 + DeepSeek 官方 DSpark：8K/16K KV，各完成四类负载、三次重复。
- RadixArk/Qwen3.8-27B-NVFP4 + RadixArk DSpark：8K KV/12 state、16K KV/12 state、16K KV/8 state，并发上限均为 2；均完成四类负载、三次重复。target/draft KV FP8，SSM BF16，extra_buffer；graph/overlap 关闭。
- 正式合计 1329 请求、0 错误、15 次取消验证；所有配置各 22 次 flush 成功、0 次拒绝。60 对冷/热及热/驱逐后输出比较未分叉，仅为有限 probe。
- 运行 session 56461 已以 exit_code=0 结束；最后模型服务的 server.log 确认清理完成、无 live scheduler。未启动 checkpoint 或 CPU 探针。

## 主要结果

| 对照 | 观测 |
|---|---|
| 8B KV 8K → 16K | 六会话后续命中率 0% → 96.08%；burst 496.2 → 866.2 tok/s；p95 TTFT 6324 → 3605ms |
| 8B 共享前缀热 → 驱逐后 | TTFT 约 21–24 → 251–252ms |
| 27B 固定 12-state，KV 8K → 16K | 六会话均 0% 命中；burst 77.5 → 78.4 tok/s，没有明显收益；KV 驱逐减少 |
| 27B 固定 16K KV，state 12 → 8 | 共享前缀热命中 96% → 0%；热 TTFT 88.4 → 602.5ms；persistent conv+SSM 约 0.95 → 0.65 GiB，verify intermediate 仍约 1.72 GiB |

结果符合存在缓存压力和优化空间的预期；不能将容量翻倍的收益直接归为 hidden-only 策略收益。8B projected hidden 替代 draft KV 的理论两侧 payload 节省约 7.3%；27B BF16 projected hidden 与当前 FP8 draft KV 都是 10KiB/token，直接替代不省 payload。

## 结果入口与复现

- `gpu-workloads/RESULTS.md`、`summary.json`、`summary.csv`：正式观测和完整分配日志索引。
- `gpu-workloads/ANALYSIS.md`：解释、可优化空间、未归因项和限制。
- `gpu-workloads/README.md`：四种 trace、控制条件、预算和命令。
- 正式 runs：`qwen3-k8-baseline-controlled`、`qwen3-k16-baseline-v1`、`qwen38-k8-s12-r2-baseline-snapshot`、`qwen38-k16-s12-r2-baseline-snapshot`、`qwen38-k16-s8-r2-baseline-snapshot`。
- 每个 run 有 COMPLETE、manifest、冻结 runner/fixtures、启动参数、逐请求结果、metrics 与资源采样。首次 8B 8K baseline-v1 与下载尾段重叠，已排除，使用 controlled 复测。
- 汇总：`python3 work_dirs/dspark-state-management/summarize_gpu.py`；机制 pilot 汇总：`python3 work_dirs/dspark-state-management/summarize.py`。
- 27B target revision `319f741cce68d7914884900c138a1fbb70a42f30`；draft revision `b9a5dbdf03bc999c6c73c426b19c2d9041cea393`。路径与下载清单见 `gpu-workloads/qwen38-models.json`，权重已完整缓存。

## 延后事项：不要误记为已验证

1. checkpoint 边界探针未运行。`gpu_workloads.py --checkpoint-probe` 已准备：单会话交替分支，以及 inactive 原输入/加长输入重访；不进入四负载表。可用 `--profiles qwen38-k16-s12-r2 --checkpoint-probe --attempt checkpoint-boundary-v1`。先确认用户要求继续再启动。六会话与 inactive 的 0% 不能全部归因为容量驱逐；需区分 checkpoint 位置、可匹配边界和实际驱逐。
2. CPU offload 尚无本机通过案例。先前三种 kernel/page_first 或 direct/page_first_direct 均在 warmup、压力/回载测量之前 CUDA illegal access；不能说已经证明 load-back 代码本身失败。根因未定位，未修复。
3. `pilot.py` 已增加 `--mem-layout layer_first` 与 `--disable-tma`，尚未运行新配置。优先试已有 `direct + layer_first`，然后隔离默认路径。target+draft 回载应核对 167936 bytes/KV-token，不能只凭请求成功宣称双侧恢复。
4. `offload_transfer_probe.py` 只准备并做了 py_compile，未执行 GPU：复用仓库原生 HiCache 往返测试，允许设置 page/layers/dim 和 pinned/registered allocator。模型几何（page64、36层、row1024）可覆盖大页 batch-copy；小测试几何可能只走小拷贝 fallback。每个 CUDA 错误案例用独立进程。
5. 源码线索：AOT test_kvcacheio.py 对 CUDA13 有已知 segfault skip；kernel/page_first staged backup 与 direct/page_first_direct 都可能走 batch-copy，未证明它就是本机根因。不能移除 skip 当作修复。TMA、host-register 指针域、wheel/source 一致性等仍待排查。
6. 当前 checkout 有 mamba-max-states-per-path、int8 checkpoint pool；本轮未开。int8 checkpoint 与 HiCache 配置互斥，且须核算 active+checkpoint 两池总字节数。ReplaySSM 对 DSPARK/DFLASH 要求 KDA，Qwen3.8 GDN 已被校验拒绝，不再反复尝试。
7. 不改 runtime 前先讨论根因与方案；目前只允许保留实验/文档。既有 target/draft sidecar 跟随 KV lifecycle；没找到持久 hidden-only context 恢复路径。MAMBA/FULL/SWA component 有独立管理，不能泛称整个 infra 完全不分级。

## 主机掉线问题

用户报告 SSH 掉线。内存耗尽/严重 swap 可令 SSH 不响应或进程被杀，但没有系统日志证据，未确认此次掉线原因。早期广泛编译校准确实曾到 RAM 约 46GiB、swap 约 15GiB，已终止；之后限制编译线程并关闭广泛 autotuning。正式 27B 阶段 host available 采样约 40GiB，不能由此排除短时异常，也不能声称主机一定不会受影响。后续恢复工作先核对资源，必要时再读系统 OOM/驱动日志；此次按要求不追加诊断。
