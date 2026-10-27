# Qwen3.8-27B checkpoint 独立探针结果

## 范围与验证

源码 `baa9befa381511cd4d56f83980951e5a45fffc6d`，沿用正式实验的模型 revision、16K KV、FP8 target/draft KV、BF16 SSM、extra_buffer、max-running=2、chunked-prefill=512，graph/overlap 关闭。仅改变 state slots=8/12；没有修改 SGLang runtime，没有启用 CPU HiCache。

两档各 82 个测量请求，共 164 个，错误和 host load-back bytes 均为 0。八类独立控制各做两次；两次 cached_tokens 序列完全一致。会话数量扫描每档做一次、每种数量三轮，不把它冒充两次独立重复。每档 32 次同输入输出比较无分叉，属于有限 probe。

目录分别为 `runs/qwen38-k16-s8-r2-checkpoint-isolated-v1`、`runs/qwen38-k16-s12-r2-checkpoint-isolated-v1`。runner/fixtures 与 manifest 的 SHA256、checkpoint-plan 与 COMPLETE 的 SHA256 已核对。两档使用相同 plan hash：`31dda96415388fa1f628e34ee21b5ae0761f82bf6b0f24e4dfd96ffd6b27c0e5`。

每个 case 从独立 flush 开始；各档 21 次 flush 成功。请求及阶段顺序见 `checkpoint-plan.json`，所有输入 token IDs 已冻结；原始结果、metrics、资源采样和日志按步骤保存。机器可读汇总见 `checkpoint-summary.json`。

## 实测命中

表中数字都是每个请求的 cached_tokens，已经去掉首次冷播种请求；不是百分比。A/B 为独立 3200-token 输入，S0/S1 为同一个 2048-token 主前缀的两个 128-token suffix 分支。

| 独立 case | 8-state | 12-state |
|---|---|---|
| 单会话相同输入 S0→S0 | 2048 | 2048 |
| 单会话分支 S0→S1→S0→S1 | 2048、2048、2048 | 2048、2048、2048 |
| 单个共享输入 A→A | 3072 | 3072 |
| 同一个 3072-token 主前缀换分支再复访 | 3072、3072 | 3072、3072 |
| A→B→A→B：后两个请求 | 0、0 | 3072、3072 |
| A→B→B→A：后两个请求 | 3072、0 | 3072、3072 |
| 独立 flush→P1536→P1536 | 1024 | 1024 |
| 独立 flush→P1536→(P1536+64) | 1536 | 1536 |

会话数量扫描（每请求输入 2176，输出 64；三个 cycles，后两轮是复访）：

| 会话数量 | 8-state 后两轮 | 12-state 后两轮 |
|---:|---|---|
| 2 | 每个请求 0 | 每个请求 2048 |
| 4 | 每个请求 0 | 每个请求 0 |
| 6 | 每个请求 0 | 每个请求 0 |

## 对问题的解释

1. **8-state 本身能正确复用单路径。** 单输入和单会话分支控制均命中；此前两组共享输入全 miss 不能解释成模型或配置完全不支持 prefix cache。
2. **访问其他路径后才发生稳定的复用丢失。** 8-state 的 ABAB 全 miss、ABBA 中刚访问的 B 命中而 A miss，12-state 则两种顺序都命中。这是容量/保留策略引发顺序相关失效的实测证据。相同运行条件下 KV 仍有大量空闲；增加 KV 不会增加可用 checkpoint 槽。
3. **checkpoint 边界确实影响复用长度，但不是单会话全 miss。** P1536 原输入最多匹配 1535，不能使用深度 1536 的快照，因此回退到较早快照，实测命中 1024；加长后实测命中 1536。该两路从独立播种状态开始，排除了中间复访改变 cache 的混淆。原四负载里 inactive 的 0% 不能仅以请求末尾边界解释，因为这里已经证明无轮转压力时能复用较早 checkpoint。
4. **12-state 对两个会话足够，对四/六会话的本 trace 不足以维持联合复用。** 从单会话、两个会话到四个会话的失效边界与之前六会话 0% 一致，进一步支持多路径 checkpoint 压力的解释。

源码给出压力来源：scheduler 在 chunk 结束时调用 `stash_chunked_request`→`checkpoint_kv_cache`→`tree_cache.checkpoint`；Mamba component 会把 tracking slot 捐给树，并分配新的 tracking slot继续计算。Rust tree 每个新的 Mamba checkpoint 都进入 Mamba LRU。本实验 `mamba_max_states_per_path=-1`，未限制每条路径的 checkpoint 数。

所以一条 3200-token 输入会经过 512/1024/1536/2048/2560/3072/3200 等 prefill 边界，一条 2176-token 输入也会经过多个边界。**8-state 不是“保留两个最终状态后还剩六个空槽”。** 当前 no-overlap 下，每个新 active 请求最初需要 live+tracking 两个 slots，历史 checkpoint 与它们争用总池；途中还可能有 pinned checkpoint。保存中间边界是为了将来的分支恢复，也可能淘汰另一条路径的可用恢复点。

这一机制与实测失效模式吻合，但本轮没有记录每个节点的 checkpoint 深度和驱逐事件，因此不能把上述边界列表当成“所有状态同时驻留”的实测图，也不能声称已逐个还原所有 LRU victim。最终结果支持 checkpoint 保留策略是后续优化方向，尚未验证任何新策略的收益。

源码入口：`python/sglang/srt/managers/scheduler.py:3644`、`python/sglang/srt/mem_cache/common.py:253`、`python/sglang/srt/mem_cache/unified_cache/components/mamba.py:606`、`rust/sglang-radix-tree/src/components/mamba.rs:150`。

## 资源与比较限制

两次启动前分别复查共享 GPU：338MiB、util 0%、未观测到其他计算任务；主机 available 约45GiB。8-state/12-state 采样整卡峰值分别约29.89/30.18GiB，主机 available 最低均约40.18GiB。结束后再次确认 GPU 338MiB、util 0%、无计算进程。

整卡峰值比旧实验低，主要基线占用不同：本次约338MiB，旧实验约1.7GiB；不能当作模型或状态策略优化收益。Prometheus gauges 仍可能滞后，不能当作 checkpoint 原子事件日志。以上是该批无HiCache探针的历史结论。后续CPU回载和新的3136边界输出分叉另见 [CPU_OFFLOAD.md](../CPU_OFFLOAD.md)，不混入此处164请求；归档只提交摘要，原始产物留在work_dirs。
