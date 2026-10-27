# 队友 benchmark 结果复核

基于 `lvelf/benchmark` / `1af0d7917bf0e88a3a3a51c7c44e6bd4f01fab44` 的归档汇总与相关源码复核，不是独立重跑。队友新增压力测试与本机固定容量、关闭graph/overlap的机制实验不同，不能直接比较绝对吞吐。原始harness、输入和日志不在该分支，证据范围限于已归档汇总。

## 最有价值的新增证据

### 8B：同 target KV 容量对照

W2：24 个独立 4096-token 文档前缀、Zipf(α=1)，工作集约98K。graph/overlap 开启，自动分配容量；与我们 graph/overlap 关闭、固定8K/16K槽和确定性轮转的机制实验不同。

第一轮并发64：

| 配置 | target KV slots | 输出 tok/s |
|---|---:|---:|
| plain | 58687 | 984.24 |
| plain-kv38093 | 38093 | 527.06 |
| DSpark | 38093 | 601.17 |

计算：缩小 plain 容量损失约46.45%；同容量 DSpark 比 plain 高14.06%，但相对满容量 plain 低38.92%。这强烈支持该 trace 下静态容量挤占是主要损失来源，且比仅做我们原来的人工紧预算测试更接近部署问题。

e2-rep2 与第一轮有36个共同点，直接按 JSON 重算吞吐比值：median=1.000231，range=0.993281–1.060024。DSpark W2/c64 复测637.25（第一轮601.17），plain复测985.07，方向不变。第一轮部分 server log 被覆盖，summary 中其 log_hit_rate 为 null；重复轮有命中指标，DSpark≈0.631、plain≈0.791。不能把旧日志已不在的字段写成此次重新直接核验的原始数据。

该对照控制 target token 容量，不是等总显存或等全部 kernel/graph 配置；因此支持“容量是主因”，不宜扩大为“所有差异已完全归因”。

### 27B：服务能力和状态成本

E2 W2/c16 的归档 JSON：

| 配置 | 输出 tok/s | token 命中比例 | 采样最大 running |
|---|---:|---:|---:|
| plain | 470.06 | 0.716 | 14 |
| DSpark-n2-nopg | 172.60 | 0.242 | 2 |
| MTP-n4-nopg | 206.06 | 0.315 | 4 |
| DFlash2-n3-nopg | 123.77 | 0.197 | 2 |

DSpark 低并发W2/c1 为118.90，相对plain67.96约1.75倍；高并发时主要被配置下的服务容量和前缀复用限制。这与我们的 checkpoint 机制实验互补。

DSpark E2 实际分配约：target加载20.14、draft加载3.95、persistent state0.65、verify intermediate1.72、target KV0.26、draft KV0.08、graph0.29（原日志口径，GiB量级）。因此该配置下 draft KV 绝对值很小，verify/persistent state/weights 的管理更值得优先分析。

队友使用 overlap/extra_buffer_lazy、不同 static fraction、graph和chunk配置；不能直接将其173 tok/s与我们约78 tok/s相比来评估回归。B组目前仅单次扫描，缺同资源预算plain对照和更多重复，不能将2–3并发当作所有可行配置的硬件极限。

### 计算成本也需要保留

8B自然对话并发64：DSpark≈0.92× plain、DFlash≈0.50×。报告的 ignore_eos decode负载接受长度异常偏高，后续自然对话复核更可信。即使增加可运行容量，大batch的verify计算也可能不划算；最终应看TTFT/TPOT/吞吐，不能只看命中率和池容量。

## 对候选方向的判断

- 同意：只保留/丢弃 draft KV 不能解释或消除全部额外显存成本。8B更大的项是draft权重，27B还要重点看verify state。本结论针对当前长度/配置，不能推广到任意长上下文。
- 收紧：“现有系统完全没有显存管理机制”过强。当前已有FULL/SWA/MAMBA分组件、HiCache、特定hybrid unified byte pool、checkpoint表示/保留配置等；尚未证明的是本次路径会随负载自动把draft权重/graph/verify空间返还target。
- 收紧：984 vs601、470 vs206是不同配置的性能参照，不是新策略的保证收益或严格数学上界。动态切换、prefill重建、host传输、重新分配和graph切换都有成本。
- MTP约5GiB空间的初始化顺序问题值得独立验证。源码确实先 `draft_worker.alloc_memory_pool` 再 `init_lm_head`/共享embedding/head，但“提前共享后并发4→15”仍为推算，且它是MTP路径，不应直接归为DSpark修复。未在本机改动或复现。
- 原4B归档是 native DFlash（draft 5 SWA+1 FA），不是社区DSpark/Speculators适配完成。报告中115/656输出不一致也未完成归因，不能当作完整准确率验证通过。
- 继续聚焦DSpark状态保留/恢复是合理的。无需因为队友提出负载自适应就立即重构通用infra；先利用现成HiCache，测清恢复成本和实际可用字节，再决定策略层范围。

## 显存限制与“缓存充足时 spec 一定更快”的澄清

8B 的 plain/DSpark 均采用自动解析的 mem-fraction-static=0.776，max-running=64、decode graph最大batch=64，graph/overlap开启。不是用外部张量占显存，也不是强制两者有相同target KV容量。实际静态账目分别为 plain 15.28权重+8.06 KV≈23.34GiB；DSpark 15.28+2.10权重+5.24 target KV+0.72 draft KV≈23.34GiB。graph另约0.92/2.23GiB，运行时激活等还需空间。mem-fraction-static不是进程硬显存限额。

E3额外使用max-total-tokens=38093把plain限制到与DSpark相同的target KV容量，plain因此留下更多未分配显存。这是容量控制，不是等总显存/等compute/batch的控制。W2的约98K前缀工作集大于两档58.7K/38.1K容量，双方均有容量压力，不能描述为“plain全部缓存、只有spec刚好越界”。相同静态fraction也不是分别调优后的最佳配置；尚不能排除提高DSpark fraction或调整graph后差距缩小。

27B分初始探测与实际E2配置：初始cookbook plain .90自动71.8K KV/62 states/15请求，DSpark .88自动15.9K KV/5 states/1请求。E2改用dspark-n2-nopg：fraction=.92、max-total-tokens=8800、max-mamba-cache-size=8、prefill graph关闭，有效请求上限2；plain沿用.90/71.8K/62states/15请求。它比较两种部署配置，不是只改spec开关。每请求4-slot是该配置器的保守并发预算比例，不能理解为任意时刻每个请求都私有持有四份状态。

缓存充足只排除了部分容量损失，不能保证计算加速。固定相同active batch B和上下文，spec每个输出token的近似成本为(T_draft(B,γ)+T_verify(B,γ)+T_control)/E[本轮输出token数]；必须低于plain的T_decode(B)才有收益。大batch已摊薄target权重读取、GPU算力趋于饱和时，verify γ个候选不再近似免费，低接受长度会使额外计算更贵。

归档反例：8B w1-natural/c64，第一轮plain实际max-running64、TPOT21.23ms、2118tok/s；DFlash也到64，KV used34898小于其38542容量、无retract，却TPOT53.17ms、1062tok/s，接受长度约2.23。重复轮吞吐和TPOT接近。该低共享短请求负载不能用“前缀未缓存造成大量重算”解释全部退化。DSpark同点约1955tok/s，但实际max-running60 vs plain64，其较小退化仍可能包含准入/batching差异，不能单独认定为纯计算原因。具体算子构成仍需profiling。

27B还有无大量前缀复用也可变慢的服务容量例子：w1-natural/c16，plain560tok/s、TPOT19.58ms、max-running15；DSpark257tok/s、TPOT6.80ms、max-running2。单请求decode明显更快，但整机吞吐仍更低；state/verify预算限制并发，不需要“前缀刚被驱逐”才会出现。

后续若要严格分离：8B应增加无压力/无跨请求复用、相同实际active batch的compute对照；27B尚缺同KV+同state预算plain对照。关闭prefix cache不等于去掉运行中的attention KV/recurrent state。当前未运行这些额外实验。

## 探针实现澄清

checkpoint-isolated-v1 是外部请求/采样脚本：原生HTTP flush_cache、generate、server_info、metrics，加NVML/psutil。没有runtime hooks、逐节点创建/驱逐插桩、Nsight或profiling插件。控制序列能证实访问顺序和state预算影响命中；精确checkpoint victim/depth序列仍未直接采集。不能把源码推导的512-token边界列表说成实际驻留图。

## CPU offload 后续状态

本机原生8B direct/LF已完成驱逐和回载；默认registered+batch故障已在框架外复现。27B经 `00fd51eab3` 扩展Mamba LF支持后传输机制和逐字节检查通过，但端到端输出检查未通过；关闭HiCache的GPU-only路径也有checkpoint复用分叉。详见 [CPU_OFFLOAD.md](CPU_OFFLOAD.md)。这不实现draft权重/graph卸载、静态GPU池扩张或独立draft保留策略。
