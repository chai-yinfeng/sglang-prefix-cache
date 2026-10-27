# 结果解释与优化空间

## 指标和预算的准确含义

本文“会话命中率”是 `sum(cached_tokens) / sum(input_tokens)`，即跳过 prefill 的 token 比例，不是请求命中率。8B/16K 的 cycle 1 六个请求各命中 2048/2176 tokens，cycle 2–3 各命中 2112/2176 tokens，合计 96.0784%；请求级别则全部有命中。8B/8K 对应逐请求 cached_tokens 均为 0。

8K/16K 设置的是全局 `max-total-tokens=8192/16384`，不是增加单请求输入或 context-length。两档 context-length 均为 8192；六会话固定每请求 2048-token 主前缀+128 suffix、64 输出，按 0…5 顺序串行轮转，suffix 在两个固定分支间交替，生成输出不反馈到下一轮。仅六个主前缀已需 12288 slots。8K 的 LRU 循环扫描可在每次复访前淘汰相应前缀，形成整轮 miss；容量小于工作集不意味着命中率等于容量/工作集。16K 能保留这组约 13–15K 工作集。

8B 的紧预算由显式池上限人为设置，整卡并未用满；这证明受限池容量的成本，并非证明这张 32GB 卡默认部署 8B 必然遭遇相同压力。

27B 配置为 64 层 target：每四层三个 GDN linear/recurrent attention、一个 full attention，共 48 GDN+16 FA。draft 是 5 层 full attention，无 SWA/recurrent；捕获 target 层 5/19/33/47/61 的 hidden，经投影后构造 draft context KV。原生 target 树在同一前缀节点检查 FULL 和 MAMBA 都有效；draft context KV 作为 sidecar 随对应 KV 索引管理。有效命中取可恢复前缀边界的交集，不能对 token slots 和 state slots 直接取数值最小值。

`12-state` 是全模型 recurrent 状态池的 12 个可分配 slots，active state、tracking 和 cached checkpoint 共用，不是 12 个会话。每 slot 含全部 48 层 GDN 的 SSM 与 conv 状态：BF16 SSM 为 48×48×128×128×2=72MiB，conv 约 2.8125MiB，合计约 74.8125MiB。物理分配另有一个 padding slot，12-slot 池约 0.9498GiB。当前 no-overlap 配置最初分配一个 live slot+一个 tracking slot；树上的 pinned checkpoint 会另占资源，不能把采样 mamba_used=3 简化为恒定的每请求三个初始 slots。

有 FA KV 不代表有任意 token 位置的 GDN checkpoint。若请求范围内没有满足联合条件的正深度 checkpoint，cached_tokens 可以为 0，即使物理 KV 仍存在。六会话全miss的逐节点保留/淘汰序列尚未重建，不能仅由“12 slots 已满”认定全部根因。后续独立checkpoint探针已经完成，结果见 CHECKPOINT_RESULTS.md。

## 27B/16K/12-state 的显存账本

以下以历史整卡采样峰值 31.544GiB 为准；单位 GiB，加载项是日志中的可用显存增量，不等同于纯权重 tensor 精确占用。

| 项目 | GiB | 依据 |
|---|---:|---|
| target 加载增量 | 20.14 | Load weight end |
| draft 加载增量 | 3.96 | 独立 draft runner 的 Load weight end |
| persistent SSM+conv | 0.950 | 13 个物理 state slots；日志约 0.91+0.04 |
| verify intermediate SSM+conv | 约 1.72 | SSM=(2 请求+1 padding)×8 verify positions×72MiB=1.6875GiB，另有 conv windows |
| target FA KV | 0.502 | (16384+64 padding)×16层×2(K/V)×4 KV heads×256 dim×1 byte |
| draft context KV | 0.157 | (16384+64 padding)×5层×2×8 KV heads×128 dim×1 byte |
| 上述小计 | 约 27.43 | 两侧 KV 仅约 0.659GiB |
| 加载 target 前卡上已有占用 | 约 1.76 | 总容量31.843−日志 free30.08；包括既有占用和初始化上下文，不能全部归为外部进程 |
| 其他初始化及运行时占用差额 | 约 2.35 | 峰值减上述账目；未采集 allocator 快照，不能逐项归因为激活或 workspace |
| 整卡采样峰值 | 31.54 | 200ms NVML |

加载前 manifest 的整卡占用为 1788MiB；模型初始化完成日志 free≈2.11GiB，正式阶段进一步增长到采样峰值。约 2.35GiB 残差可包含 backend workspace、临时激活、allocator reserved/fragmentation 等，但不能据现有数据精确拆分。mem-fraction-static=0.97 不会自动把余量都分给 KV，显式 max-total-tokens 仍约束池大小。

RadixArk 包是混合精度，不能按全部 27B 参数都是4bit估算：量化配置 group_0 为 FP8（attention 投影等），group_1 为 NVFP4（MLP/lm_head 等），另有 BF16 embedding、vision 等。只读取 safetensors headers 统计，target tensor payload 约 20.4159GiB，其中 BF16 4.0663、FP8 7.7889、U8 packed/scales 等 8.5608GiB；draft 文件 payload 约 3.4596GiB BF16。磁盘 payload 与运行时加载增量口径不同，不能直接相加或用其中一方替代另一方。

## Qwen3-8B：容量边界已经打到

正式对照选择 `qwen3-k8-baseline-controlled` 与 `qwen3-k16-baseline-v1`。前者三次重复均在权重下载结束后运行；原 `qwen3-k8-baseline-v1` 仅留作校准，不混入正式表。

- 相同共享前缀从热命中约 21–24ms 回到驱逐后约 251–252ms。两档都强制施加超过自身容量的 churn，所以两档压力后都会 miss；这不是 16K 比 8K 更差。
- 固定六会话工作集：8K 后续轮次命中率 0%，16K 为 96.08%；相应串行输出吞吐约 143.5 → 218.4 tok/s。这说明重复计算成本可被缓存容量改变，不是仅看 GPU 总占用推测压力。
- 长生成阶段：8K 确实驱逐 inactive 缓存；16K 的同一输入/输出长度能保留该工作集。顺序 probe 会继续改变缓存，0% probe 命中包含扫描放大，不能声称生成阶段一次性驱逐了所有前缀。
- Burst：相同 12 请求、客户端并发 8，8K 实际采样运行峰值约 3，16K 约 7；输出吞吐约 496 → 866 tok/s，p95 TTFT 约 6324 → 3605ms。两档均出现队列，未观测到 retract，说明当前主要先经 admission 限流，而非靠强制 retract 释放。
- 两个正式配置各完成三次取消，survivor 正常完成，空闲时 active usage/running/queue 归零；冷/热及热/驱逐后共 24 对输出比较未分叉。此范围不是完整准确率证明。

## 这些结果能支持什么

可优化成本包括：可复用前缀被挤出后的 target 重算、inactive 缓存与 active 请求的容量竞争，以及容量受限引起的排队。现有源代码没有 hidden-only 持久恢复路径，因此此轮没有测得 hidden-only 的收益。

8K 与 16K 是约两倍 KV 容量对照；Qwen3-8B projected hidden 替代 draft context KV 的理论两侧 payload 节省仅约 7.3%。不能把两倍容量带来的吞吐改善直接归为该方案能实现的收益。策略实验应进一步围绕工作集临界点做较小预算步长，直接测物理回收量和恢复成本。

CPU 回载可能比重新执行 target prefill 更便宜，但本轮未开启 HiCache，无法从这些数值推导 CPU 路径的实际带宽、时延或收益。

## Qwen3.8-27B

8K/12-state、16K/12-state 与 16K/8-state 均已完成四种负载、各三次重复，合计 801 个测量请求，错误为 0。

- 固定 12-state，将 KV 从 8K 提到 16K，六会话后续轮次仍均为 0% 命中，吞吐约 39.7 → 40.4 tok/s；burst 约 77.5 → 78.4 tok/s，没有容量翻倍带来的明显提升。
- 16K 的会话阶段原生采样仍有 3328–3840 个 free KV slots，同时出现 free Mamba slots=0。仅按 KV token 容量判断会漏掉 checkpoint 状态压力；checkpoint 的数量、位置与有效匹配边界仍需分开归因。
- 6K 长生成的 KV eviction 从约 5632 降到 2048 tokens，但两档 inactive probe 命中均为 0%；不能把减少 KV 驱逐直接解读成有效恢复改善，需结合无压力的 checkpoint 边界探针。
- 固定 16K KV 和并发上限 2，state slots 从 12 降到 8 后，共享前缀热命中从 96% 降到 0%，热访问 TTFT 从 88.4ms 增到 602.5ms，约 6.8 倍；三次重复一致。persistent conv+SSM 从约 0.95 降到 0.65 GiB，verify intermediate 仍约 1.72 GiB。这是明确的 state 容量取舍，不是由 KV 容量或并发上限变化推断。
- 三档 burst 吞吐约 77.5 / 78.4 / 76.4 tok/s，未观测到 retract。不能把这几个百分点的波动当成策略收益；本轮没有实现策略，且并发上限固定为 2。
- 27B 三档共完成 9 次取消；18 对冷/热、18 对热/驱逐后输出比较未出现分叉，属于有限 probe 的正确性观察。

checkpoint 独立探针现已完成，见 [CHECKPOINT_RESULTS.md](CHECKPOINT_RESULTS.md)。8-state 单路径可以命中，ABAB 复访均 miss，而 ABBA 的 B 能命中；12-state 两种顺序都命中。会话扫描中 12-state 的两会话可命中、四/六会话全 miss。无压力的 P1536 复访命中 1024，加长输入后命中 1536，说明边界限制本身并不必然导致 0%。这些结果进一步支持 checkpoint 工作集与保留策略的影响，但还没有逐节点的 LRU victim 记录。

校准时，16-state-slot / 8K-token 配置实际分配：持久 conv+SSM 约 1.25 GiB，verify intermediate SSM+conv 约 1.72 GiB；target KV 约 0.25 GiB，draft KV 约 0.08 GiB。Load weight end 的增量统计为 target 约 20.14 GiB、draft 约 3.96 GiB；不是直接按参数数目推算。

此路径 target/draft KV 都使用 FP8。按配置，target FA KV=32 KiB/token，draft context KV=10 KiB/token；BF16 projected context hidden=10 KiB/token，原始五层 hidden=50 KiB/token。因而直接用 BF16 projected hidden 替代 FP8 context KV 并不节省 payload；需要基于不同表示精度或不同保留范围另行讨论。27B 的优化优先级应先关注 recurrent/checkpoint、verify 临时状态与 admission，不能照搬 8B 的 hidden-only 空间估算。

必须分别看 KV slots、GDN state slots、verify intermediate buffers 和实际 admission 上限，不用跨模型吞吐差异归因 attention。低 state cap 也会使原生 sizing 调低有效请求并发及 verify buffer；这种系统联动必须一起报告。

当前 checkout 已提供 `--mamba-max-states-per-path` 和独立的 `--enable-int8-mamba-checkpoint` / `--int8-mamba-ckpt-size` 能力；本轮没有开启或验证其收益。后者创建独立 checkpoint pool，需要同时核算 active pool 与 checkpoint pool 的总字节数，不能只按 checkpoint 压缩比声称整池容量翻倍；而且配置校验明确拒绝与 HiCache 同时启用。这些已有能力应先于新建状态管理 infra 评估。

## 实验边界

- 固定自然文本合成 trace，generation 使用固定长度；不是线上业务或质量 benchmark。
- graph/overlap 关闭，性能绝对值不代表最优部署配置；用于固定 runtime 设置下的状态容量研究。
- 稳态会话指标覆盖 cycle 1–3，并有三次重复，尚非长时间 trace 或线上 tail SLO。
- 整卡 NVML 采样峰值不是可驱逐状态量；理论 KV payload、预分配、实际 live/evictable/pinned 和临时 workspace 需区分。
- 新算法/状态策略尚未实现，观测到压力不等于已获得对应优化收益。
- 27B 正式配置显式固定 max-running-requests=2；burst 排队包含该人为上限，不能仅凭队列长度声称 state admission 已成为瓶颈，也不能据此声称释放 KV 后会提高并发。需要额外的同显存预算并发扫描验证。8B 的 max-running=8 相同，而实际运行峰值随 KV 容量改变，证据范围不同。
- flush 成功由 server.log 交叉核对。flush 后立即读取的 Prometheus 容量 gauge 可能仍是最近一次周期采样（可见 evictable 非零），不能将该快照解释为原子化的清空后池计数；active/running/queue 的零值也不是完整泄漏审计。
