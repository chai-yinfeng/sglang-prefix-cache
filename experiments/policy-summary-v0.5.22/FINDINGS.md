# 第二阶段汇总：推测解码的条件 policy 层（2026-10-27）

目标（用户确定）：做一个抽象的 policy 层，从推测解码和调度器读取信号，决定显存与计算怎么分配，再交给 SGLang 下层执行。形式上是一个按负载切换行为的条件 policy，由一个控制器驱动三个执行器：

- ① 显存划分；
- ② 按"位置接受概率 × 请求权重"分配 verify 名额；
- ③ 前缀分层。

代码在分支 `lvelf/policy-layer`，各阶段的结果见同级目录：

| 阶段 | 目录 | 一句话结论 |
|---|---|---|
| P0 / P1 | `policy-p0p1-v0.5.22/` | 骨架开销在噪声范围内；DSpark 原生的置信度规划器（compact ragged verify，需要 FA4）把 8B 自然对话并发 64 从 plain 的 0.99 倍提到 1.21 倍 |
| P1b | `policy-p1b-v0.5.22/` | 负面结果：按负载覆盖规划器的预算没有收益；规划器胜过所有固定预算比例 |
| P2 | `policy-p2-v0.5.22/` | capture 后定池大小（默认关闭）把 DSpark 的 KV 从 38K 提到 61K；P3 更正：plain 也同样受益 |
| P3 | `policy-p3-v0.5.22/` | HiCache 分层把 DSpark 在 W2 上的命中率拉回 plain 的水平，并发 64 吞吐 +21% |
| P4 | `policy-p4-v0.5.22/` | 请求加权的 verify 排序：高优先级 TPOT p50 −8% 到 −9%，吞吐 −0.6% 到 −3.3%；DFlash 和 EAGLE3 只做了分析 |

## 集成实验：所有有效组件叠加

- 机器：Qwen3-8B，RTX 5090，FA4，page 64；单次运行；原始数据在 `work_dirs/stress-qwen3-8b/runs/final/`；汇总表：`integration.csv`。
- `dspark-full`：
  - compact 置信度规划器（计算）；
  - `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`（显存）；
  - HiCache `direct` + `layer_first`、主机池 12 GB（分层）；
  - `--speculative-policy observe`（trace）。
- 对照组 `plain-pcs`：同样打开 capture 后定池大小，这是公平的对照。

| 负载 | 并发 | plain-pcs | DSpark 全部组件 | 倍数 | 第一阶段（默认配置）的倍数 |
|---|---|---|---|---|---|
| w1-natural | 16 | 846 | 1406 | **1.66** | 约 1.7 |
| w1-natural | 64 | 2101 | 2462 | **1.17** | 0.92 |
| W2 | 16 | 654 | 821 | **1.26** | 0.85 |
| W2 | 64 | 1521 | 1415 | **0.93** | 0.61 |

单位 tok/s。KV 容量：plain-pcs 90,240，DSpark 全部组件 59,136。W2 并发 64 的命中率：0.885 和 0.883。

- 第一阶段里推测解码在高负载下输给 plain 的两种机制，都基本被消除了：
  - 计算侧：置信度规划器，自然对话并发 64 时从 0.92 倍变成 1.17 倍；
  - 命中率：分层，W2 并发 64 时与 plain 持平。
- DSpark 在 W2 并发 64 下，绝对吞吐从第一阶段的 601 tok/s 提高到 1415 tok/s（+135%）。其中一部分来自两边共有的显存修正：plain 也从 984 提高到 1521。
- 唯一还没赢的点是 W2 并发 64（0.93 倍）。差距来自能同时运行的请求数：DSpark 最多 91 个，plain 127 个，DSpark 的 TTFT 中位数也高出 2.2 秒，都在排队。原因是推测本身的显存成本（draft 权重约 2 GB、推测 graph 约 2.2 GB、draft KV 每 token 多 14%）仍然压缩着运行中请求能用的 KV。

## 已确认的认识

1. **计算侧**：只有 batch × verify 宽度越过步时表的转折点（8B + 5090 上约 64–128 token/步）之后，verify 才开始浪费算力。DSpark 的置信度规划器已经在每一步、每个请求上做了接近最优的分配，policy 层不应该覆盖它的预算，只需要补上它没有的东西：请求权重（P4a）。
2. **显存侧**：8B 上大部分容量损失来自默认的静态划分过于保守，SGLang 早就有修正机制，只是默认关闭。27B 上 MTP 的 5 GB 也是记账问题，第一阶段后续调查的 Q3 已经证实。推测本身真正的显存成本，只能靠运行时回收（见下）。
3. **前缀侧**：从 CPU 取回比重算便宜 3–20 倍（Q6）。分层对推测配置是"白给"的收益，因为它补上的正是推测因 KV 变小而损失的命中率。
4. **评估方法**：在 NVFP4 这类 logit 很粗的模型上，贪心输出逐 token 一致不是有效的正确性标准，要用 teacher-forced logprob 或 KL。另外，单次运行的 p99 很容易被一两个离群请求带偏（P1 的尾延迟问题没有复现），需要看逐请求的分布。

## 后期：设计与可行性（没有实现）

### ① 的 d：持续高负载时回收推测显存

- **收益上限**：W2 并发 64 时，DSpark 全部组件是 1415 tok/s，plain-pcs 是 1521，差 7%。并发 16 时推测快 26%，所以必须按负载切换，不能一直关着。
- **已有的部分**：
  - VMM 竞技场按需映射物理页，KV 张量一开始就按预留的最大容量分配虚拟地址，所以扩池不会改变地址，已捕获的 CUDA graph 仍然有效（`mem_cache/kv_vmm_backing.py`，`KvVmmBufferOwner.ensure_prefix` 可以用更大的值反复调用）；
  - 启动时的 capture 后定池大小（`model_runner_components/kv_pool_runtime.py`）；
  - `release_memory_occupation`，但它只能整类释放，而且要求服务空闲。
- **缺的部分**，按依赖顺序：
  1. **运行时关闭推测**：现在把预算设为 0 只能做到"每个请求只 verify bonus token"，draft 前向照跑（并发 64 时 draft 每步要 8 ms）。需要一条"跳过 draft、直接走 target decode"的路径。
     - 重新开启时，DSpark 的 draft 上下文来自 target 第 [1, 9, 17, 25, 33] 层的 hidden state。关闭期间可以继续写入投影后的 draft KV（`dspark_kv_inject`，代价很小），这样重开时上下文仍然是新鲜的。
  2. **只释放 draft 权重和推测 graph**：需要按模型区分的释放标签，并且不能要求服务空闲；重开时重新加载权重（从 pinned 内存加载 2 GB，约几十毫秒）、重新捕获 graph（或保留 graph）。
  3. **扩大 KV 分配器**：追加新的空闲页（现在的 `allocator.resize` 会 `clear()`），同步 Scheduler、`pool_stats_observer`、指标里缓存的 `max_total_num_tokens`。DSpark 和 DFlash 的 draft KV 池按 target 的 slot 索引，必须同步扩。
  4. **重开推测时收缩**：先淘汰水位线以上的前缀并禁止在那里分配，再解除映射。
- 风险最大的是第 1 步和第 4 步：它们跨越推测 worker、调度器和缓存三个子系统。

### ① 的 c：hybrid 模型的 verify 中间状态按预算定大小（27B）

- verify 中间状态的形状是 `[L, max_reqs+1, D, ...]`，按"请求 × 位置"索引（`mem_cache/memory_pool.py:862-892`）。27B 上 DSpark 每个请求约 0.57 GB，是并发上限的主要来源。
- compact 规划器的预算 B 通常远小于 bs × D。如果中间状态改成按"verify token"扁平索引，再按 B 的上限来分配，在相同显存下能多跑几个请求。这需要改 GDN 的 verify kernel 和状态提交（`update_mamba_state_after_mtp_verify`）的索引方式。
- 另一条路是第一阶段后续调查的 O3：给 DSpark / DFlash 加 ReplaySSM 式的压缩。MTP 已经用 ReplaySSM 把这部分降到约 0.01 GB。
