# Policy 层 P2：显存划分（2026-10-27）

> 更正（P3，`../policy-p3-v0.5.22/FINDINGS.md`）：capture 后定池大小对 plain 同样有效，plain 的 KV 涨到 90.2K。两边都打开后，DSpark 在 W2 并发 64 下仍是 plain 的 0.70 倍。下文的 1.08 倍是与默认配置的 plain 比较的结果；这个开关回收的是两边共有的余量，不是推测本身的显存成本。

问题：8B 上推测解码让 KV 容量少了 26–35%，W2 高负载下只有 plain 的 0.5–0.7 倍（见 `../phase1-followups-v0.5.22/FINDINGS.md`）。这部分显存能不能在不改模型的前提下要回来？

- 环境：Qwen3-8B BF16，RTX 5090，FA4 attention，开发分支 `lvelf/policy-layer`；
- 负载：W2（24 篇 4096-token 文档，Zipf α=1，输出 128），以及 w1-natural（用来检查激活值峰值是否安全）；
- 每个点单次运行；原始数据在 `work_dirs/stress-qwen3-8b/runs/p2a/`。

## 服务期间的显存峰值

第一阶段的压力测试每秒采样一次 NVML。卡的总量是 31.8 GiB，下表是所有负载点里的最大占用：

| | 峰值 | 一直没用到的 |
|---|---|---|
| plain | 26.5 GiB | 5.4 GiB |
| DSpark | 28.2 GiB | 3.7 GiB |
| EAGLE3 | 27.2 GiB | 4.6 GiB |
| DFlash | 30.6 GiB | 1.3 GiB |

SGLang 按 `mem_fraction_static` 给 CUDA graph 和激活值预留了余量，这个余量是按启发式算的，偏保守。推测配置还额外有 2.2 GB 的推测 graph 不计入静态预算，结果就是 DSpark 即使在峰值时也有约 3.5 GB 闲置。

## capture 后定池大小（`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING=1`）

这是 SGLang 自带、默认关闭的选项：
- KV 池用 VMM 预留虚拟地址；
- 捕获完 CUDA graph 后量一下实际剩余显存，扣掉激活值余量，剩下的全部给 KV。

W2 吞吐（tok/s，括号内为相对 plain 的倍数）：

| 配置 | KV 容量（token） | 并发 16 | 并发 64 |
|---|---|---|---|
| plain（默认） | 58,687 | 665 | 1006 |
| DSpark（默认） | 38,093 | 589（0.89） | 684（0.68） |
| **DSpark + capture 后定池** | **60,775** | **869（1.31）** | **1088（1.08）** |
| DSpark + capture 后定池，fraction 0.90 | 51,347 | 797（1.20） | 916（0.91） |

- **只打开这一个开关，DSpark 在 W2 高负载下就从 plain 的 0.68 倍变成 1.08 倍**，KV 容量甚至超过了默认配置的 plain。
- 安全性：w1-natural 并发 64 下，DSpark + capture 后定池跑出 2068 tok/s（默认配置是 2088），没有 OOM。启动后剩 1.15 GB 显存，运行中最低到 0.74 GB（有 Triton 告警，但没出错）。
- 手动把 fraction 设成 0.90 反而更差：显式指定 fraction 后，SGLang 不再跳过 graph 的预留。
- 注意这里的 plain 没有打开这个开关，不完全公平。P3 会把两者都打开后再比较。

## 对 policy 层"显存划分"部分的结论

- 在 8B dense 模型上，推测解码的容量损失**大部分来自默认的静态划分过于保守**，而不是推测本身必需的显存。机制早就有了，只是默认没开。27B 上 MTP 那 5 GB 闲置（Q3）也是同一类问题：显存划分的记账不对。
- 原计划的 P2b（运行时用 VMM 扩大 KV 池）在 8B 上**没有必要了**：打开 capture 后定池大小之后，启动时只剩约 1.15 GB，没有可扩的空间。VMM 那一层本身支持在运行中继续映射（`KvVmmBufferOwner.ensure_prefix` 可以用更大的值反复调用），缺的是分配器不清空地追加空闲页，以及同步各处缓存的 `max_total_num_tokens`。这条路只在"关掉推测、释放 draft 权重和 graph"时才有意义，留作后期工作。
- 剩下真正属于推测本身的显存成本：draft 权重约 2 GB，draft KV 每 token 多 14%，推测 graph 约 2.2 GB。
