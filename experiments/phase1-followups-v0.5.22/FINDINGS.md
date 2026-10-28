# 第一阶段后续调查（2026-10-27）

搭 policy 层之前，把第一阶段和队友复核（`../dspark-state-management/TEAMMATE_REVIEW.md`）里的疑点逐一查清。单卡 RTX 5090 32GB，WSL2。

本目录只放结论和汇总表。harness、输入和逐请求的原始结果留在 benchmark worktree 的 `work_dirs/`：
- `ckpt-divergence-27b/`：Q1、Q2 的 `probe*.py`；
- `stress-qwen38-27b/runs/{e3-v1,q3-mtpfix}/`：Q3、Q4；
- `stress-qwen3-8b/runs/q5-profile/`：Q5；
- `hicache-cost-8b/`：Q6。

除特别注明外，源码都是 v0.5.22 `baa9befa38`，没有改 runtime。

| 文件 | 内容 |
|---|---|
| `q1q2-output-comparisons.csv` | 所有输出对比：是否逐 token 一致、首个分叉位置、首 token top-5 logprob 的最大偏移 |
| `q1-teacher-forced.csv` | 同一段续写在不同 state 下逐 token 打分的差异 |
| `q3q4-27b-serving.csv` | 27B 的负载扫描：同池 plain 对照，以及 MTP 初始化修复前后 |
| `q5-8b-decode-profile.csv` | 8B 自然对话并发 64 时，每轮迭代的 draft / verify / 调度耗时中位数 |
| `q6-8b-hit-cost.csv` | 8B 前缀命中三种来源的 TTFT：GPU 命中、CPU 取回、重算 |

## 结论一览

| # | 问题 | 结论 |
|---|---|---|
| Q1 | 27B 复用 GDN checkpoint 后贪心输出与冷启动不同 | 不是 bug，是数值路径差异，teacher-forced 打分没有系统性下降 |
| Q2 | 27B HiCache 回载后输出又不同于 GPU 热复用 | 回载本身逐位正确；差异来自 prefill 后的 KV 去重 |
| Q3 | MTP 的 embedding/head 在划分池子之后才共享，约 5 GB 闲置 | 已证实；修复后 MTP 在 27B 上高负载基本追平或超过 plain |
| Q4 | B 组缺少同预算 plain 对照 | 池子相同时，推测快 1.25–2.1 倍；高负载损失全部来自池子预算 |
| Q5 | 8B 高并发下推测的计算侧损失 | verify 占约 80%，70–85% 的 verify 计算被浪费 |
| Q6 | 前缀从 CPU 取回 vs 重算 | 取回比重算便宜 3–20 倍（1K–16K） |
| Q7 | B 组重复一轮 | 按用户要求暂停（以 8B 为主），已完成的点保留在 `e2-rep2` |

## Q1：27B checkpoint 复用后输出分叉

配置同队友：KV 8192、12 个 state、2 个运行请求、page 64、chunk 512、FP8 KV、bf16 SSM、`extra_buffer`，graph 和 overlap 关闭。prompt 取 ShareGPT 文本，长度分两组：
- 3150 / 3190 / 3300：走普通的跟踪路径；
- 3200 / 3328 / 3456：64 的倍数，走分叉点强制跟踪（`_force_track_h`）。

| 对照 | 结果 |
|---|---|
| 冷启动重复 | 逐 token 一致 |
| 复用 checkpoint vs 冷启动 | 第 2–31 个 token 分叉，首 token 偏 0.25–0.94 |
| plain（不推测） | 同样分叉，与推测解码无关 |
| 冷启动只改 chunk 大小（512→1024） | 首 token 偏 0.30–0.79 |
| 同一请求与另一请求并发 | 首 token 不变，解码第 2–77 个 token 分叉 |
| 同一深度的 state 用两种合法方式得到（chunk 内中间 h vs 先 prefill 前缀） | 两者互相分叉，首 token 偏 0.13–0.62 |
| fp32 SSM state | 仍然分叉 |
| KV 换到别的物理页 | 逐 token 一致，位置无影响 |
| teacher forcing：固定续写，比较平均 logprob 差 | 复用 checkpoint：−0.03 到 +0.01；删掉 64 个上下文 token：−0.37 到 −0.002，5 例中 4 例明显下降 |

结论：没有发现 state 取错。分叉来自几类数值路径差异：
- 已缓存前缀的 KV 是 FP8，同一次 extend 内的 token 是 bf16；
- chunk 和 GEMM 的形状不同。

这个 NVFP4 checkpoint 的 logit 很粗，top-1/top-2 经常完全打平，所以任何扰动都会在几十个 token 内翻转贪心选择。

另外，同一序列在两次服务启动之间重放时，冷启动和从 3072 复用逐位一致，但从 3136 复用的首 token 不同（偏 0.88）。部分路径跨进程不可复现，可能是 Triton autotune 选到了不同配置（未证实）。

## Q2：HiCache 回载 vs GPU 热复用

队友补丁 `00fd51eab3`（Mamba `layer_first`），HiCache 设置为 `direct` + `layer_first`、`write_through`、host ratio 6。

- **普通路径**：回载与 GPU 复用逐 token 一致，DSpark 和 plain 共 6/6；回载后再 extend 14、64、128、164 个 token 也都一致（4/4）。
- **3200 这类长度**：复现了队友的现象。首 token 逐位一致，解码第 1 步起出现 0.001–0.08 的差异，之后分叉；回载后再在 GPU 上复用，与回载那次逐位一致。
- **机制（代码追踪加实验验证）**：请求的 extend 跟踪到一个完整 chunk 时，prefill 结束后会把这段插入 radix 树。
  - 如果树上已有同 token 的 GPU 页（之前的请求经另一条数值路径算出），就释放自己的 KV，把请求的页表改指过去（`mem_cache/unified_radix_cache.py:1213-1216`，Rust `unified_tree_core.rs:1954-1980`）。解码因此读到的是别的请求算的 KV。
  - 如果那一页已被驱逐到 CPU，就保留自己的 KV（`unified_tree_core.rs:1900-1904`）。
- **验证**：在同一次启动里构造"树上已有该页"和"没有该页"两种 GPU 复用，得到同样的特征（首 token 一致，解码第 1 步差 0.008）。
- 两种 KV 都是同一段 token 的合法编码，所以这不是 bug。

**对实验方法的影响**：
- 27B 可以当性能基线；
- 正确性检查要改用 teacher-forced logprob 或 KL，不能用"贪心输出逐 token 一致"；
- 需要逐位对比时，只在同一次启动内比较，并留意是否发生了去重。

## Q3：MTP 显存记账

**问题**：`EagleWorkerV2` 先加载 MTP 头（含自带的 bf16 embedding 和 lm_head，各 2.54 GB），target 的池子按这时的剩余显存划分；之后 `alloc_memory_pool` 才调用 `init_lm_head` 共享 target 的 embedding/head，释放出的约 5 GB 只能闲置。

**诊断补丁**（只用于实验，没有提交到仓库）：把共享提前到 `__init__` 的末尾。

```diff
         self.plan_stream, self.plan_stream_ctx = get_plan_stream(self.device)
+
+        # Diagnostic (Q3): share the target embed/lm_head before the target KV
+        # pool is sized, so the draft's duplicate copies are not charged to it.
+        self.init_token_map()
+        self.init_lm_head()
```

cookbook MTP 配置（EAGLE 3/1/4 + ReplaySSM，`mem-fraction-static` 0.93，mamba 比例 2.17）：

| | target KV | state slot | 最大并发 |
|---|---|---|---|
| 修复前 | 21,130 | 5 | 1 |
| 修复后，0.93 | 71,538 | 53 | 13 |
| 修复后，0.88 | 58,147 | 38 | 9 |
| plain | 71,824 | 62 | 15 |

修复后 0.93 在捕获 graph 后只剩 0 GB 显存，低并发 TPOT 从 7.3 ms 恶化到 9.5 ms。cookbook 的 0.93 是在 5 GB 闲置的前提下定的，修复后应该调低到 0.88。

修复后 0.88 的吞吐（tok/s，括号内为相对 plain 的倍数）：

| 负载 | 并发 1 | 4 | 8 | 16 | 修复前 MTP 并发 16 |
|---|---|---|---|---|---|
| W2 共享前缀 | 124（1.82） | 304（1.45） | 385（1.18） | 427（0.91） | 206（0.44） |
| 自然对话 | 125（1.71） | 403（1.62） | 642（1.60） | 698（1.25） | 372（0.66） |

只剩 W2 并发 16 低于 plain，此时 MTP 最多同时跑 9 个请求，plain 是 15 个。这个问题值得报给上游。

## Q4：27B 同池子对照

plain 使用和推测配置完全相同的 KV 和 state 池（`--max-total-tokens`、`--max-mamba-cache-size`，prefill graph 关闭），单次运行。单元格为 tok/s，括号内是推测相对同池 plain 的倍数：

| 负载 | 并发 | 满池 plain | 同池 plain（DSpark-n2 的池子） | DSpark-n2 | 同池 plain（MTP-n4 的池子） | MTP-n4 |
|---|---|---|---|---|---|---|
| W2 | 1 | 68 | 62 | 119（1.92） | 63 | 106（1.68） |
| W2 | 16 | 470 | 99 | 173（1.75） | 159 | 206（1.30） |
| 自然对话 | 1 | 73 | 72 | 150（2.08） | 72 | 124（1.72） |
| 自然对话 | 16 | 560 | 125 | 257（2.06） | 237 | 372（1.57） |

池子相同时，推测在所有点都快 1.25–2.1 倍。高负载下输给满池 plain，全部是池子预算（显存挤占）造成的。这是 A 组 E3 的对应实验，结论比 A 组更强（A 组同容量时只快 11–18%），因为 27B 上推测配置能跑的 batch 本来就小。

## Q5：8B 高并发下的计算侧

w1-natural 负载，并发 64，1024 个请求，cookbook 风格（graph + overlap）。第 25 秒起抓 20 步 decode 的 torch profile（关闭 Python 调用栈追踪，否则一步会被拖到几百 ms）。每轮迭代取中位数：

| | batch | 每请求 verify token | draft | verify / decode | 一轮迭代 | 接受长度 | 每 token 时间 | bench 吞吐 |
|---|---|---|---|---|---|---|---|---|
| plain | 64 | - | - | 13.6 ms | 14.3 ms | 1 | 14.3 ms | 2098 tok/s |
| DSpark | 62 | 8 | 8.1 ms | 45.5 ms | 56.7 ms | 2.84 | 20.0 ms（0.72 倍） | 1885（0.90 倍） |
| DFlash | 61 | 16 | 16.5 ms | 84.6 ms | 104.3 ms | 2.26 | 46.2 ms（0.31 倍） | 1058（0.50 倍） |

- **verify 占约 80%。**
  - 每轮要对 batch × D 个 token 做 target 前向：DFlash 约 976 个，DSpark 约 496 个。这已经是计算瓶颈，每个 verify token 约 0.085–0.09 ms。plain 一步 64 个 token 只要 13.6 ms，主要花在读权重。
  - 接受长度只有 2.3–2.8，70–85% 的 verify 计算被浪费。
- draft 占 14–16%，CPU 和调度只占 2–5%（被 overlap 掩盖）。
- 结论：计算侧可调的是"随 batch 大小缩小 verify 宽度 D，必要时关闭推测"。
- 另外看到，约 30K token 的超长对话分块 prefill 时，每块 4096 token 占 GPU 360–560 ms，期间所有 decode 都被挡住。

## Q6：8B 前缀命中的代价

graph 和 overlap 开启，KV 池 40,000 token，page 64，HiCache 用 `direct` + `layer_first`、`write_through`、host ratio 2。每个请求只生成 1 个 token，客户端延迟近似 TTFT。每点 3 次取中位数，服务器空闲。

| 前缀长度 | 重算 | GPU 命中 | CPU 取回 | DSpark：CPU 取回 |
|---|---|---|---|---|
| 1K | 93 ms | 17 ms | 34 ms | 38 ms |
| 4K | 319 ms | 22 ms | 42 ms | 47 ms |
| 8K | 686 ms | 22 ms | 54 ms | 60 ms |
| 16K | 1567 ms | 23 ms | 76 ms | 86 ms |

- CPU 取回的成本：约 15 ms 固定开销，加上每 token 约 2.7 µs，接近 PCIe 5.0 带宽；layer_first 让传输和计算按层重叠。
- 重算的成本随长度超线性增长，16K 时约每 token 96 µs。
- DSpark 多传 draft KV，取回时间多约 14%，与字节比（164 / 144 KiB）一致。
- 限制：只测了空闲服务器上的单个请求；主机内存只有 47 GB。

## 对 policy 层的含义

- **显存侧**：要调的是各类用途之间的边界（verify 中间状态、state 池、draft 权重和 KV）。"把前缀卸到 CPU"是一个现成、便宜的动作。MTP 的那部分损失靠修好记账就能消除。
- **计算侧**：随 batch 大小调整 verify 宽度，必要时关闭推测。在 27B 这类 batch 本来就小的配置里，这一侧收益有限。
- **评估方法**：正确性检查用 teacher-forced logprob 或 KL。
