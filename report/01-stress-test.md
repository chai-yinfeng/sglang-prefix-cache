# 第一部分：压力测试 —— 推测解码在显存压力下哪里输、输多少、policy 该往哪改

单卡 RTX 5090 32GB，SGLang v0.5.22（`baa9befa38`），除注明外没有改 runtime。本文汇总我们的压力测试（A 组 8B、B 组 27B、后续调查 Q1–Q6）和队友的固定容量机制实验。每一节大致对应一页 PPT：**现象 → 怎么发现的 → 有多严重 → 对 policy 的含义**。

---

## 0. 一句话结论

**低负载时，推测解码能快 1.5–2 倍；负载一高就比 plain 还慢：8B 降到 plain 的 0.50–0.72 倍，27B 降到 0.26–0.44 倍。** 主要原因是 draft 静态占住的显存挤掉了 target 的 KV 和 state 池（draft 权重、verify 中间状态、graph）。其次是大 batch 下 verify 的计算浪费。现有系统对计算开销有一点自适应机制，但不会随负载把显存还给 target。原 proposal 关注的 draft KV 是所有额外开销里最小的一项。

PPT 首页可以用的三个数字：
- 8B 共享前缀负载、并发 64：DSpark 只有 plain 的 **0.61 倍**。把 plain 的 KV 容量限制到和 DSpark 一样后，DSpark 反而快 **14%**。损失全部来自容量。
- 27B hybrid：DSpark/DFlash2 最多同时跑 **2–3 个请求**，plain 能跑 **15 个**。
- draft KV 只占 DSpark 容量损失的约 **1/4**（8B），在 27B 上只有 0.04–0.16 GB。

---

## 1. 实验设置（"怎么发现的"总览）

### 模型
| 组 | target | drafter | 特点 |
|---|---|---|---|
| A | Qwen3-8B BF16（KV 144 KiB/token） | EAGLE3（1 层，4 KiB/token）、DFlash b16、DSpark block7（各 5 层，20 KiB/token） | dense，纯 KV 压力 |
| B | Qwen3.8-27B NVFP4（48 GDN + 16 FA） | MTP（+ReplaySSM）、DSpark、DFlash2 | hybrid，KV、state、verify 中间状态同时受压 |
| C（队友） | Qwen3.5-4B（24 GDN + 8 FA） | DFlash | hybrid，用来复查 verify 中间状态的大小 |

### 配置
- 用 cookbook 推荐的部署参数：CUDA graph 和 overlap 开启。
  - 8B：`mem_fraction_static` 自动（0.776），`--max-running-requests 64`。
  - 27B：fp8 KV、bf16 SSM、`extra_buffer_lazy`，各算法沿用 cookbook 实测的 fraction。
- 队友的机制实验不同：固定 KV 8K/16K 槽，关闭 graph/overlap。绝对吞吐不能和我们的直接比。

### 负载
- **W2 共享前缀**：24 组真实文档前缀，每组 4096 token，按 Zipf(α=1) 访问；问题约 20 token，输出 128。前缀工作集约 **98K token**，大于所有配置的 KV 容量。
- **w1-natural 自然对话**：ShareGPT 对话套 chat template，不忽略 EOS；平均输入约 430、输出约 240 token。
- （早期的 W1 用 `ignore_eos` 的随机输入，接受长度被高估，已由 w1-natural 取代。）

### 实验
| 代号 | 做法 | 回答什么问题 |
|---|---|---|
| E1 | 启动探测：server log、`/server_info`、NVML 交叉核对 | 显存被谁占了 |
| E2 | `bench_serving` 扫描并发 1–64 | 推测解码的收益在哪个负载消失 |
| E3 / Q4 | 用 `--max-total-tokens`（以及 `--max-mamba-cache-size`）把 plain 限制到和推测相同的容量 | 损失来自容量还是推测本身 |
| E4 | 打开 `--speculative-adaptive` | 现有自适应机制能不能救 |
| Q5 | torch profiler 抓 decode 步 | 高并发下的计算花在哪 |
| Q3 | 诊断补丁（调整 MTP 初始化顺序） | 记账缺陷的影响有多大 |
| Q6 | 单请求 TTFT：GPU 命中 / CPU 取回 / 重算 | 前缀放到 host 值不值 |
| 队友 | 固定容量下调 KV 槽数、state 数和访问顺序 | 容量断崖和 checkpoint 保留压力 |

可靠性：8B 主要结果跑了两轮，36 个共同点的吞吐比值中位数为 1.000（范围 0.993–1.060）；27B 只跑了一次。

---

## 2. 问题一：推测解码一到高负载就输给 plain

**现象**：并发上去之后，推测解码的吞吐跌到 plain 以下。

**怎么发现的**：E2 并发扫描，记录各配置相对 plain 的吞吐倍数。

**有多严重**：

8B，W2 共享前缀（相对 plain 的倍数）：

| 并发 | EAGLE3 | DFlash | DSpark | plain tok/s |
|---:|---:|---:|---:|---:|
| 1 | 1.54 | 1.48 | 1.71 | 88 |
| 8 | 1.15 | 1.04 | 1.19 | 390 |
| 16 | **0.95** | **0.69** | **0.85** | 654 |
| 64 | 0.72 | **0.50** | 0.61 | 984 |

27B（相对 plain 的倍数）：

| 负载 | 并发 | MTP | DSpark | DFlash2 | plain tok/s |
|---|---:|---:|---:|---:|---:|
| W2 | 1 | 1.56 | 1.75 | 1.72 | 68 |
| W2 | 16 | **0.44** | **0.37** | **0.26** | 470 |
| w1-natural | 1 | 1.70 | 2.06 | 2.08 | 73 |
| w1-natural | 16 | 0.66 | 0.46 | 0.64 | 560 |

- 8B 在 W2 上从**并发 16 起**推测解码全线落后。27B 在 W2 上从**并发 4 起**、在自然对话上从**并发 8 起**全线落后。
- 27B 的拐点就是各配置的并发上限。超过之后吞吐不再增长，多出来的请求排队，TTFT 达到 8–16 秒。
- 低负载的收益是真实的。队友的低负载基线（cache-off，并发 1/8，graph 关闭）里，EAGLE3、DFlash、DSpark 分别是 plain 的约 1.6、1.9、2.3 倍。

**对 policy 的含义**：没有一个静态配置能在所有负载下都最优，推测要不要开、开多少必须**随负载决定**。

可做图：横轴并发，纵轴相对 plain 的吞吐倍数，四条线，标出 1.0 基准线。

---

## 3. 问题二：真正的原因是显存被静态挤占，不是推测本身

**现象**：推测配置的 target KV 容量更小，导致前缀命中率更低、能同时跑的请求更少。

**怎么发现的**：先用 E1 看显存构成，再用 E3（8B）和 Q4（27B）把 plain 限制到和推测完全相同的池子大小做对照。

**有多严重**：

8B 的 target KV 容量（E1）：

| | plain | EAGLE3 | DFlash | DSpark |
|---|---:|---:|---:|---:|
| KV 容量（token） | 58,687 | 43,609（−26%） | 38,542（−34%） | 38,093（−35%） |
| W2 并发 64 时的命中率 | 0.79 | 0.71 | 0.62 | 0.62 |

- W2 并发 64 时，plain 最多同时跑 45 个请求，推测配置只有 21–26 个。
- 四种配置的 static 预算（权重 + KV 池）都是约 23.3 GB，**draft 的权重和 KV 全部从 target KV 里扣**。
- 并发 1 时，DSpark 的 TTFT 中位数是 335 ms，plain 是 23 ms。对 DSpark 来说，中位数请求已经未命中。

同容量对照（E3，8B W2 并发 64，tok/s）：

```
plain（58.7K）  984
   └─ 容量缩到 38.1K → plain-kv38093  527   （−46%）
        └─ 同容量下开 DSpark → DSpark  601   （+14%）
```

- 容量相同时，推测解码的命中率和 plain 几乎一样（DSpark 0.62 vs 0.64）。**命中率下降完全来自容量。**
- 容量相同时，推测仍然更快：EAGLE3 和 DSpark 在并发 16–64 都快 1.11–1.18 倍。DFlash 是例外，只有 0.93–0.96 倍，属于计算侧的问题，见问题五。
- 工作集接近容量时，吞吐对容量非常敏感：容量少 35%，吞吐少 46%。

27B 同池对照（Q4，plain 用和推测完全相同的 KV + state 池）：推测在**所有点都快 1.25–2.1 倍**，例如 W2 并发 16 时 DSpark 173 vs 同池 plain 99。高负载下输给满池 plain，**全部是池子预算造成的**。这个结论比 8B 更强。

队友的固定容量实验从另一个角度看到同一件事：8B DSpark 的 KV 从 8K 提到 16K，六个会话的后续命中率从 **0% 变成 96%**，burst 吞吐从约 496 涨到约 866 tok/s。容量处在断崖附近时，一点显存就决定缓存是否有用。

**对 policy 的含义**：policy 最大的空间在**显存划分**。高负载时把推测占的显存（draft 权重、draft KV、verify 状态、graph）还给 target 的 KV/state 池，低负载时再收回。这里能参考的收益量级是 8B W2 并发 64 的 601 → 984 tok/s、27B W2 并发 16 的 206 → 470 tok/s。这些是不同配置之间的参照，不是严格上界，切换本身也有成本。

可做图：瀑布图 984 → 527 → 601；或者"同容量 vs 满容量"的双柱图。

---

## 4. 问题三：显存都去哪了？draft KV 是最小的一项

**现象**：原 proposal 要管理的 draft KV，在所有额外显存开销里最小。

**怎么发现的**：E1 显存构成探测；4B 的数据来自队友。

**有多严重**（单位 GB）：

| 项目 | 8B（A 组） | 27B（B 组） | 4B hybrid（队友，DFlash） |
|---|---|---|---|
| draft 权重（固定） | **1.95–2.23** | **3.95–5.84** | 1.36 |
| draft KV | 0.16–0.74 | 0.04–0.16 | 0.375 |
| verify 中间状态 | 无（dense） | DSpark **1.14**，DFlash2 **1.72**；每个请求约 0.57 | **6.75**（是 Mamba 池 3.12 的两倍多） |
| target prefill CUDA graph | 0.77 | 1.8–2.2 | — |
| 推测的 decode/verify graph | 1.3–2.0（在 static 预算之外） | 0.1–0.3 | — |

- 8B：DSpark 的 target KV 少了 2.82 GB，其中 draft KV 只有 0.72 GB。**即使把 draft KV 全部丢掉，也只能收回约 1/4 的损失**，大头是 draft 权重。
- 27B：draft KV 每 token 是 target 的 31%，但 KV 池本身被压得很小，绝对值可以忽略。
- 实测之外还有一个结构性的事实：SGLang 的 draft 和 target 共用同一个 slot 分配器，所以单独丢掉 draft KV 在今天也释放不出显存。

**对 policy 的含义**：policy 的对象应该是**所有推测相关的显存**（权重、verify 状态、graph、KV），而不只是 draft KV。"draft KV 选择性保留"适合放到相关工作里讨论，不适合作为主线。

可做图：每个配置一根堆叠条形图（target 权重 / draft 权重 / target KV / draft KV / state / verify / graph / 剩余）。

---

## 5. 问题四：hybrid 模型上，瓶颈是 state slot 和 verify 中间状态

**现象**：在 27B hybrid 上，限制并发的是 state slot，不是 KV。DSpark/DFlash 的 verify 中间状态比持久 state 池和 KV 加起来还大。

**怎么发现的**：E1 启动探测、E1 容量调优（显式指定 `--max-mamba-cache-size` 和 `--max-total-tokens`），以及队友的 state 数和访问顺序实验。

**有多严重**：

| | plain | MTP | DSpark | DFlash2 |
|---|---:|---:|---:|---:|
| 按 cookbook 配方，state slot 限制的最大并发 | **15** | 1 | 1 | 2 |
| 调参之后能达到的最大并发 | 15 | 4 | **2** | **3** |
| 每个请求的显存 | 约 0.44 GB | — | 约 0.9–1.0 GB（verify 0.57 + state 0.3 + KV） | 同 DSpark |

- verify 中间状态**随并发线性增长**，每多一个请求多约 0.57 GB（8 个 state slot × 74.8 MB）。MTP 用 ReplaySSM 把它压到 0.01 GB，但 DSpark/DFlash 在 GDN 上目前用不了 ReplaySSM（队友已确认）。
- W2 前缀命中率：plain 0.60–0.72，推测只有 **0.16–0.32**。推测配置的 KV 池（7.2K–17.6K token）只装得下 2–4 个 4K 前缀。
- 关掉 prefill graph 能省约 1.9 GB，但只多出 1 个并发。

队友的固定容量机制实验（27B DSpark）：

| 对照 | 结果 |
|---|---|
| state 固定 12，KV 8K → 16K | 后续命中率**仍然 0%**，吞吐 77.5 → 78.4 tok/s：扩 KV 解决不了 |
| KV 固定 16K，state 12 → 8 | 共享前缀热命中率 **96% → 0%**，热 TTFT 约 88 → 603 ms |
| 访问顺序（8 个 state） | ABAB 复访全部未命中，ABBA 中最近的 B 能命中；12 个 state 时都能命中 |

state 12 → 8 只省下约 0.30 GiB 持久状态，1.72 GiB 的 verify 中间状态不变。一个请求会产生多个 checkpoint，active、tracking 和 cached 状态共用同一个预算。所以"8 个 state"不等于能保留 8 个完整会话。

**对 policy 的含义**：
- hybrid 模型上，policy 要划分的是 **state 池和 KV 池之间的比例**，并且要把 verify 中间状态算进去：verify 宽度越大，需要的 state 越多。
- 长期方向是压缩 verify 中间状态（做 DSpark/DFlash 版的 ReplaySSM），成本较大。
- checkpoint 的保留要考虑多路径访问（ABAB），不能只按 LRU。

可做图：每个请求的显存构成（plain vs DSpark）；并发上限柱状图 15 vs 1–3。

---

## 6. 问题五：计算侧的浪费，大 batch 下 verify 不划算

**现象**：请求都装得下、没有显存压力的时候，推测在高并发下仍然更慢。

**怎么发现的**：E2 的 w1-natural 负载（短请求，KV 装得下），加上 Q5 的 torch profiler。

**有多严重**：

w1-natural 并发 64（8B）：

| | 吞吐倍数 | TPOT p50 | 接受长度 | 最大 running |
|---|---:|---:|---:|---:|
| plain | 1.00 | 21.2 ms | — | 64 |
| EAGLE3 | 1.03 | 22.5 ms | 2.27 | 63 |
| DSpark | 0.92 | 25.8 ms | 2.82 | 60 |
| DFlash（块宽 16） | **0.50** | **53.2 ms** | 2.23 | 64 |

Q5 profiling（w1-natural 并发 64，每轮迭代的中位数）：

| | 每请求 verify token | draft | verify/decode | 一轮迭代 | 每个输出 token 的时间 |
|---|---:|---:|---:|---:|---:|
| plain | — | — | 13.6 ms | 14.3 ms | 14.3 ms |
| DSpark | 8 | 8.1 ms | 45.5 ms | 56.7 ms | 20.0 ms |
| DFlash | 16 | 16.5 ms | 84.6 ms | 104.3 ms | 46.2 ms |

- **verify 约占每轮的 80%**。DFlash 每轮要对约 976 个 token 做 target 前向，这已经进入计算瓶颈（每个 verify token 约 0.085–0.09 ms）。
- 接受长度只有 2.3–2.8，所以 **70–85% 的 verify 计算被浪费**。
- draft 占 14–16%，CPU 和调度只占 2–5%（被 overlap 掩盖）。
- DSpark 的 0.92 倍里可能还混有准入差异（最大 running 60 vs 64）。DFlash 的 KV 没满、也没有 retract，退化基本是计算原因。
- 附带观察：约 30K token 的超长对话分块 prefill 时，每块 4096 token 占 GPU 360–560 ms，期间所有 decode 都被挡住。

**对 policy 的含义**：计算侧要做的是**随 batch 大小收缩 verify 宽度/预算，必要时关掉推测**。DFlash 一直按固定的 16 宽度 verify，是最明显的目标；DSpark 自带置信度 planner，空间较小。在 27B 这类 batch 本来就小的配置上，这一侧的收益有限。

可做图：每轮迭代时间的堆叠柱（draft / verify / 其他），三个配置并排。

---

## 7. 问题六：现有的自适应机制是拿显存换计算

**现象**：SGLang 自带的 `--speculative-adaptive`（只支持 EAGLE）在默认配置下比固定 EAGLE3 还差，大多数点比 plain 也差。

**怎么发现的**：E4，打开 adaptive，并检查 server log 里的显存。

**有多严重**：

| 负载 / 并发 | plain | EAGLE3 固定 | adaptive 默认 | adaptive 精简（候选步数 {0,1,3}） |
|---|---:|---:|---:|---:|
| W2 并发 64 | **984** | 710 | 293（0.30 倍） | 706 |
| w1-natural 并发 64 | 2118 | 2178 | 768（0.36 倍） | 2155 |

- 默认配置为每个候选步数（0/1/3/5/7）各建一套运行状态和 CUDA graph，**每套 1.2–1.6 GB**，启动后可用显存降到 **0.00 GB**。
- 把候选集合调小后能和 plain 持平，但**补不回显存损失**。W2 并发 64 仍比满容量 plain 少 28%，因为 draft 的权重、KV 和 graph 照样占着。

**对 policy 的含义**：计算和显存两个问题是**耦合**的。为计算侧的灵活性多建 graph 会吃掉 KV，所以两件事要由**同一个控制器**统一权衡，不能分别解决。顺带一个工程修补：adaptive 的 graph 数量应该受显存预算约束（O4）。

---

## 8. 问题七：记账缺陷，MTP 约 5 GB 显存从未被使用

**现象**：27B 的 MTP 启动后剩 5.02 GB 空闲（plain 只剩 0.20 GB），而它的并发上限只有 1。

**怎么发现的**：E1 探测发现异常，读 `eagle_worker_v2.py` 找到初始化顺序，再用诊断补丁验证（Q3）。

**原因**：
1. MTP 头加载时带了自己的一份 bf16 embedding 和 lm_head，各 2.54 GB。
2. target 的 KV/state 池**按这时的剩余显存划分**。
3. 之后 `init_lm_head` 才改成和 target 共用 embedding/head，释放出的约 5 GB 只能闲置。

**有多严重**（修复 = 把共享提前到划分池子之前；诊断补丁，没有提交）：

| | target KV | state slot | 最大并发 | W2 并发 16 | 自然对话 并发 16 |
|---|---:|---:|---:|---:|---:|
| 修复前 | 21,130 | 5 | 1（调参后 4） | 206（0.44 倍） | 372（0.66 倍） |
| 修复后，fraction 0.88 | 58,147 | 38 | **9** | 427（**0.91 倍**） | 698（**1.25 倍**） |
| plain | 71,824 | 62 | 15 | 470 | 560 |

修复后要把 fraction 从 cookbook 的 0.93 调低到 0.88，否则捕获 graph 后会只剩 0 GB，低并发 TPOT 变差。

**对 policy 的含义**：policy 需要一份**准确的显存账本**，要知道 draft 权重、KV、graph 各占多少、哪些部分是共享的。这个具体问题本身是 bug，值得报给上游，不算 policy 的贡献。

---

## 9. 问题八：前缀放不下时，放到 host 比丢掉便宜得多

**现象**：容量不够时前缀被驱逐，只能重算。如果改成取回 CPU 上的副本，代价会小很多。

**怎么发现的**：Q6，单请求 TTFT 微基准（8B，HiCache `direct` + `layer_first`，服务器空闲）。

**有多严重**：

| 前缀长度 | 重算 | GPU 命中 | CPU 取回 | DSpark：CPU 取回 |
|---|---:|---:|---:|---:|
| 1K | 93 ms | 17 ms | 34 ms | 38 ms |
| 4K | 319 ms | 22 ms | 42 ms | 47 ms |
| 8K | 686 ms | 22 ms | 54 ms | 60 ms |
| 16K | 1567 ms | 23 ms | 76 ms | 86 ms |

- CPU 取回比重算便宜 **3–20 倍**。取回成本约等于 15 ms 固定开销加上每 token 2.7 µs，接近 PCIe 5.0 带宽。重算成本随长度超线性增长。
- DSpark 多传 draft KV，取回时间多约 14%，和字节比（164 / 144 KiB）一致。
- 反面例子来自队友的基线：前缀工作集（24K）超过容量（16K 槽）时，所有配置的复访命中率都只有约 **0.2%**，缓存等于没开。

**对 policy 的含义**：显存不够时，"把前缀卸到 host"是一个现成而且便宜的动作。驱逐时应该按**重建代价 / 字节数**评估前缀的价值（原 proposal 的 cost-aware 思想用在这里），而不是只看最近使用时间。限制：只测了空闲服务器上的单个请求，主机内存只有 47 GB。

---

## 10. 方法上的发现：不能用"贪心输出逐 token 一致"检查正确性

**现象**：27B 复用 GDN checkpoint，或从 HiCache 回载之后，贪心输出和冷启动不同。

**怎么发现的**：Q1/Q2 逐项排查：换物理页、fp32 SSM、只改 chunk 大小、teacher-forced 打分，再加上代码追踪。

**结论**：
- 没有发现 state 取错。分叉来自数值路径差异：已缓存前缀的 KV 是 FP8，同一次 extend 内的是 bf16；chunk 和 GEMM 的形状也不同。这个 NVFP4 checkpoint 的 logit 很粗，top-1/top-2 经常打平。
- HiCache 回载本身逐位正确。差异来自 prefill 后的 KV 去重：请求改为读别的请求算出的同 token KV。
- teacher-forced 打分：复用 checkpoint 时 logprob 变化在 −0.03 到 +0.01 之间，没有系统性下降。作为对照，删掉 64 个上下文 token 时 5 例中有 4 例明显下降。

**对 policy 的含义**：policy 改动的正确性检查要用 **teacher-forced logprob 或 KL**，不用"贪心逐 token 一致"。逐位对比只在同一次启动内做，并留意是否发生了去重。

---

## 11. 汇总：问题 → 严重程度 → policy 改进方向

| # | 问题 | 最关键的数字 | policy 改进方向 | 后续在哪做了（下一部分展开） |
|---|---|---|---|---|
| 1 | 高负载下推测输给 plain | 8B W2 并发 64：0.50–0.72 倍；27B W2 并发 16：0.26–0.44 倍 | 按负载决定推测的开关和力度 | 贯穿 policy 层的各轮 |
| 2 | 损失来自显存静态挤占 | 984 → 527（−46%）→ 601（+14%）；27B 同池快 1.25–2.1 倍 | **主线**：显存划分随负载调整，高负载把推测的显存还给 KV/state | P2 capture 后定池；第二轮运行时扩 KV 池；第三轮预留 VA、运行时暂停推测 |
| 3 | draft KV 是最小一项 | 只占 DSpark 损失的约 1/4 | 管理对象扩大到权重、verify、graph | 第三轮 draft 权重可释放（只做了原语，没有接入） |
| 4 | hybrid：state slot 和 verify 状态才是瓶颈 | 并发 15 vs 2–3；verify 每请求 0.57 GB | state:KV 比例要计入 verify 状态；压缩 verify 状态 | 第四轮、第五轮 m1（按 (S+D)/S 划分）；压缩没做 |
| 5 | 大 batch 下 verify 计算浪费 | DFlash 并发 64：0.50 倍；verify 占 80%，浪费 70–85% | verify 宽度随 batch 收缩，必要时关推测 | P1/P1b/P4（DSpark）；第四轮、第五轮（DFlash 运行时宽度） |
| 6 | 计算和显存耦合 | adaptive 默认：0 GB 空闲，0.30–0.36 倍 | 一个控制器同时看两侧 | policy 层的 sensor → policy → actuator 结构 |
| 7 | 记账缺陷（MTP） | 并发 1 → 9，W2 并发 16：0.44 → 0.91 倍 | 准确的显存账本；bug 报上游 | Q3 诊断补丁；第二轮的 memory ledger |
| 8 | 前缀驱逐后只能重算 | host 取回便宜 3–20 倍 | 前缀分层（GPU / host / 丢弃），按价值密度驱逐 | P3 HiCache 分层；第二轮 value-density 驱逐 |
| 9 | 正确性检查方法 | 0.5% 级 token 翻转是数值噪声 | 用 teacher-forced / KL 检查 | 第四轮、第五轮的正确性检查 |

可以给 policy 用的信号（压力测试里已经观测到）：running 请求数、排队长度、前缀命中率、接受长度、每步耗时、剩余显存。

---

## 12. 口径与限制（讲的时候注意别说过头）

- **"现有系统完全没有显存管理"说得太强。** SGLang 已经有分组件的池子、HiCache、hybrid 统一字节池等机制。准确的说法是：它**不会随负载**把 draft 权重、graph、verify 空间还给 target。
- **984 vs 601、470 vs 206 是不同配置之间的性能参照，不是新策略的保证收益或严格上界。** 动态切换、重建、host 传输和 graph 切换都有成本。
- **E3 控制的是 target KV 的 token 容量，不是总显存相同。** 被限制容量的 plain 留下了更多未分配的显存。W2 的 98K 工作集大于两种容量，双方都有容量压力。相同的 static fraction 也不是分别调到最优的结果。
- **27B 的 E2 比较的是两套部署配置**（fraction、graph 设置都不同），不只是推测开关；Q4 补了同池对照。27B 只跑了一次。
- 并发 ≥ 8 的单格有波动，复现时单格最大偏差 34%。
- **27B 不是正确性已确认的基线。** 队友的 27B CPU 回载严格输出检查没有通过，状态边界问题和数值差异还没有完全区分开。27B 只当性能基线用。8B 的 target+draft 回载已经验证过。
- 还没覆盖的负载：W3 多轮/agentic、W4 长上下文。长上下文下 draft KV 占比更高，"draft KV 不重要"这个结论目前只限于这些长度。
- W1（`ignore_eos`）的 3.8–4.1 倍加速是被高估的，以 w1-natural 为准。

---

## 13. 数据来源

| 内容 | 文件（仓库内 `experiments/`） |
|---|---|
| 第一阶段总结 / 提升空间地图 | `stress-test-summary.md` |
| 8B：E1–E4、复核、重复 | `stress-qwen3-8b-v0.5.22/FINDINGS.md`，`e2-v1/`、`e2-rep2/`（`summary.csv`） |
| 27B：E1、容量调优、E2 | `stress-qwen38-27b-v0.5.22/FINDINGS.md`，`e1-probe/`、`e1-capacity/`、`e2-v1/` |
| Q1–Q6 后续调查 | `phase1-followups-v0.5.22/FINDINGS.md` 和 `q*.csv` |
| 队友：8B 基线、容量压力负载 | `qwen3-8b-prefix-baseline-v0.5.22/RESULTS.md` |
| 队友：4B hybrid 显存构成 | `qwen35-4b-hybrid-prefix-baseline-v0.5.22/RESULTS.md` |
| 队友：KV/state 容量断崖、checkpoint 访问顺序、CPU 回载 | `dspark-state-management/FINDINGS.md`、`CPU_OFFLOAD.md`；对我们结果的复核见 `TEAMMATE_REVIEW.md` |

原始数据（server log、逐请求结果）不在 git 里，在 benchmark worktree 的 `work_dirs/` 下。
