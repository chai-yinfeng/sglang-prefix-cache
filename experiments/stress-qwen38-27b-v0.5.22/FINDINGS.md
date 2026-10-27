# B 组结果：Qwen3.8-27B NVFP4（hybrid GDN，2026-10-27）

原始数据在 worktree 的 `work_dirs/stress-qwen38-27b/runs/`。

**模型**：`RadixArk/Qwen3.8-27B-NVFP4`（FP4 lm_head），64 层，其中 48 层 GDN、16 层 full attention（GQA 4 头 x 256）。drafter：MTP（checkpoint 自带）、`RadixArk/Qwen3.8-27B-DSpark`（5 层）、`incoai/Qwen3.8-27B-DFlash2`。

**配置**：cookbook 的 RTX 5090 配方。fp8 KV，bf16 SSM 状态，`extra_buffer_lazy`（每个请求 4 个 state slot），各推测方式用 cookbook 实测的 `mem-fraction-static`：plain 0.9，MTP 0.93（加 ReplaySSM），DSpark 0.88，DFlash2 0.91（chunked prefill 1024）。mamba 比例按 cookbook 公式、以 L = 4400 计算：plain/MTP 2.17，DSpark/DFlash2 6.52。并发上限请求 32。启动时带 `MAX_JOBS=2` 等编译限制（见 `01-stress-test-plan.md` 的风险一节）。

## E1 显存构成（启动探测）

| | plain | MTP+ReplaySSM | DSpark | DFlash2 |
|---|---:|---:|---:|---:|
| target 权重 | 20.14 | 20.14 | 20.14 | 20.14 |
| draft 权重 | 0 | **5.84** | **3.95** | **4.02** |
| target KV（fp8） | 2.20（71.8K token） | 0.64（21.1K） | 0.48（15.9K） | 0.42（14.0K） |
| draft KV | 0 | 0.04 | 0.16 | 0.14 |
| mamba state 池 | 4.60（62 slot） | 0.44（5 slot） | 0.44（5 slot） | 0.65（8 slot） |
| verify 中间状态 | 0 | 0.01（ReplaySSM 环形缓冲） | **1.14** | **1.72** |
| CUDA graph | 1.87 | 1.91 | 2.15 | 1.04 |
| 初始化后剩余 | 0.20 | **5.02** | 0.99 | 1.09 |
| **state slot 限制的最大并发** | **15** | **1** | **1** | **2** |
| 探测请求的接受长度 | - | 2.46 | 1.88 | 2.78 |

单位 GB。

### 发现
1. **按 cookbook 的显存设置，推测解码在这张卡上最多只能同时服务 1–2 个请求，plain 能服务 15 个。** 这与 cookbook 的说明一致（配方只在并发 1 下验证过），但意味着推测解码在 hybrid 大模型上几乎只适用于单用户场景。
2. **并发上限由 state slot 决定，不是 KV。** 每个请求要 4 个 slot，每个 74.8 MB，即约 300 MB/请求；而 4.4K token 的 fp8 KV 只要约 141 MB/请求。
3. **draft 权重是最大的额外开销（4–6 GB）**；DSpark/DFlash2 的 verify 中间状态（1.1–1.7 GB）比持久 state 池和 KV 加起来还大。MTP 用 ReplaySSM 把 verify 中间状态压到 0.01 GB。
4. **draft KV 很小**（0.04–0.16 GB）。在 hybrid 模型上，单位 token 的 draft KV 是 target 的 31%（DSpark 10 KiB vs 32 KiB），但因为 KV 池本身被压得很小，绝对值可以忽略。**以 draft KV 为对象的保留策略在这里几乎没有可调的余地。**
5. 两处明显的空间：MTP 初始化后还剩 **5 GB** 没用；target 的 prefill CUDA graph 在所有配置里都要占 **1.8–2.2 GB**。

## E1 容量调优（`e1-capacity`）

做法：显式指定 `--max-mamba-cache-size`（N 个请求 × 4 个 slot）和 `--max-total-tokens`（N × 4400），必要时用 `--cuda-graph-backend-prefill disabled` 关掉 target 的 prefill CUDA graph。

| 配置 | 结果 | 并发上限 | verify 中间状态 | CUDA graph | 初始化后剩余 |
|---|---|---:|---:|---:|---:|
| plain-nopg | 启动 | 15（不变） | 0 | 0.17 | 1.98 |
| mtp-n8（0.93） | 失败：池子没有预算 | - | - | - | - |
| mtp-n6-nopg（0.95） | 失败：需要 2.67 GB，预算约 2.6 GB | - | - | - | - |
| dspark-n4-nopg（0.88） | 失败：池子没有预算 | - | - | - | - |
| **dspark-n2-nopg（0.92）** | 启动 | **2** | 1.72 | 0.29 | 2.31 |
| dflash2-n4-nopg（0.91） | 失败：池子没有预算 | - | - | - | - |
| **dflash2-n3-nopg（0.93）** | 启动 | **3** | 2.29 | 0.30 | 1.08 |

- **池子的预算 = `mem_fraction_static` × 加载前可用显存 − 所有权重**。MTP 的权重有 25.98 GB（target 20.14 + MTP 头 5.84），即使 0.95 也只剩约 2.6 GB 给所有池。"初始化后剩余"不在这个预算里，是留给激活和 CUDA graph 的。
- **verify 中间状态随并发线性增长**：每多一个请求多约 0.57 GB（= 8 个 state slot × 74.8 MB）。DSpark/DFlash2 每个请求总计约 0.9–1.0 GB（verify 0.57 + state 0.3 + KV），plain 约 0.44 GB。
- 关掉 prefill graph 能省约 1.9 GB，但只换来多 1 个并发。
- 失败时的报错建议"把 mem-fraction 调到 0.865 以上"，而实际已经是 0.93–0.95。这条提示只算了权重，没有算显式指定的 mamba 池，有误导性（可以作为一个小 issue 反馈上游）。

**结论：在 32GB 卡上跑 27B hybrid，DSpark/DFlash2 最多 2–3 个并发，plain 是 15 个。主要限制是 verify 中间状态**，也就是 ReplaySSM 在 MTP 上消除掉的那部分；但 DSpark/DFlash 在 GDN 上目前用不了 ReplaySSM（队友已验证）。

## E2 负载扫描（`e2-v1`，单次运行）

配置：plain（cookbook 配方，并发上限 15）、mtp-n4-nopg（上限 4）、dspark-n2-nopg（上限 2）、dflash2-n3-nopg（上限 3；W2 下 KV 只有 7.2K token，实际只能跑 2 个）。

| 负载 | 并发 | plain tok/s | MTP | DSpark | DFlash2 |
|---|---:|---:|---:|---:|---:|
| W2 | 1 | 68 | 1.56× | 1.75× | 1.72× |
| W2 | 2 | 113 | 1.41× | 1.49× | 1.13× |
| W2 | 4 | 209 | 0.90× | 0.83× | 0.62× |
| W2 | 8 | 325 | 0.60× | 0.53× | 0.40× |
| W2 | 16 | 470 | **0.44×** | **0.37×** | **0.26×** |
| w1-natural | 1 | 73 | 1.70× | 2.06× | 2.08× |
| w1-natural | 2 | 127 | 1.89× | 1.99× | 2.01× |
| w1-natural | 4 | 249 | 1.49× | 1.01× | 1.40× |
| w1-natural | 8 | 401 | 0.91× | 0.63× | 0.87× |
| w1-natural | 16 | 560 | 0.66× | 0.46× | 0.64× |

- 低并发时推测解码收益明显：TPOT 5.5–9 ms，plain 是 13.5–19.6 ms；接受长度 MTP 约 2.6–2.9，DSpark/DFlash2 约 3.2–3.8。
- **W2 在并发 4 左右出现拐点，w1-natural 在并发 4–8 之间。** 拐点就是各配置的并发上限，超过之后吞吐不再增长（MTP 约 200/370，DSpark 约 173/255，DFlash2 约 125/350 tok/s），多余的请求全部排队，TTFT 达到 8–16 秒。
- **W2 的前缀命中率**：plain 0.60–0.72，推测解码只有 0.16–0.32。推测配置的 KV 池（7.2K–17.6K token）只能装下 2–4 个 4K 前缀。

## 发现：MTP 有约 5 GB 显存从未被使用

启动顺序（server log + `eagle_worker_v2.py`）：
1. 加载 MTP 头，包括它自带的一份 bf16 embedding 和 lm_head，各 2.54 GB（vocab 248,320 × hidden 5120）。日志记为"draft 权重 5.84 GB"。
2. **按此划分 KV 池和 state 池**，可用显存降到 2.85 GB。
3. `EagleDraftWorker.__init__` 里的 `init_lm_head()`（`eagle_worker_v2.py:238`）把这两份替换成与 target 共用（`set_embed_and_head`），释放约 5 GB。

此时池子已经分好，这 5 GB 只能闲置（初始化后剩余 5.02 GB，plain 只有 0.20 GB）。MTP 每个请求约 0.44 GB，**如果划分池子时就把这 5 GB 算进去，MTP 的并发上限可以从 4 提到约 15，与 plain 持平**。这是第二阶段最便宜的一个改动：把共享 embedding/head 的步骤提到划分池子之前，或者在估算 draft 权重时扣掉会被共享的部分。（推算，尚未验证。）
