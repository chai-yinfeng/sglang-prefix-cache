# Policy 第五轮：第四轮之后的四项（2026-10-28）

第四轮找到了 policy 真正起作用的两个场景（DFlash 运行时 verify 宽度；hybrid 模型按宽度划分显存）。这一轮按收益 / 代价排序，把剩下能做的四项做完：

| 编号 | 内容 | 状态 |
|---|---|---|
| m0 | 第四轮三组时变测试各重复三次 | 完成，存在第四轮目录（`../policy-round4-v0.5.22/w4-repeats.csv`）：同一配置三次总耗时最多差 0.65 s（< 0.5%） |
| m1 | (S+D)/S 状态池比例从手动参数变成 policy 的代码 | 完成，见下 |
| m2 | 同一批里混对话和校对请求：整批统一宽度损失多少 | 进行中 |
| m3 | 开环（泊松到达）负载下的延迟：TTFT、TPOT、E2E p50 / p95 | 进行中 |
| m4 | 窄宽度的 target verify CUDA graph | 代码已提交，测试进行中 |

- 环境同第四轮：Qwen3-8B BF16 + `z-lab/Qwen3-8B-DFlash-b16`，RTX 5090，flashinfer，CUDA graph 和 overlap 开启，capture 后定池大小；hybrid 是 Qwen3.5-4B + `z-lab/Qwen3.5-4B-DFlash`。除注明外单次运行。
- 原始数据：`work_dirs/stress-qwen3-8b/runs/r5-*`；脚本 `phase_bench.py`（新增 `mix` 阶段）、`openloop_bench.py`、`summarize_r5.py`。

## m1：policy 在启动时按 verify 状态划分 hybrid 显存

第四轮发现：hybrid 模型上，每个运行中的请求除了 S 个状态槽（`extra_buffer` + overlap 时 S = 5），推测时还要 D 个 verify 中间状态。SGLang 的配置器已经把中间状态算进状态池（main state 和 intermediate 联合求解），但状态池占总预算的比例固定是 `--mamba-full-memory-ratio`（默认 0.9），和 D 无关，所以 D 越大能跑的请求越少。第四轮手动把比例乘以 (S + D) / S 拿到了大部分收益。

这一轮把这条规则写进 policy 层：开 `--speculative-policy memory`（或包含它的 `conditional`）时，配置器在启动时把状态池比例乘以 (S + D) / S（commit `4de9017eb0`）：

- 规则本身在 `spec_policy/memory_split.py`（纯函数），配置器在推测需要中间状态时调用它，并打印 `Speculative policy memory scales the mamba state:KV memory ratio 0.90 -> 3.78 for 16 verify states per request.`；
- 叠加在用户给的比例上：用户的比例描述的是持久状态和 KV 的平衡，policy 只补上推测那一部分；`--max-mamba-cache-size` 仍然直接决定状态池；
- 其他 policy（`width`、`priority` 等）和不开 policy 时行为不变；单元测试覆盖这两种情况；
- 运行中 `memory` policy 的 KV 扩池对 hybrid 池是空操作（只支持普通 MHA 池），不会误动作。

Qwen3.5-4B，w1-natural，吞吐（tok/s），括号里是请求上限（`m1-hybrid-memory-policy.csv`；前三列来自第四轮）：

| 配置 | 默认划分 | 第四轮手动比例 | **memory policy** | 第四轮手算划分 |
|---|---|---|---|---|
| DFlash b16，c16 / c64 | 969 / 1000（4） | 1267 / 1368（8） | **1280 / 1364**（8） | 1278 / 1421（9） |
| DFlash b4，c16 / c64 | 1640 / 1918（12） | 1773 / 2248（16） | **1790 / 2245**（16） | 1758 / 2345（20） |
| MTP（D = 4），c16 / c64 | 1482 / 1741（12） | - | **1599 / 1998**（16） | 1622 / 2330（24） |

- policy 和第四轮手动设的比例结果一致（同一个比例 3.78 / 1.62，差别在单次运行误差内），说明规则已经由 policy 自动执行。
- c64 相对默认划分：b16 +36%，b4 +17%，MTP +15%；占手算划分收益的 86% / 77% / 44%。c16 时 b16、b4 已经达到或超过手算划分。
- MTP 拿到的比例最低，原因不在推测部分：手算划分按"每个请求约 2K token KV"给了 24 个请求，而默认 0.9 的持久状态 / KV 平衡对这种短上下文负载本来就偏向 KV（第四轮 plain 用手算划分也涨了 29%）。这部分取决于负载的上下文长度，规则故意不去猜。
- 这是 policy 层第一个在启动时决定显存划分、并且有明显收益的真实代码路径（第二、三轮的显存执行器只在运行中扩 KV 池）。
