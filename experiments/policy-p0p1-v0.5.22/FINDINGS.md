# Policy 层 P0 / P1（2026-10-27）

设计见第一阶段后续调查（`../phase1-followups-v0.5.22/FINDINGS.md`），目标是一个按负载切换行为的条件 policy：一个控制器，三个执行器（显存划分、按优先级分配 verify 预算、前缀分层）。

- 机器：Qwen3-8B BF16，单卡 RTX 5090，开发分支 `lvelf/policy-layer`，基于 v0.5.22 `baa9befa38`。
- 负载：w1-natural（ShareGPT 真实对话，套 chat template，不忽略 EOS），每档请求数 max(32, 4×并发)，单次运行。
- 原始数据在 benchmark worktree 的 `work_dirs/stress-qwen3-8b/runs/{p0-observe,p1-fa4,p1-sps-fa4}/`。

| 文件 | 内容 |
|---|---|
| `w1-natural-sweep.csv` | P1 三方对比、P0 observe 模式，以及作对照的第一阶段 `e2-rep2` 数据 |
| `sps-table-8b-rtx5090-fa4.json` | DSpark 步时表（FA4），用 `sglang.benchmark.dspark_sps_profiler` 测得，compact 模式使用 |
| `sps-table-8b-rtx5090-flashinfer.json` | 同上，FlashInfer 版本，只作对照 |

## P0：policy 层骨架

代码在 `python/sglang/srt/managers/scheduler_components/spec_policy/`：

| 文件 | 内容 |
|---|---|
| `types.py` | `Observation`、`PoolUsage`、`SpecPolicyPlan`，都是 frozen msgspec 结构 |
| `sensor.py` | 每步把 batch 结果和池用量整理成观测：decode 步间隔和接受长度的指数滑动平均、排队数、KV / Mamba 用量 |
| `policy.py` | `SpecPolicy` 接口；`ObservePolicy` 只观察、不改动 |
| `actuator.py` | 只在 plan 变化时调用引擎现有的旋钮，目前是 DSpark 的 `set_dspark_forced_budget_frac` |
| `controller.py` | 每处理完一个 batch 结果，依次做观测、决策、执行，可选写 JSONL trace |

- 开关：`--speculative-policy {none,observe}`，以及 `--speculative-policy-trace-path`。
- Scheduler 只增加了 `maybe_init_spec_policy()`，并在 `process_batch_result` 末尾每步调用一次。
- 单元测试：`test/registered/unit/managers/scheduler_components/test_spec_policy.py`（CPU）。

冒烟测试（DSpark，FlashInfer，observe 模式），开销在单次运行的噪声范围内：

| 并发 | observe 模式 | 第一阶段 DSpark（`e2-rep2`） |
|---|---|---|
| 16 | 1423 tok/s | 1450 tok/s |
| 64 | 1962 tok/s | 1956 tok/s |

trace 和独立测量对得上：并发 64（batch ≥ 48）时，中位数每步 57 ms（Q5 用 profiler 测的是 56.7 ms），接受长度 2.8。

## P1：DSpark 现成的置信度 verify 预算（compact ragged-verify）

SGLang 已经给 DSpark 实现了"按接受概率分配 verify 名额"：
- 开关：`SGLANG_RAGGED_VERIFY_MODE=compact`；
- 接受概率：confidence head 给出的前缀累积生存概率；
- 预算：每步由步时表选出吞吐最大的值。

这个功能默认关闭。在 RTX 5090 上还必须配 `--attention-backend fa4`，因为 FlashInfer 不能在 CUDA graph 里做 ragged verify。

**步时表**：
- 测法：profiler 要求服务器以 static 模式启动，并设置 `SGLANG_DSPARK_ENABLE_SPS_RECORD=1` 和 `SGLANG_SIMULATE_ACC_LEN=1.0`。
- 结果：FA4 下，每步 8 个 token 时每秒 71.9 步，64 个时 69.8，128 个时 58.9，256 个时 34.8，512 个时 19.2。转折点在 64–128 token/步之间。

**三方对比**（都用 FA4，单位 tok/s）：

| 并发 | plain | static DSpark | compact DSpark | compact / plain | compact / static |
|---|---|---|---|---|---|
| 1 | 100 | 185 | 183 | 1.83 | 0.99 |
| 4 | 335 | 625 | 589 | 1.76 | 0.94 |
| 16 | 847 | 1460 | 1352 | 1.60 | 0.93 |
| 32 | 1249 | 1747 | 1882 | 1.51 | 1.08 |
| 64 | 2105 | 2088 | 2543 | 1.21 | 1.22 |

| 并发 | p99 TPOT：static | p99 TPOT：compact |
|---|---|---|
| 16 | 21 ms | 68 ms |
| 32 | 50 ms | 70 ms |
| 64 | 100 ms | 101 ms |

- **高并发的计算损失被消除了**：并发 64 时，static DSpark 和 plain 打平，compact 快 21%。按位置的接受概率筛选 verify 名额，验证了 Q5 的判断。
- **低并发时 compact 略亏 1–7%**：batch 小时 verify 本来几乎免费，砍掉的位置仍有接受的可能。
- **并发 16 和 32 时 compact 的 p99 TPOT 明显变差**，原因待查。推测是置信度低的请求长期只拿到 bonus token。
- FA4 和 FlashInfer 的 static 结果接近：plain 在并发 64 时分别是 2105 和 2117 tok/s。

## 下一步

1. 按负载切换：batch × D 低于转折点时全量 verify，高于时交给 compact 规划器，加迟滞。
2. 查清 p99 TPOT 变差的原因，并用请求权重或保底 verify 长度保护尾延迟。
