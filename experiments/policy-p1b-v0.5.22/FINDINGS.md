# Policy 层 P1b：按负载切换 verify 预算（2026-10-27）

接着 P1（`../policy-p0p1-v0.5.22/FINDINGS.md`）。P1 里 DSpark 的 compact 模式在低并发时比 static 慢 1–7%，在并发 16–32 时 p99 TPOT 变差。P1b 检验两件事：
- 按负载切换（转折点以下全量 verify）能不能补回低并发的那点差距；
- 尾延迟是否真的变差。

- 环境：Qwen3-8B，RTX 5090，FA4，w1-natural，单次运行；
- 原始数据：`work_dirs/stress-qwen3-8b/runs/{p1b,p1b-oracle}/`。

| 文件 | 内容 |
|---|---|
| `p1b-load-aware-sweep.csv` | load-aware policy 的扫描，以及 static / compact 在并发 16、32 下的重跑（带逐请求 TPOT 分位数） |
| `p1b-planner-oracle-c64.csv` | 并发 64 下 DSpark 规划器与固定预算比例的对比 |

## 实现

`LoadAwareVerifyPolicy`（`--speculative-policy load-aware`，代码在 `spec_policy/policy.py`）：
- 一步要 verify 的 token 数（batch × D）低于转折点时，设置 `dspark_verify_budget_frac=1.0`，也就是全量 verify；
- 高于转折点 × 1.5 时，交回 DSpark 的置信度规划器；
- 切换后至少停留 16 个 decode 步。

转折点由步时表推出：取"每秒步数仍不低于峰值 60% 的最大 token 数"，在 FA4 步时表上是 192。60% 是按 P1 测得的交叉点定的，不是推导出来的。

## 结果

| 并发 | static | compact（规划器） | load-aware |
|---|---|---|---|
| 1 | 185 | 183 | 182 |
| 4 | 625 | 589 | 607 |
| 16 | 1460 | 1439 | 1374 |
| 32 | 1739 | **1964** | 1713 |
| 64 | 2088 | 2543 | 2538 |

单位 tok/s。并发 1、4、64 的 static 和 compact 取自 P1，并发 16、32 是 P1b 的重跑。load-aware 在 5164 个 decode 步里只切换了 2 次：4790 步全量 verify，374 步交给规划器。

**load-aware 不比规划器好**：
- 并发 16：load-aware 处于全量 verify，只有 1374，比规划器（1439）还低。也就是说，compact 在低并发时比 static 慢，来源是 compact 路径本身的开销（ragged 布局、规划计算），而不是预算削减。运行时的预算旋钮消除不了这部分开销。
- 并发 32：落在迟滞区里，一直停在全量 verify，比规划器少 13%。

**尾延迟问题没有复现**：重跑时，compact 和 static 的 p99 TPOT 相当（并发 16：23.4 vs 22.0 ms；并发 32：48.7 vs 50.6 ms）。逐请求平均 TPOT 的 p90 在并发 16 时是 11.3 vs 10.9 ms，并发 32 时是 18.3 vs 21.0 ms。P1 里的 68–70 ms 是单个离群请求，不是规划器让低置信度请求长期挨饿。

## 上限检查：规划器 vs 固定预算比例（并发 64）

同一个 compact 服务上，先用规划器跑，再通过 `/set_internal_state` 的 `dspark_force_budget_frac` 依次固定预算比例：

| 预算 | tok/s | 中位 TPOT | 接受长度 |
|---|---|---|---|
| 规划器 | **2532** | 17.4 ms | 2.40 |
| 1.0（全量） | 2064 | 24.1 ms | 2.83 |
| 0.6 | 2346 | 20.1 ms | 2.75 |
| 0.4 | 2367 | 18.4 ms | 2.55 |
| 0.25 | 2296 | 18.1 ms | 2.21 |

规划器比最好的固定比例还高 7%：它每步按置信度在请求之间分配预算，固定比例做不到这一点。

## 结论

- 对 DSpark 来说，SGLang 原生的置信度规划器已经是按负载调整的，而且在测试的范围内接近最优。我们的 policy 层**不应该**再去覆盖它的预算。`load-aware` 这个选项保留在代码里，作为记录在案的负面结果。
- policy 层在 ② 上能做的事情只剩两处：
  1. 请求权重（SLO 等级），规划器目前只看接受概率；
  2. 没有置信度规划器的算法，即 DFlash 和 EAGLE3。
- 低并发时 compact 路径的固定开销（1–6%）只能在启动时选 static 模式来避免，运行时改不了。
