# Policy 层 P4：按请求优先级分配 verify 名额，以及向其他算法推广（2026-10-27）

设计里定下的优先级是：**位置的接受概率 × 请求权重**。P1b 已经说明，DSpark 原生规划器在吞吐上接近最优，所以 P4 只补它没有的两样东西：
- 请求权重（P4a）；
- DFlash 和 EAGLE3 的支持（P4b）。

- 环境：Qwen3-8B，RTX 5090，FA4，DSpark compact ragged verify，使用 P1 测的步时表；单次运行；
- 原始数据：`work_dirs/stress-qwen3-8b/runs/p4a/`；汇总表：`p4a-priority-classes.csv`。

## P4a：请求加权的 verify 名额排序

**实现**（`lvelf/policy-layer`）：

| 位置 | 改动 |
|---|---|
| `speculative/dspark_components/dspark_planner.py` | 规划器维护一张按 `req_pool_idx` 索引的权重表；排序前用 `weight_first_position` 把每行 confidence 的第 0 列乘上请求权重 |
| `dspark_worker_v2.py` | 新增 `set_dspark_request_weights`，和 `set_dspark_forced_budget_frac` 是同类运行时旋钮 |
| `spec_policy/` | `--speculative-policy priority`：`priority > 0` 的请求权重为 1，其余为 0.25；执行器只在 slot 被新请求占用或权重变化时写入 |

第 0 列乘以权重后，这一行所有位置的前缀生存概率都会乘上同一个权重，所以 top-k 等于按"生存概率 × 权重"排序，而每个请求内部的位置顺序不变。预算 B 仍然按未加权的生存概率计算，所以规划器对吞吐的估计不受影响。

**负载**：
- 客户端：`priority_bench.py`，闭环，256 个请求，每 4 个里有 1 个 `priority=1`；
- prompt：ShareGPT 第一轮用户消息，套 chat template，输出长度取参考回复的长度；
- 两组服务都开 `--enable-priority-scheduling`，准入顺序相同，只比较 verify 加权的效果。

**结果**（TPOT 单位 ms）：

| 并发 | 配置 | 吞吐（tok/s） | 高优先级 TPOT p50 / p90 | 低优先级 TPOT p50 / p90 |
|---|---|---|---|---|
| 32 | 不加权 | 2803 | 10.08 / 11.97 | 10.05 / 11.92 |
| 32 | 加权 | 2787（−0.6%） | **9.26 / 11.14** | 10.39 / 12.43 |
| 64 | 不加权 | 3472 | 14.73 / 18.92 | 14.84 / 18.07 |
| 64 | 加权 | 3357（−3.3%） | **13.37 / 15.57** | 16.34 / 19.81 |

- 不加权时，两个类别的 TPOT 完全一样，说明规划器本身对请求一视同仁。
- 加权后，高优先级请求的 TPOT p50 下降 8–9%，并发 64 时 p90 下降 18%；低优先级慢 3–10%；总吞吐只少 0.6–3.3%。TTFT 两组都由准入顺序决定，基本一样。
- 效果是中等量级，原因有二：
  - 权重只在规划器已选定的预算内部重新排序，预算本身按吞吐最优来定；
  - 每个请求仍然保底 verify 1 个位置。

  要拉开更大的差距，需要额外给高优先级请求追加预算，或者提高它们的保底长度，代价是吞吐。
- 这次实现过程中出了一个 bug：执行器一开始读的是 `req.req_pool_idx`，实际字段在 `req.kv.req_pool_idx`。单元测试没有发现，因为假对象是照着错误的形状写的。已修正，测试里的假对象改成和真实 `Req` 一致。

## P4b：向 DFlash 和 EAGLE3 推广（只做分析，没有改代码）

| 算法 | 计算侧现有的自适应 | 能不能接入 ② |
|---|---|---|
| DSpark | 置信度规划器（compact ragged verify）+ 步时表；P4a 又加了请求权重 | 已接入 |
| EAGLE / EAGLE3 | `--speculative-adaptive`：按接受率在候选步数之间切换（第一阶段 E4：默认配置在 32 GB 卡上会耗尽显存，缩小候选集合后可用） | 它已有自己的机制；policy 层可以把档位条件接到候选步数上 |
| DFlash | **没有**：block 固定为 16，没有 ragged verify 路径（`cutoff_verify_lens=None`），也没有 confidence head | 需要移植 DSpark 的 compact verify（ragged 布局、按 verify 长度定 graph 档位、无 confidence head 时用在线逐位置接受率估计），工作量大 |

DFlash 的计算侧损失是三者中最大的（第一阶段 Q5：并发 64 时 verify 每请求 16 个 token，只有 plain 的 0.31 倍），也最缺自适应机制。实际部署时，用同一类 drafter、但带 confidence head 和 ragged verify 的 DSpark 替代 DFlash 更现实。
