# Policy 第三轮：① d 运行时关推测并回收显存（2026-10-27）

第二轮（`../policy-round2-v0.5.22/`）留下的 ① d 拆成三步，按顺序做，每步单独提交：

| 步骤 | 内容 | 状态 |
|---|---|---|
| s1 | KV 张量的虚拟地址预留超出启动大小（`SGLANG_POST_CAPTURE_KV_RESERVE_FACTOR`） | 完成，见下文 |
| s2 | draft 权重放进可释放的 VMM 区域，释放后地址不变（`speculative/draft_weight_offload.py`） | 完成（单元测试：释放后 graph 重放结果逐位一致），尚未接入服务 |
| s3 | 运行时暂停推测（跳过 draft，只 verify 每个请求的根 token）+ 按实测吞吐切换的档位策略 `--speculative-policy regime` | 完成，见下文 |
| s4 | 暂停时释放 draft 权重、把显存给 KV，恢复前收缩 KV | 不做：上界实验显示在本机任何负载上都赢不了规划器，见下文 |

- 环境：Qwen3-8B，RTX 5090，FA4，page 64；DSpark 使用 compact 规划器，开启 capture 后定池大小和 HiCache；单次运行。
- 原始数据：`work_dirs/stress-qwen3-8b/runs/s1-va-fix/`。

## s1：预留 2 倍虚拟地址后，扩池不再停在启动大小

配置同第二轮 R3（`dspark-full`，`--mem-fraction-static 0.85`，`--speculative-policy memory`），另外设置 `SGLANG_POST_CAPTURE_KV_RESERVE_FACTOR=2.0`：target 和 draft 两个 KV 池各预留 104,576 个 token 的虚拟地址，启动时只映射 40,128 个。

**第一次运行在扩池后触发了 device-side assert。** 原因：KV 写入 kernel（`store_cache`）的越界检查上限 `size + page_size` 在 CUDA graph capture 时作为整数参数固定下来。池子扩到启动大小以上后，graph 里的写入仍按旧上限检查，新分配的槽位全部越界。修复：MHA 池的写入上限改为整个预留范围（`_slot_bound()`，三处：`set_kv_buffer` 的越界检测、`store_cache` 的 `size_limit`、`move_kv_cache`）。新增的单元测试在扩池前 capture 一个写 KV 的 graph，扩池后让它写到新范围的末尾；修复前这个测试复现了同样的 assert，修复后通过。DSpark 的 `fused_kv_norm_rope_write` 不带槽位上限，不受影响。

修复后的结果（W2，`s1-va-reserve.csv`）：

| 并发 | 吞吐（tok/s） | TTFT 中位数 | TPOT 中位数 | KV 容量 |
|---|---|---|---|---|
| 16 | 784 | 0.39 s | 12.6 ms | 40,128 |
| 32 | 980 | 2.16 s | 13.5 ms | 40,128 → 56,960 |
| 64 | **1378** | 3.23 s | 16.0 ms | 56,960 |

- 扩池发生在并发 32 快结束时：40,128 → 56,960，越过了第二轮的上限 52,288，扩后剩余 1.43 GB 空闲显存。之后的窗口里，最低空闲显存扣掉 1 GiB 余量后不够 2,048 个 token，所以没有再扩。
- 并发 64：1378 tok/s。第二轮上限为 52,288 时是 1347（+2.3%，在单次运行的噪声范围内），只观察不扩是 1144（+20%）。
- 服务没有中断，没有 OOM。

这一步本身收益很小：0.85 的 fraction 下，空闲显存本来就只够再扩约 1.2 万个 token。它的作用是给 s2 / s3 铺路：关推测后释放的 draft 权重（2.1 GB，约 1.3 万个 token）只有在预留范围足够大时才能变成 KV 容量。

## s3：运行时暂停推测

### 实现

- **暂停路径**（`DSparkWorkerV2.set_dspark_spec_paused`）：decode 时不跑 draft 前向，用 `bonus_only_proposal` 构造一个"所有 draft 都等于 bonus token"的提议，用 `bonus_only_layout` 让每个请求只 verify 根 token（compact ragged verify，长度 1）。这一步本质上就是一次 target decode，仍走已 capture 的 CUDA graph。draft KV 照常从 target hidden 注入，所以恢复推测后 draft 的上下文是完整的。只支持 compact 模式（需要预算规划器）。
- **运行时开关**：`/set_internal_state {"dspark_spec_paused": true}`。policy 的 plan 新增字段 `spec_paused`，由执行器调用上面的开关。
- **档位策略** `--speculative-policy regime`（`SpecRegimePolicy`）：
  - 不做模型预测，直接测：对当前模式维护一个 64 步的 decode 吞吐窗口（提交的 token 数 ÷ decode 步耗时）；
  - 决策过期（1024 步，每次探测失败翻倍，上限 16384）或平均 batch 变化超过 25% 时，切到另一模式跑一个窗口，快 3% 以上才保留，否则切回；
  - batch < 24 时不暂停；窗口内实测接受长度 ≥ 1.8 时也不探测暂停（依据见下文的步耗时比）；
  - `conditional` 现在是 regime + priority + memory 三者组合。

### 结果（并发 64，`s3-pause.csv`）

| 负载 | 规划器 | 全程暂停 | plain（同日） | regime（加门槛） |
|---|---|---|---|---|
| W2 | 1399 | 922 | 1506 | 1418 |
| w1-natural | 2469 | 1845 | 2039 | 2472 |
| w1-natural，温度 1.0 | 2385 | 1807 | 2026 | 2367 |

单位 tok/s。

1. **在这三个负载上，暂停都更慢**，比规划器慢 24–34%，也比 plain 慢 9–39%：
   - w1-natural：暂停时每步 ITL 16.6 ms，plain 是 15.5 ms。DSpark 路径每步多 7%（aux hidden capture、draft KV 注入、按 8 个槽位分配 verify 窗口）。
   - W2：暂停时每步耗时和 plain 差不多（20.3 vs 20.9 ms），但吞吐只有 922。原因是显存：一个请求完成要多跑约 2.8 倍的步数，占着 KV 的时间也长约 2.8 倍，新请求排队，TTFT 中位数从 3.0 s 升到 4.8 s。**显存紧张时，推测解码加快 KV 周转，这是它在 W2 并发 64 上仍然领先暂停的主要原因。**
   - 温度 1.0 没有降低 DSpark 的接受长度（2.36 vs 2.38）：DSpark 的 draft 也按同一温度采样。
2. **步耗时比**：规划器每步约 26–28 ms（ITL × 接受长度），暂停时每步 16.6–20.3 ms，比值 1.3–1.7。所以只有接受长度低于约 1.7 时暂停才可能赢。这就是 regime 探测门槛 1.8 的来源。
3. **不加门槛的 regime**（第一版）在三个负载上各探测一次暂停，三次都输，都切回了推测。决策是对的，但每次探测都有代价：W2 1367、w1 2436、w1-t1 2339，比规划器低 1–2%。**加门槛后**一次都没探测，吞吐与规划器持平（+1.3% / +0.1% / −0.8%，在单次运行的噪声范围内）。
4. **低接受长度时 regime 会暂停**：用 `SGLANG_SIMULATE_ACC_LEN=1.0` 把接受长度固定为 1（模拟 draft 完全失效），w1-natural 并发 64：
   - 规划器 1266 tok/s；
   - regime 1707 tok/s（**+35%**）。它在 58.6% 的 decode 步上暂停，探测两次都赢了；没暂停的步主要在开头（第一个窗口还在测推测）和结尾（batch 降到 24 以下就恢复推测）。

### 正确性

6 个 greedy prompt，各生成 160 个 token，比较四种输出：推测全程开、全程暂停、生成中途暂停再恢复、推测全程开重跑一次。

- 直接比文本：暂停有 3/6 在中途分叉；推测全程开的重跑本身也有 1/6 分叉（并发 batch 的数值噪声）。
- 用 teacher-forced 检查（`s3-teacher-forced.csv`）：在 plain target 上把 prompt + 输出整体 prefill 一次，看每个生成的 token 是不是 target 的 top-1。
  - 四种输出各 960 个 token 中，非 top-1 的有 7–10 个，最大差距 0.125–0.25 nats，都是 bf16 下的近似并列（差距 0 表示完全并列）。
  - 推测全程开的输出也一样（8 个，最大 0.25）。
  - 所以暂停没有引入错误的 token，分叉来自不同 verify 宽度下的数值差异。

## s4：为什么不做释放 draft 权重和 KV 收缩

设想是：暂停时把 draft 权重（2.25 GB，s2 已经能在地址不变的前提下释放）交给 KV，恢复推测前再收缩 KV（停止从高位页分配，淘汰或等待占着高位页的请求，再 unmap）。实现量大：分配器的 drain、radix 树的定向淘汰、调度器容量视图和泄漏检查的同步、过期 draft KV 的请求降权。

做之前先测上界。释放权重后，暂停的 DSpark 最多有 61K + 13.7K ≈ 75K 个 KV token。用 plain target 把 KV 限制在 75K（`--max-total-tokens 75000`），在最吃显存的 W2 并发 64 上跑：**1386 tok/s，低于规划器的 1399**。暂停的 DSpark 每步还比 plain 多约 7%，所以即使 s4 完全实现，暂停 + 回收也赢不了规划器。在 w1 上显存不是瓶颈（各方案同时运行的请求数都在 82–88），回收显存帮不上忙。

结论：在 Qwen3-8B + DSpark + RTX 5090 上，① d 的"关推测"只在 draft 几乎失效时才有收益，s3 的 regime 策略已经覆盖这种情况；显存回收这一半没有正收益，所以不实现。已写的 KV 收缩原语（VMM 区间 unmap、按增长边界 decommit、占位页映射）没有提交，保存在 `MyDocs/Plan/s4-kv-shrink-primitives.patch`。对 DFlash 这类高并发下接受长度只有 1.4 左右的 drafter，这个判断可能不同，但 DFlash 先要有 ragged verify 才能支持暂停。
