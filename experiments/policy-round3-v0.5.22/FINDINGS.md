# Policy 第三轮：① d 运行时关推测并回收显存（2026-10-27）

第二轮（`../policy-round2-v0.5.22/`）留下的 ① d 拆成三步，按顺序做，每步单独提交：

| 步骤 | 内容 | 状态 |
|---|---|---|
| s1 | KV 张量的虚拟地址预留超出启动大小（`SGLANG_POST_CAPTURE_KV_RESERVE_FACTOR`） | 完成，见下文 |
| s2 | draft 权重放进可释放的 VMM 区域，释放后地址不变（`speculative/draft_weight_offload.py`） | 完成（单元测试：释放后 graph 重放结果逐位一致），尚未接入服务 |
| s3 | 运行时跳过 draft，只 verify 每个请求的根 token | 进行中 |

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
