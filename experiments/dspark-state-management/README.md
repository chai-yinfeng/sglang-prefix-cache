# DSpark 状态管理调查与容量校准

源码基线：`baa9befa381511cd4d56f83980951e5a45fffc6d`（v0.5.22）。
本目录归档调查、测试结果和分析摘要。历史性能实验未改runtime；27B CPU实验使用本地Mamba layer-first扩展00fd51eab3。原始harness和逐请求产物仍位于work_dirs/dspark-state-management，不随摘要上传。

先读 [FINDINGS.md](FINDINGS.md) 和 [PROGRESS.md](PROGRESS.md)；特别注意27B端到端输出检查未通过。

## 范围

- 主模型：Qwen3-8B + DeepSeek 官方 DSpark；Qwen3.8-27B NVFP4 + RadixArk DSpark。
- 补充：社区 Qwen3.5-4B SWA draft，须先核对 Speculators 格式、权重名、RoPE 和 capture 层契约。
- 目标：固定实际 GPU 状态字节预算，测量状态保留、驱逐、恢复的容量和延迟取舍。
- 不把 UnifiedRadixCache（树和生命周期）与 `--enable-unified-memory`（物理多池字节共享）混为一谈。后者当前只支持 hybrid target，不能直接用于纯 FA Qwen3-8B。

## 源码核对

### 恢复不是只能完整重新 prefill

1. GPU radix 命中：直接复用有效前缀，只执行尚未命中的 suffix。完整 miss 才需要从头 prefill。
2. CPU HiCache 命中：scheduler 调用 `init_load_back`，加载 host KV，再缩短 prefill 范围。Qwen DSpark 使用 SIDECAR；target KV 与 draft context KV 在同一回载操作中恢复。
3. recurrent target：Mamba component 还有 host checkpoint 回载、per-request 状态准备和恢复。不代表任意 token 边界都能从 checkpoint 恢复。
4. PD decode retraction 另有 CPU tensor / host_pool backup-resume。该分支依赖 disaggregation 模式，不是普通单机 retract 的通用保证；hybrid recurrent 不支持该 host_pool retraction 时会使用 CPU tensor 路径。
5. 没有 host 副本、没有 hidden 表示的区间：未发现从 target KV 直接恢复 DSpark 中间特征的路径。按剩余有效前缀重新计算 target，并注入 draft context KV。

默认 Qwen draft sidecar 随 target KV 的索引、backup 和 load 一起行动；它不是独立的 draft-only host cache。未找到现成的“target 留 GPU，单独把 draft KV 搬 CPU，命中后只恢复 draft”的 readiness/admission 路径。不能将已有双侧回载描述成已经支持独立保留策略。

源码入口（相对仓库根）：

- `python/sglang/srt/speculative/base_spec_worker.py`：`_can_pack_hicache_mtp`、`_build_hicache_draft_plan`。Qwen DSpark 选择 SIDECAR，DeepSeekV4 DSpark 才是专用 PACKED 分支。
- `python/sglang/srt/mem_cache/kv_cache_builder.py`：`maybe_register_hicache_draft`。
- `python/sglang/srt/mem_cache/hybrid_cache/hybrid_pool_assembler.py`：`build_full_draft_pools`；DRAFT sidecar 的 indices_from_pool=KV。SWA sidecar 另走 TRAILING_PAGES，不能套用 full draft 结论。
- `python/sglang/srt/mem_cache/unified_radix_cache.py`：`init_load_back`、`load_back`、`_load_back_transfers`、`_build_sidecar_transfers`。
- `python/sglang/srt/managers/schedule_policy.py`：admission 调用 `init_load_back` 并调整 prefix/extend 长度。
- `python/sglang/srt/mem_cache/unified_cache/components/mamba.py`：`prepare_load_back`、`finalize_load_back`。
- `python/sglang/srt/managers/schedule_batch.py`：`load_kv_cache`、`release_req`，PD decode retraction backup 的作用域。

### 没有找到 hidden-only 持久缓存路径

- `dspark_worker_v2.py::_forward_prefill` 捕获 target aux hidden，立即 `inject_target_hidden`，然后将 `logits_output.hidden_states` 和 token indices 设为 None。
- decode commit 将已接受的 target hidden 注入 draft pool；graph hidden 缓冲会被后续 forward 覆盖，不是历史 hidden cache。
- `models/dflash.py::project_target_hidden` 已提供 concat hidden -> projected/normed context hidden。
- `models/dspark.py::write_target_hidden_kv` 和 `dspark_kv_inject.py::inject_target_hidden` 接受 projected hidden，可跳过重复 projector。这是可复用的计算原语，不是现成的持久化、匹配、驱逐和恢复功能。
- `mem_cache` 中未找到 Qwen DSpark context hidden pool/host mirror、radix 关联、readiness 或 hidden-only 配置。
- 不能据此声称全仓库没有任何 hidden buffer；PD transfer、API return_hidden_states、graph 暂存等用途不同。

因此：现有 CPU offload 路径保存的是 KV/SSM 等原生状态；“只存 projected hidden，不存 context KV”仍需新增状态所有权、位置/模型身份、ready 边界和重建调度。

### 内存计量

Qwen3-8B BF16 的理论 payload：target KV 144 KiB/token，5 层 draft context KV 20 KiB/token，原始五层 aux hidden 40 KiB/token，projected context hidden 8 KiB/token。
projected hidden 替代 draft KV 节约 12 KiB/token（两侧历史总 payload 约减少 7.3%）。这是表示开销，不含 allocator、页 padding、capture/verify/graph workspace。
独立驱逐若只释放逻辑引用而不归还物理页，不能计作可用显存收益。

## 容量校准 pilot

`pilot.py` 复用上一轮隔离环境、tokenizer/corpus 和 streaming client。无权重下载、无 runtime monkey patch。

- GPU 原生 KV 和 CPU HiCache 两种配置；初始建议 4096 / 8192 / 16384 slots。
- 两侧 BF16 payload 分别为 0.640625 / 1.28125 / 2.5625 GiB，按实际启动日志和 server_info 交叉核对。
- 这是原生 GPU-only 与 HiCache 的同模型同物理池容量对照；不将该 slot 扫描当作跨模型等显存对照。
- 固定 anchor：冷请求 -> GPU 热命中 -> 插入总长度超过 1.5 倍池容量的不同前缀 -> anchor 再访问。
- 检查 `cached_tokens_details`、evicted tokens、host load tokens/bytes/duration，证明是 GPU miss + CPU load，而非意外 GPU hit。
- host ratio=3；host 容量也必须足以保留 anchor，否则测试退化为双层 miss。
- 记录冷/热/恢复 output IDs 和 acceptance。输出一致仅代表这些 probe；有分叉必须保留，不能自动归因 BF16。
- 三种状态均有完整请求结果、metrics 和启动记录。`FAILED.json` 保留失败，不写成通过。

脚本为每个 run 冻结 manifest、fixtures、runner/client 源码；顶层文件只作为最新一次入口，历史证据以 run 目录为准。
读取 native `/generate` 的 cache report，不添加 OpenAI API 专用参数。单请求 socket 超时为 60 秒。

命令（需要沙箱外 GPU/localhost 权限）：

```bash
work_dirs/qwen3-8b-prefix-baseline-v0.5.22/.venv/bin/python \
  work_dirs/dspark-state-management/pilot.py --capacities 8192 --modes gpu cpu
```

## 四类负载

1. 长前缀重复/分叉：区别 device hit、host hit、完整 miss；恢复行数与正常 suffix 行数分开。
2. 会话轮转/冷热切换：固定 trace；扫描实际工作集/可用容量比 0.5、1、1.5、2；包含多轮稳态而非一次 replay。
3. 长上下文持续生成：扫描 active pinned bytes 增长与可驱逐 inactive bytes；不要用模型权重占比制造状态压力。
4. 突发并发/retract/结束/取消：以实际 admission、state-slot 约束、retract 和回收结果确认边界是否触发。单纯请求数高不代表有压力。

Qwen3.8-27B 需要独立校准：target FA KV、draft context KV、GDN persistent/checkpoint、verify intermediate state、graph/workspace 及 host mirrors。
优先固定 dtype/SSM策略/verify宽度/backend，再用 `max-total-tokens`、`max-mamba-cache-size` 或已有 unified byte pool 控制不同瓶颈；启动配置值不代替实际分配量。
不通过占用无关 GPU 张量模拟预算，不把 `mem-fraction-static` 直接当硬进程显存上限。

机制pilot见 `RESULTS.md`、`runs/`。历史四类GPU负载在 `gpu-workloads/`：两个模型5配置均完成三次重复，共1329测量请求、0错误；另有164个独立checkpoint探针请求，见 `gpu-workloads/CHECKPOINT_RESULTS.md`。CPU offload定位见 [CPU_OFFLOAD.md](CPU_OFFLOAD.md)：8B原生direct/LF通过；27B经 `00fd51eab3` 扩展Mamba LF支持后已完成联合回载，但端到端输出检查失败，GPU-only对照也复现3136 checkpoint分叉，不能宣称正确性通过。默认batch-copy故障未修复。新实验在 `hybrid-offload/`，与历史性能表分开。续接状态见 `PROGRESS.md`。
