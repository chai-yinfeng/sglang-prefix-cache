# Policy 层第二轮：显存执行器、分层策略、组合 policy（2026-10-27）

第一轮（`../policy-summary-v0.5.22/`）里，policy 层只做出了骨架和请求权重。显存划分和前缀分层两条主线，当时只是打开了 SGLang 现成的功能做实验，并没有写成 policy。这一轮把它们补成 policy 层的代码：

| 设计中的部分 | 这一轮的实现 | 结果 |
|---|---|---|
| ① a 显存账本 | `spec_policy/ledger.py`：按用途统计字节（target / draft KV 每 token、target / draft 权重，共享存储只计一次），作为 trace 的第一行写出；空闲显存每 32 步采样一次，记入 Observation | 完成 |
| ① 运行时显存划分 | 运行时扩池：`PagedTokenToKVPoolAllocator.grow`、`kv_pool_runtime.grow_post_capture_kv_pool`、`ModelRunner.grow_kv_pool`、`Scheduler.grow_kv_capacity` 加上容量视图刷新；`--speculative-policy memory`（`MemoryHeadroomPolicy`） | W2 并发 64 +18%（见 R3） |
| ③ 前缀分层的价值密度淘汰 | `--radix-eviction-policy value-density`（`ValueDensityStrategy`，Python TreeCore，从 Rust 自动回退） | 只有很小的差异（见 R2） |
| 组合 policy | `--speculative-policy conditional`：请求权重（②）+ 显存余量（①），各自只写自己的 plan 字段 | 完成 |
| ① b decode 预留跟随预算 | 不做：并发 64 时预留 64 × 16 = 1024 token，只占 59K 池子的 1.7%，而且 compact 模式实际仍按完整窗口分配 KV | 测算后放弃 |

- 环境：Qwen3-8B，RTX 5090，FA4，page 64；DSpark 使用 compact 规划器，同时开启 capture 后定池大小和 HiCache；单次运行。
- 原始数据：`work_dirs/stress-qwen3-8b/runs/{r0-specoff,r2-tiering,r3-memory}/`。

## r0：能不能用"预算为 0"代替关闭推测？不能

在 `dspark-full` 上通过 `dspark_force_budget_frac=0.0001` 把 verify 预算压到接近 0，并发 64：

| 负载 | 规划器 | 预算约为 0 | plain-pcs |
|---|---|---|---|
| W2 | 1403 | 794 | 1521 |
| w1-natural | 2462（取自集成实验） | 1465 | 2101 |

单位 tok/s。预算为 0 时，draft 前向照跑，每步只产出 bonus token（接受长度 1.0），远差于 plain。所以条件控制器的"高负载关推测"必须真正跳过 draft 并回收它的显存（① d），不能靠压预算实现。表 `r0-budget-zero.csv` 中 w1-natural 的"planner"一行实际上仍是 0.0001：脚本在两种负载之间没有重置预算。

## R2：W3-mixed 上的价值密度淘汰

负载 W3-mixed（`w3-mixed-prefix.meta.json`）：
- 24 篇文档，1K、4K、12K 三种长度各 8 篇；
- Zipf(α=1) 热度，按随机排列分配，与长度无关；
- 工作集 139K token，输出 128。

`ValueDensityStrategy`：
- 价值 = 复用率 × 再取回代价 ÷ token 数；
- 复用率 = (命中次数 + 1) × 0.5^(闲置时间 / 半衰期)；
- 节点仍在 GPU 上且开了 HiCache 时，再取回代价是 CPU 取回（15 ms + 2.7 µs/token）；否则是该深度上的 prefill 时间差；
- 代价参数用 Q6 的实测值，可以通过 `--radix-eviction-policy-config` 重新标定。

| 淘汰策略 | 树核心 | 并发 16 | 并发 64 | 并发 64 命中率 |
|---|---|---|---|---|
| LRU | Rust | 427 | 486 | 0.741 |
| LRU | Python | 426 | 468 | 0.728 |
| 价值密度 | Python | 420 | 486 | 0.749 |

单位 tok/s。和同一核心上的 LRU 相比，价值密度在并发 64 时吞吐 +3.8%、命中率 +2.1 个百分点，但都在单次运行的噪声范围内。原因是 CPU 层（12 GB）已经接住了大部分被挤出 GPU 的前缀，剩下的未命中主要是每篇文档第一次访问时的冷启动，淘汰顺序影响不到。要看到明显差别，需要 CPU 层也放不下的工作集，或者不开 HiCache。

## R3：运行时扩池（`--speculative-policy memory`）

配置：`dspark-full` 但 `--mem-fraction-static 0.85`，KV 从 40,128 个 token 起步，启动后剩余 4.10 GB。`MemoryHeadroomPolicy` 的做法：
- 600 步窗口内取采样到的最低空闲显存，扣掉 1 GiB 余量，换算成 token 数去扩；
- 扩池时先映射 target 和 draft 两个池的物理页，再给分配器追加新页。

| 负载 | 并发 | 只观察（不扩） | memory（扩） |
|---|---|---|---|
| W2 | 16 | 777 | 777 |
| W2 | 32 | 952 | 975 |
| W2 | 64 | 1144 | **1347（+18%）** |
| w1-natural | 64 | - | 2415（安全性检查） |

单位 tok/s。扩池记录（`r3-growth-events.csv`）：
- 在 W2 并发 32 期间，KV 从 40,128 扩到 52,288，正好碰到预留上限，扩后剩余 2.07 GB；
- 服务没有中断，没有 OOM，没有断言失败；
- 重启后跑自然对话，同样扩到了 52,288，并发 64 吞吐 2415 tok/s（自动 fraction 配置是 2462）。
- 并发 64 的 TTFT 中位数从 5.1 秒降到 3.7 秒。

**这次的上限来自预留的虚拟地址，而不是空闲显存**（扩完还剩 2 GB）。KV 张量的虚拟范围在启动时按 capture 前的大小预留。如果按整卡容量预留虚拟地址（虚拟地址几乎不花钱），扩池就只受物理显存限制。这也是 ① d 的前提：释放 draft 权重和 graph 后，要能扩出原预留之外的空间。

## 仍未完成的部分

| 部分 | 状态 | 卡在哪里 |
|---|---|---|
| ① d 高负载时关推测并回收显存 | 未做 | 需要三样东西：(1) 运行时跳过 draft（现在 decode 准备、`spec_info`、overlap future map、KV 窗口分配、draft KV 注入都按推测模式构造）；(2) 只释放 draft 权重，并在重新加载后保持地址不变（memory saver 只能整类释放，且与 capture 后定池大小互斥）；(3) 预留超出启动大小的虚拟地址。另外重开推测时要收缩 KV 池 |
| ① c hybrid 模型的 verify 中间状态按预算定大小 | 未做 | 需要改 GDN verify kernel 和状态提交的索引方式（按请求 × 位置 → 按 verify token） |
| 负载档位状态机（低 / 中 / 高） | 未单独实现 | 现有执行器本身就随负载起作用：规划器的预算、只在削减时才生效的请求权重、只在显存有余量时才扩池。档位状态机只有在 ① d 这类按档位切换的动作存在时才有意义 |
| ② 推广到 DFlash / EAGLE3 | 只做了分析 | DFlash 需要移植 ragged verify |
