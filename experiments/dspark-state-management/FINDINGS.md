# DSpark 状态管理：当前结论与下一步

本报告汇总本机固定容量GPU实验、checkpoint探针及CPU回载验证。历史性能数据固定于 `baa9befa38`，27B CPU实验使用后来提交为 `00fd51eab3` 的Mamba layer-first扩展。原始版本、模型revisions、配置和状态见各报告及JSON/CSV，后续诊断不混入原性能表。

## 缓存压力和优化空间已经观测到

| 对照 | 观测 | 解释范围 |
|---|---|---|
| 8B KV8K→16K | 六会话后续命中0%→96.08%；burst约496→866 tok/s | 人工池容量对照，不能直接当作某种状态策略收益 |
| 27B KV8K→16K，固定12-state | 六会话后续命中仍0%；burst约77.5→78.4 tok/s | 扩KV不能解除checkpoint约束 |
| 27B 固定KV16K，state12→8 | 共享前缀热命中96%→0%；热TTFT约88→603ms | persistent状态减少约0.30GiB，verify intermediate约1.72GiB不变 |
| checkpoint访问顺序 | 8-state ABAB复访全miss，ABBA最近B可hit；12-state均可hit | 单路径能命中，多路径访问引入checkpoint保留压力；没有逐个记录LRU victim |

单条输入会生成多个checkpoint，active、tracking和cached状态共享预算；8-state不等于保留8个完整会话。checkpoint位置与数量应分别研究。GPU-only四类负载1329请求及独立checkpoint164请求的有限输出比较当时没有分叉；这不覆盖后来新增的连续复访3136边界。

详细结果见 [GPU分析](gpu-workloads/ANALYSIS.md) 和 [checkpoint探针](gpu-workloads/CHECKPOINT_RESULTS.md)。队友的自动容量、graph/overlap开启结果提供不同部署条件下的证据，见 [复核](TEAMMATE_REVIEW.md)，不能直接比较绝对吞吐。

## 现有恢复能力与缺口

GPU radix命中可以直接复用有效前缀；HiCache可以保存/恢复target KV、Qwen DSpark draft sidecar和recurrent checkpoint，不能概括为只有完整重新prefill。组件有独立管理，但当前Qwen draft sidecar仍跟随target KV生命周期，不是独立draft-only保留策略。

没有找到持久context-hidden-only pool、readiness和调度路径。已有projected hidden→context KV计算原语，不等于已实现对应缓存。8B用BF16 projected hidden替代BF16 draft KV的理论两侧历史payload节省约7.3%；27B BF16 projected hidden与当前FP8 draft KV均为10KiB/token，直接替换不省payload。只释放逻辑引用也不等于物理显存返还。

HiCache不会自动卸载draft权重、graph或verify workspace，也不会自动扩大target静态池。后续策略要测实际可回收字节、恢复成本和服务指标，不能把命中率改善当作整机收益。

## CPU传输已打通，27B正确性仍未通过

本机WSL2环境中，`cudaHostRegister`内存配合`cudaMemcpyBatchAsync`可提交成功、同步时报700；独立CUDA Runtime程序不加载SGLang/PyTorch也可复现。改逐段`cudaMemcpyAsync`或改`cudaMallocHost`分配均通过。default stream的成功可能是InvalidValue后fallback，不能误认为batch路径通过。具体驱动内部原因未确认，不能推广到所有WSL/CUDA13机器。

8B原生direct/layer_first完成驱逐和target+draft回载，有限输出检查通过。27B的Mamba入口原本拒绝layer_first，原生page_first_direct/direct小测试也失败，因此新增最小支持扩展。补丁的6项GPU回归和实际27B几何逐字节L2往返均通过。

27B完整模型压力后加载3136个KV tokens和2个Mamba slots，291,766,272 bytes与target+draft+state总量吻合，但严格输出检查失败。补充两轮对照：

| 访问阶段 | HiCache | 关闭HiCache |
|---|---|---|
| cold | 0命中，输出A | 0命中，输出A |
| 第一次GPU热复访 | 3072命中，输出A | 3072命中，输出A |
| 第二次GPU热复访 | 3136命中，输出B | 3136命中，输出B |
| 压力后复访 | 3136 host命中，输出C | miss，重算输出A |

A/B/C不同，两轮均重复。GPU-only也分叉，说明不能把现象归因于新增LF传输；同边界B/C仍不同，逐字节拷贝通过也不能证明选择了正确版本的状态。状态边界/所有权与量化数值路径差异尚未区分。`DIAGNOSTIC_COMPLETE`仅表示诊断完成，不是正确性通过。

详细配置、字节算术、失败记录与证据索引见 [CPU_OFFLOAD.md](CPU_OFFLOAD.md)。目前8B可沿用已验证baseline；27B尚不能作为正确性已确认的性能baseline。

## 下一步的验证顺序

先对同一前缀位置的KV、conv/SSM及draft状态与从头计算参考做对照，并查看首处分叉logits，区分状态错误和数值差异。澄清此问题之后，再比较CPU回载与重算成本、checkpoint保留策略及独立draft状态策略。当前没有这些策略的收益数据，也没有重构通用infra。
