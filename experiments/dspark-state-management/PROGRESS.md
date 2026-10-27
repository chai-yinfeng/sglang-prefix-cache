# DSpark 状态管理进度（2026-10-27）

## 当前交付

代码补丁 `00fd51eab3`，分支 `fix/hicache-mamba-layer-first`：允许Mamba host pool使用已有layer-first分配/传输路径，跳过conv-only的空temporal tensor，增加dense/strided/conv-only的direct/kernel往返回归并更新配置文档。默认布局与batch-copy实现未改，未升级驱动或依赖。

本分支已合入队友 `lvelf/benchmark` / `1af0d7917b` 的原有33个归档文件。此更新补齐本机后续摘要，不把队友数据算作本机独立复现。

## 已完成的测试

| 范围 | 结果与限制 |
|---|---|
| 原四类GPU负载 | 5配置，各3次重复，共1329测量请求，无执行错误；保持原始基线版本，未用新补丁重跑 |
| 独立checkpoint探针 | 8/12-state各82请求，共164，无执行错误；控制case重复两次，数量扫描每档一次三轮 |
| 8B原生direct/LF CPU pilot | blocking及正常异步各一轮，共18测量请求，驱逐/回载及有限输出检查通过 |
| 原生传输与独立CUDA对照 | 本机registered host memory + batch-copy故障在框架外复现；逐段拷贝及pinned分配对照通过 |
| LF补丁回归 | 6项GPU、11项CPU及3个subtests通过；27B实际几何原生L2逐字节往返通过；pre-commit通过 |
| 27B完整模型严格检查 | 10请求无执行错误，KV+draft+state传输字节吻合，但输出一致性失败，写FAILED |
| 27B HiCache/GPU-only边界对照 | 各2轮、各22请求，无执行错误；完成诊断，不能写正确性通过 |

完整27B测试合计54测量请求加3次warmup。GPU-only在3136-token checkpoint复用时也分叉；CPU恢复与同边界GPU复访也不相同，尚未区分状态边界/版本错误和量化数值差异。不能将27B视为已确认正确的性能baseline。

## 阅读入口

- [FINDINGS.md](FINDINGS.md)：主要结论、优化空间、待归因项。
- [CPU_OFFLOAD.md](CPU_OFFLOAD.md)：本机CUDA故障、补丁、27B回载及输出对照。
- [CHECKPOINT_RESULTS.md](gpu-workloads/CHECKPOINT_RESULTS.md)：checkpoint数量、边界与访问顺序探针。
- [原GPU性能分析](gpu-workloads/ANALYSIS.md)、[性能表](gpu-workloads/RESULTS.md)。
- [队友结果复核](TEAMMATE_REVIEW.md)：配置/显存预算差异和结论边界。
- 机器可读摘要：`pilot-summary.json`、`gpu-workloads/{summary.csv,summary.json,checkpoint-summary.json,calibration-diagnostics.json}`、`cpu-offload-diagnostics/summary.json`、`hybrid-offload/{summary.csv,summary.json,audit.json}`。

## 尚未完成

- 27B输出分叉的根因定位与修复；没有修改checkpoint匹配或所有权策略。
- 默认registered+batch-copy路径的修复；当前使用direct/LF。
- Qwen3.5-4B SWA DSpark/Speculators适配；旧4B归档是native DFlash，不能混称。
- hidden-only、独立draft CPU生命周期、自适应权重/graph/verify显存管理；均未实现或测得收益。
- Mintlify CLI未安装，文档站broken-links/validate未完成；代码pre-commit已通过。

## 资源与产物

每次大模型启动前复查共享GPU；27B运行中监测外部计算PID、主机available与swap。未发现外部计算冲突，未触发资源guard。27B采样峰值最高29.86GiB，host available最低32.31GiB。服务已清理，最终GPU353MiB、util0、无计算进程，host available约45GiB。

本目录只归档文档与摘要。冻结harness、输入、逐请求输出、原始日志和metrics留在本机 `work_dirs/dspark-state-management/`，不提交权重、二进制或原始环境日志。
