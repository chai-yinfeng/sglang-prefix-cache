# CPU offload 定位记录（2026-10-27）

诊断基线 `baa9befa381511cd4d56f83980951e5a45fffc6d`；27B的layer-first支持扩展已提交为 `00fd51eab3`，分支 `fix/hicache-mamba-layer-first`。未改变驱动或依赖。环境：RTX5090、WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`、PyTorch2.14.1+cu130。每次GPU测试前检查共享GPU、RAM/swap，并在27B运行中监测外部计算PID与主机内存。

## 27B最新结论：传输已打通，端到端输出检查未通过

原生MambaPoolHost的入口只允许page_first/page_first_direct；CPU构造测试确认layer_first被拒绝，不需要先加载模型。原生 `test_captured_mamba_state_round_trip[dense-page_first_direct-direct]` 也在D2H备份后报CUDA700，说明本机不能用该替代配置绕开问题。错误后的destroy另有cudaGetErrorString的TypeError，属次生清理异常，本轮未修改。

用户确认后，补丁只启用现有layer-first分配/传输分支，补充空temporal tensor跳过，扩展原生回归并更新文档。空张量保护来自conv-only回归发现的递归，未改变默认布局或batch-copy实现。

- GPU round-trip：direct/kernel × dense/strided/conv-only，共6项通过，包括extra side state。
- CPU相关unit：11项及3个subtests通过。
- 27B实际几何不加载权重的L2TransferEngine往返：16层FP8 target KV、5层FP8 draft KV、48层BF16 GDN state，D2H/H2D逐字节一致。单state slot为78,446,592 bytes。
- 针对4个改动文件的pre-commit通过。Mintlify CLI未安装，broken-links/validate命令返回127，未完成文档站构建。

完整模型配置沿用已固定的RadixArk revisions，KV8192、SSM12、max_running2、page64、chunked_prefill512、FP8 KV/BF16 SSM、extra_buffer、正常异步、graph/overlap关闭，另开host ratio6/write_through/direct/layer_first。较大的host容量用于保留anchor的checkpoint，不能视作等RAM性能比较。

每次压力输入6×2176=13056 tokens，实测驱逐8320个KV tokens。anchor复访：device hit0、host hit3136；加载3136 KV tokens和2个Mamba slots，291,766,272 bytes，恰为：

```text
3136 × (32+10) KiB + 2 × 78,446,592 bytes = 291,766,272 bytes
```

这证明target KV、draft sidecar、recurrent状态的传输均发生；两个state slot是tree恢复和per-request CoW的两份传输，不应解释为两段独立上下文。

`hybrid-offload/qwen38-k8-s12-direct-lf-baseline-v1`：10个测量请求、无执行错误，字节检查通过，但冷/热/恢复output IDs不一致，严格检查写FAILED，未继续第二轮。

新增两组独立诊断，各2轮、22个测量请求，不算正确性通过：

| trace阶段 | HiCache命中tokens / 输出 | 关闭HiCache命中tokens / 输出 |
|---|---|---|
| cold | 0 / A | 0 / A |
| hot | 3072 device / A | 3072 device / A |
| hot-again（驱逐前） | 3136 device / B | 3136 device / B |
| 压力后复访 | 3136 host / C | 0 / A |
| 再复访 | 3136 device / C | 3072 device / A |

两轮均重复上述序列；A/B/C彼此不同。GPU-only的B与HiCache驱逐前B一致。关闭HiCache后仍在3136 checkpoint复用时分叉，因此不能把该现象归因于新增LF传输，也不能只凭拷贝逐字节一致宣称端到端正确。状态位置/版本错误和量化数值路径差异尚未区分。HiCache下同一3136边界B/C不同也仍需解释；本轮没有修改checkpoint所有权或match逻辑。

诊断runs为 `hybrid-offload/qwen38-k8-s12-direct-lf-boundary-control-v1`、`...-gpu-only-boundary-v1`，仅有DIAGNOSTIC_COMPLETE。其旧runner的outputs_equal字段只比较cold/hot/restore/restored-hot，额外hot-again必须查看boundary_control；GPU-only的该字段为true不代表所有阶段一致。当前runner已将额外阶段也计入，历史结果未改写。

本轮完整模型合计54测量请求、3次warmup，无执行错误；不能写成54个正确性测试通过。12项冻结源/fixture hash均核对通过，见 `hybrid-offload/audit.json`。27B GPU采样峰值最高29.86GiB，host available最低32.31GiB，未发现外部计算冲突、无资源guard触发。没有运行新四负载性能campaign。

结论：8B原生direct/LF仍可作已验证baseline；27B经本地小补丁可以进行状态offload实验，但完整输出验证未通过，暂不作为已确认正确的性能baseline。下一步应讨论如何对同一前缀位置的KV/state做计算参考对照，并检查首处分叉logits，区分状态错误和数值差异。

## 原生可用路径

8B DSPARK 使用 `--hicache-io-backend direct --hicache-mem-layout layer_first`，其余控制同原 pilot：KV8192、page64、host ratio3、write_through、graph/overlap关闭。

`runs/qwen3-8b-8192-cpu-direct-layer-first-v1/COMPLETE.json`（CUDA_LAUNCH_BLOCKING=1）记录：

- 压力阶段输入12288 tokens，evicted8192 tokens。
- anchor复访 device hit0、host hit1984，实际load-back1984 tokens。
- 传输333185024 bytes /1984 =167936 bytes/token，与 target144KiB + draft20KiB 完全一致。
- 冷/热/恢复 output IDs 一致，9个测量请求无错误，warmup单独保存。

普通异步复测 `runs/qwen3-8b-8192-cpu-direct-layer-first-async-v1/COMPLETE.json` 同样通过，以上驱逐、回载字节数和输出校验完全相同；无CUDA_LAUNCH_BLOCKING、无诊断入口。两次共18个测量请求无错误（warmup另计）。这些是机制和有限输出校验，不是性能结论、所有并发场景的正确性保证或27B hybrid回载验证。

## 默认路径失败的位置

`kernel + page_first` 的完整服务仍在 warmup 出错。launch blocking traceback表面落在 `src_indices[::page_size].contiguous()`，不能据此认为是H2D回载失败。

独立诊断入口 `offload_diagnostic_server.py` 在原生staged backup前后同步并打印输入：第一笔target备份为36层、384 tokens，src page starts=[64,128,192,256,320,384]，host page starts=[0,64,128,192,256,320]。调用前全GPU同步与索引读取成功，调用后同步 illegal access；尚未进入draft备份。该入口改变时序，只用于定位，源文件冻结在 `kernel-page-first-sync-diag-v1` run，不用于吞吐/延迟比较。

## 最小复现

`offload_transfer_probe.py` 复用仓库原生 `test_hicache_page_first_write_back.py::_run_mha`，备份后逐页比较host、清空device、回载后逐页比较device。每个错误案例单独进程。

| 几何/路径 | 结果 |
|---|---|
| registered，page16/layers2/row512，default stream | 通过 |
| registered，page64/layers36/row1024，default stream，1页/112页 | 通过 |
| registered，模型页几何，6页，default stream，launch blocking | 通过 |
| registered，模型页几何，6页，nondefault stream，launch blocking | D2H后同步700 |
| 同上，显式TVM FFI stream context | 仍失败 |
| registered，小页几何，nondefault stream | 通过 |
| pinned allocator，模型页几何，nondefault stream | 通过 |

default stream通过不能证明batch-copy通过：CUDA13本机头文件要求 `cudaMemcpyBatchAsync` 不能使用legacy NULL stream；原生wrapper遇InvalidValue会改走逐段 `cudaMemcpyAsync`。下面独立CUDA复现核对了这一点。

当前 `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` 默认256且未覆盖；112页大host测试未触发多段注册，不能记录为多注册验证。未执行TMA关闭的对照。

## 不依赖SGLang/PyTorch的CUDA Runtime对照

`cpu-offload-diagnostics/batch_copy_probe.cpp` 仅使用CUDA Runtime；54MiB device和host、12段4.5MiB拷贝。源数据cudaMemset后先同步，再提交拷贝，排除了模型kernel、状态管理和生产者依赖。成功路径检查全部host字节。编译：

```bash
g++ -std=c++17 -O2 work_dirs/dspark-state-management/cpu-offload-diagnostics/batch_copy_probe.cpp \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64 -lcudart \
  -o work_dirs/dspark-state-management/cpu-offload-diagnostics/batch_copy_probe
```

| API / host分配 / stream | 提交 | 同步及数据 |
|---|---|---|
| batch / mmap+cudaHostRegister / NULL | InvalidValue(1) | 未提交 |
| batch / mmap+cudaHostRegister / nonblocking | Success(0) | IllegalAddress(700) |
| cudaMemcpyAsync逐段 / mmap+cudaHostRegister / nonblocking | Success(0) | 通过，逐字节一致 |
| batch / cudaMallocHost / nonblocking | Success(0) | 通过，逐字节一致 |

日志为 `cpu-offload-diagnostics/standalone-*.log`。native pinned allocator对照也通过，与此结果吻合。

已证实：当前机器的 registered host memory + batch-copy 组合可在框架外复现故障，不能归因于DSpark缓存联合生命周期。尚未证实具体驱动内部原因，也不能泛化为所有WSL/CUDA13机器必然失败。

Compute Sanitizer全kernel尝试报 `Failed to initialize WDDM debugger interface` 和 `Device not supported`；因此没有取得有效的无错/首次访存报告。`service-memcheck.log` 为首次参数错误的工具输出，未启动模型；v1/v2为后续尝试日志，不能作为可靠memcheck通过证据。

## 修复取舍

1. 8B可用原生direct/layer_first配置；27B还需00fd51eab3的Mamba布局支持扩展，不需要升级环境。它使用逐段拷贝，可作为后续CPU生命周期实验起点；不能预先认为性能与kernel/page_first相同。
2. 若保留kernel/page_first：考虑对已确认有问题的平台/host分配组合禁用batch-copy并复用已有 `cudaMemcpyAsync` fallback，或改用内置pinned分配。前者保留现有host allocator，增加提交开销；后者改变allocator所有权、销毁和storage backend契约，范围更大。默认batch-copy路径尚未修复。
3. 不能等待700发生再fallback：CUDA context已经被污染，必须在提交前选择安全路径。不能以增加全局同步作为修复。
4. 27B联合回载已执行，传输机制通过、输出检查未通过，详情见顶部；8B结论不能替代27B正确性。
