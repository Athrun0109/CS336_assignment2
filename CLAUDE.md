# Project: CS336 Assignment 2 - Systems and Parallelism

Stanford CS336 Spring 2025 课程作业2，主题为单GPU训练优化与多GPU并行。

## 项目结构

- `cs336_systems/` — 本作业的代码目录（Python包）
- `cs336-basics/` — Assignment 1 的模型代码，作为 editable 依赖安装
- `tests/` — 测试用例，通过 `tests/adapters.py` 连接实现代码
- 作业要求文档：`cs336_spring2025_assignment2_systems.pdf`
- `env_setup_linux.md` — Linux 环境配置指南

## 环境配置

- 使用 Anaconda 环境 `cs336`，Python 3.11
- PyTorch GPU 版本需通过指定源安装：`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- `cs336-basics` 需以 editable 模式安装：`pip install -e ./cs336-basics`
- 项目本身也需安装：`pip install -e .`
- Linux 上还需额外安装 `pip install triton`（Triton 仅支持 Linux）
- Windows 和 Linux 双设备开发，GPU 为 RTX PRO 5000（48GB 显存）

## 导入方式

- 包名用下划线不用连字符：`from cs336_basics.model import BasicsTransformerLM`
- 优化器：`from cs336_basics.optimizer import AdamW`
- 自定义 softmax：`from cs336_basics.nn_utils import softmax`

## 模型配置（Table 1）

所有模型 vocab_size=10000, batch_size=4：

| Size   | d_model | d_ff  | num_layers | num_heads |
|--------|---------|-------|------------|-----------|
| small  | 768     | 3072  | 12         | 12        |
| medium | 1024    | 4096  | 24         | 16        |
| large  | 1280    | 5120  | 36         | 20        |
| xl     | 1600    | 6400  | 48         | 25        |
| 2.7B   | 2560    | 10240 | 32         | 32        |

## 作业进度

### 1.1 Profiling and Benchmarking（Windows 上完成）
- [x] 1.1.3 benchmarking_script — `cs336_systems/benchmarking_script.py`
- [x] 1.1.4 nsys_profile — 使用 Nsight Systems 完成 profiling，通过 `nsys stats result.nsys-rep` 查看统计
- [x] 1.1.5 benchmarking_mixed_precision — 在 benchmarking_script.py 中添加了 `--mixed_precision` 参数支持 bf16
- [x] 1.1.6 memory_profiling — `cs336_systems/benchmarking_script_memory.py`，2.7B 模型显存分析

### 1.2 Benchmarking PyTorch Attention（Windows 上完成）
- [x] 1.2.1 pytorch_attention — `cs336_systems/pytorch_attention_benchmark.py`
  - 48GB 显存跑通所有配置（含 seq_len=16384），无 OOM
  - seq_len=16384 时耗时异常高（显存接近上限导致）

### 1.3 JIT-Compiled Attention（需要 Linux）
- [ ] 1.3.0 torch_compile — `cs336_systems/pytorch_attention_benchmark_compiled.py`（Windows 上因缺少 Triton 无法运行）
- [ ] 1.3.1 Weighted Sum（Triton kernel 示例）
- [ ] 1.3.2 FlashAttention-2 Forward（Triton 实现）
- [ ] 1.3.3 FlashAttention-2 Backward
- [ ] 1.3.4 Benchmarking FlashAttention

### 2. Distributed Data Parallel Training（需要 Linux）
- [ ] 2.1 Distributed Communication
- [ ] 2.2 Naive DDP
- [ ] 2.3 Improved DDP

## 关键注意事项

- `BasicsTransformerLM.forward()` 输入是整数 token IDs，用 `torch.randint` 生成，不是 `torch.randn`
- Benchmarking 需要 `torch.cuda.synchronize()` 确保 GPU 计算完成再计时
- Benchmark warm-up（GPU预热）与 learning rate warmup 是不同概念，前者是让 GPU 做 JIT 编译/内存分配等初始化
- NVTX 标注用于 nsys profiling，可通过 monkey patching 替换原函数而不修改原代码：`cs336_basics.model.scaled_dot_product_attention = annotated_version`
- `torch.autocast` 参数是 `device_type="cuda"`（字符串），不是 `device=torch.device(...)`
- Memory profiling 时 `_record_memory_history` 应在 warmup 之后启动，避免记录初始化开销
- 从 1.3 开始的所有作业（torch.compile、Triton、Distributed）都需要在 Linux 上完成
- `torch.compile` 在 Windows 上会报 TritonMissing 错误，需要 Linux + Triton
- Q, K, V 用 `torch.randn` 创建时需加 `requires_grad=True` 才能 backward
- Attention 的显存瓶颈在于 seq_len×seq_len 的 attention score 矩阵，FlashAttention 通过分块计算避免显式构建
