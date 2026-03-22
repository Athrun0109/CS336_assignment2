# Project: CS336 Assignment 2 - Systems and Parallelism

Stanford CS336 Spring 2025 课程作业2，主题为单GPU训练优化与多GPU并行。

## 项目结构

- `cs336_systems/` — 本作业的代码目录（Python包）
- `cs336-basics/` — Assignment 1 的模型代码，作为 editable 依赖安装
- `tests/` — 测试用例，通过 `tests/adapters.py` 连接实现代码
- 作业要求文档：`cs336_spring2025_assignment2_systems.pdf`

## 环境配置

- 使用 Anaconda 环境 `cs336`，Python 3.11
- PyTorch GPU 版本需通过指定源安装：`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- `cs336-basics` 需以 editable 模式安装：`pip install -e ./cs336-basics`
- 项目本身也需安装：`pip install -e .`

## 导入方式

- 包名用下划线不用连字符：`from cs336_basics.model import BasicsTransformerLM`
- 优化器：`from cs336_basics.optimizer import AdamW`

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

- [x] 1.1.3 benchmarking_script — 已完成，位于 `cs336_systems/benchmarking_script.py`
- [ ] 1.1.4 Nsight Systems Profiler (nsys_profile)
- [ ] Flash Attention 2 Triton kernel
- [ ] Distributed data parallel training
- [ ] Optimizer state sharding

## 关键注意事项

- `BasicsTransformerLM.forward()` 输入是整数 token IDs，用 `torch.randint` 生成，不是 `torch.randn`
- Benchmarking 需要 `torch.cuda.synchronize()` 确保 GPU 计算完成再计时（加 `if torch.cuda.is_available()` 判断以兼容 CPU 环境）
- Benchmark warm-up（GPU预热）与 learning rate warmup 是不同概念
- NVTX 标注用于 nsys profiling，可通过 monkey patching 替换原函数而不修改原代码
