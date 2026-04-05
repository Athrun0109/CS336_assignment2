# CS336 环境配置指南 (Linux)

## 步骤1：创建 conda 环境

```bash
conda create -n cs336 python=3.11 -y
conda activate cs336
```

## 步骤2：安装 PyTorch GPU 版本

根据你的 CUDA 版本选择（RTX PRO 5000 建议 cu128）：

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 步骤3：安装 Triton（Linux 专用，Windows 不支持）

```bash
pip install triton
```

## 步骤4：安装项目依赖

```bash
cd /path/to/assignment2-systems

# 安装 cs336-basics（editable 模式，从项目内的子目录安装）
pip install -e ./cs336-basics

# 安装项目本身
pip install -e .
```

> 注意：`cs336_basics` 不需要单独安装，它在项目的 `cs336-basics/` 子目录中，
> 通过 `pip install -e ./cs336-basics` 以 editable 模式安装即可。
> Linux 上的安装路径会自动适配，不需要和 Windows 保持一致。

## 步骤5：安装其他 pip 依赖

以下是作业和开发中额外用到的包（项目 pyproject.toml 未覆盖的部分）：

```bash
pip install einops einx jaxtyping
pip install psutil submitit tiktoken
pip install pytest regex tqdm wandb
pip install pandas tabulate matplotlib
pip install pymupdf pymupdf4llm
```

## 步骤6：验证安装

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import triton; print('Triton version:', triton.__version__)"
python -c "from cs336_basics.model import BasicsTransformerLM; print('cs336_basics OK')"
```

## 环境概览

| 类别 | 包 | 版本 | 安装方式 |
|------|-----|------|---------|
| Python | python | 3.11 | conda |
| 深度学习 | torch | 2.10.0+cu128 | pip (PyTorch 源) |
| 深度学习 | torchvision | 0.25.0+cu128 | pip (PyTorch 源) |
| 深度学习 | torchaudio | 2.2.0 | pip (PyTorch 源) |
| GPU 编程 | triton | (latest) | pip |
| 张量操作 | einops | 0.8.1 | pip |
| 张量操作 | einx | 0.4.2 | pip |
| 类型标注 | jaxtyping | 0.3.9 | pip |
| 数学 | numpy | 1.26.4 | conda/pip |
| 数学 | sympy | 1.14.0 | pip |
| 分词 | tiktoken | 0.12.0 | pip |
| 正则 | regex | 2025.11.3 | pip |
| 测试 | pytest | 9.0.2 | pip |
| 进度条 | tqdm | 4.67.3 | pip |
| 实验追踪 | wandb | 0.23.1 | pip |
| 数据处理 | pandas | (latest) | pip |
| 表格 | tabulate | 0.10.0 | pip |
| 绘图 | matplotlib | (latest) | pip |
| 任务提交 | submitit | 1.5.4 | pip |
| 系统 | psutil | 7.2.2 | pip |
| PDF | pymupdf / pymupdf4llm | 1.27.2.2 | pip |
| 项目代码 | cs336-basics | 1.0.3 | pip -e ./cs336-basics |
| 项目代码 | cs336-systems | (本项目) | pip -e . |
