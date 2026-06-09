# 算法展开求解 LASSO 问题 — 深度学习大作业

## 项目概述

本项目探索**算法展开 (Algorithm Unrolling)** 技术，将经典 ISTA 算法展开为可学习的神经网络，聚焦于 LASSO 稀疏编码问题：

$$\min_x \tfrac{1}{2}\|Ax - b\|_2^2 + \lambda\|x\|_1$$

系统对比了 **9 种方法**（经典基线 / 展开网络 / 维度无关架构 / 序列模型），覆盖精度、不同展开层数、ID/OOD 泛化、**跨维度泛化**四个维度，并在无噪声与有噪声两种条件下各跑一遍，所有学习型模型结果在多随机种子下取均值±标准差。

**问题规模**：m=100, n=200, k=5 (well-posed 区, m/n=0.5)，展开网络采用 ISTA 等价初始化 (η=1/L)，训练含验证集早停。

### 方法清单

| 类别 | 方法 | 实现位置 |
|------|------|---------|
| 经典基线 | ISTA, FISTA | `lasso/classical.py` |
| 展开网络 | LISTA, LISTA-CP | `lasso/lista.py` |
| 展开网络 | LISTA-Momentum | `lasso/lista_momentum.py` |
| 维度无关架构 | DA-LISTA | `lasso/lista_momentum.py` |
| 序列模型 | RNN-LISTA, LSTM-LISTA, Transformer-LISTA | `lasso/sequence.py` |

> 完整实验结果以 `all_experiment_results.json` 与 `report/new_report.md` 为准。

## 项目结构

```
project/
├── README.md                       # 本文件
├── requirements.txt                # Python 依赖
├── .gitignore
├── run_all_experiments.py          # 主实验脚本 (唯一入口)
├── all_experiment_results.json     # 实验结果 (运行后生成)
├── report/
│   ├── new_report.md               # 最终实验报告
│   ├── training_comparison.png     # 全模型对比 / 层数影响
│   ├── cross_dimension.png         # 跨维度泛化
│   └── W_matrix_analysis.png       # LISTA-Momentum 的 W 矩阵分析
├── common/                         # 公共工具
│   ├── metrics.py                  #   评估指标 (relative_error 等)
│   └── numerical.py                #   数值工具 (proximal_l1 等)
└── lasso/                          # 模型实现 (single source of truth)
    ├── problem.py                  #   LASSO 问题定义与数据生成
    ├── classical.py                #   ISTA / FISTA
    ├── lista.py                    #   LISTA / LISTA-CP (+ softplus 阈值)
    ├── lista_momentum.py           #   LISTA-Momentum / DA-LISTA
    └── sequence.py                 #   RNN / LSTM / Transformer-LISTA
```

`run_all_experiments.py` 中的所有模型均从 `lasso` 包导入，脚本本身不重复定义模型，避免“提交代码 ≠ 运行代码”。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

```bash
python run_all_experiments.py
```

脚本会自动选择 GPU（若 `torch.cuda.is_available()`）否则 CPU，依次完成：全模型对比 (T=10, m=100, n=200, k=5)、不同 T∈{5,10,20}、ID/OOD 泛化、跨维度泛化 (训练 n∈{100,150,200}，测试 n∈{50,100,200,300,400})、W 矩阵谱分析，并将结果写入 `all_experiment_results.json`、图表写入 `report/`。

训练配置：1000 样本、50 轮、Adam lr=1e-3、验证集早停 patience=10、3 种子。展开网络 (LISTA/CP/Momentum) 采用 ISTA 等价初始化。

> **注**：在 Anaconda + MKL 环境下如遇 `OMP: Error #15`（libiomp 重复初始化），请设置环境变量 `KMP_DUPLICATE_LIB_OK=TRUE` 后再运行。

### 3. 查看报告

```bash
cat report/new_report.md
```

## 依赖

- Python >= 3.9
- PyTorch >= 2.0（GPU 实验需 CUDA 版本，如 `torch==2.5.1+cu124`）
- NumPy, Matplotlib

## 参考文献

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
2. Chen, X., et al. (2018). Theoretical linear convergence of unfolded ISTA. NeurIPS.
3. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. SIAM J. Imaging Sci.
4. Monga, V., Li, Y., & Eldar, Y. C. (2021). Algorithm unrolling: Interpretable, efficient deep learning. IEEE SPM.

## 许可证

本项目仅用于学术研究和教育目的。
