# 算法展开求解 LASSO 问题 — 深度学习大作业

## 项目概述

本项目探索**算法展开 (Algorithm Unrolling)** 技术，将经典 ISTA 算法展开为可学习的神经网络，聚焦于 LASSO 稀疏编码问题。

### 核心创新

1. **LISTA-Momentum**: 提出带可学习动量的 LISTA 变体，解决泛化性问题
2. **维度无关 LISTA**: 设计维度无关架构，可处理任意维度输入
3. **LSTM-LISTA v2**: 使用逐元素 LSTM 建模优化过程，实现最佳性能

### 主要结果 (训练样本=1000, T=10, n=200)

| 排名 | 方法 | 类别 | 无噪声 | 有噪声 | vs ISTA | 泛化退化 |
|-----|------|------|--------|--------|---------|---------|
| 1 | **LSTM-LISTA** | 序列模型 | **0.023** | **0.030** | **36.6×/28.1×** | 1.22×/1.12× |
| 2 | RNN-LISTA | 序列模型 | 0.027 | 0.033 | 31.2×/25.5× | **1.11×/1.22×** |
| 3 | DA-LISTA | 展开网络 | 0.110 | 0.110 | 7.7×/7.7× | 1.23×/1.11× |
| 4 | Transformer-LISTA | 序列模型 | 0.211 | 0.209 | 4.0×/4.0× | 1.06×/1.09× |
| 5 | FISTA | 经典基线 | 0.819 | 0.819 | 1.03×/1.03× | 1.00×/1.00× |
| 6 | ISTA | 经典基线 | 0.842 | 0.842 | 1.00×/1.00× | 1.00×/1.00× |
| 7 | LISTA-CP | 展开网络 | 0.971 | 0.964 | 0.87×/0.87× | 1.03×/1.04× |
| 8 | LISTA | 展开网络 | 1.161 | 1.003 | 0.73×/0.84× | 0.86×/1.00× |

## 项目结构

```
project/
├── README.md                          # 本文件
├── requirements.txt                   # Python 依赖
├── .gitignore                         # Git 忽略规则
├── run_noiseless_experiments.py       # 主实验脚本 (无噪声 vs 有噪声)
├── noiseless_experiment_results.json  # 实验结果
├── report/
│   └── main.md                        # 实验报告
├── common/                            # 公共工具
│   ├── __init__.py
│   ├── metrics.py                     #   评估指标 (relative_error 等)
│   └── numerical.py                   #   数值计算 (proximal_l1 等)
└── lasso/                             # LASSO 问题实现
    ├── __init__.py
    ├── problem.py                     #   问题定义与数据生成
    ├── classical.py                   #   经典算法 (ISTA/FISTA)
    ├── lista.py                       #   LISTA/ LISTA-CP/ LISTA-CP-FISTA
    ├── lista_universal.py             #   LISTA-Momentum (通用架构)
    ├── lista_dimension_agnostic.py    #   维度无关 LISTA
    └── lista_sequence_v2.py           #   LSTM-LISTA v2
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

```bash
# 运行完整实验 (无噪声 vs 有噪声，所有模型)
python run_noiseless_experiments.py
```

### 3. 查看报告

```bash
cat report/new_report.md
```

## 依赖

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

## 参考文献

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
2. Chen, X., et al. (2018). Theoretical linear convergence of unfolded ISTA. NeurIPS.
3. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. SIAM.
4. Monga, V., Li, Y., & Eldar, Y. C. (2021). Algorithm unrolling: Interpretable, efficient deep learning. IEEE SPM.

## 许可证

本项目仅用于学术研究和教育目的。
