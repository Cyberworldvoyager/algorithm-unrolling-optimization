# 算法展开求解连续优化问题 — 深度学习大作业

## 项目概述

本项目探索**算法展开 (Algorithm Unrolling)** 技术，将经典迭代优化算法展开为可学习的神经网络。

### 核心创新

1. **LISTA-Momentum**: 提出带可学习动量的 LISTA 变体，解决泛化性问题
2. **维度无关 LISTA**: 设计维度无关架构，可处理任意维度输入
3. **理论分析**: 提供谱半径稳定性验证和参数分析

### 主要结果

| 方法 | ID 性能 | OOD 性能 | 泛化性 |
|------|---------|---------|--------|
| ISTA | 0.843 | 0.844 | ✓ |
| LISTA-CP | 0.448 | 1.215 | ✗ |
| **LISTA-Momentum** | **0.526** | **0.555** | ✓ |

| 维度 n | ISTA | LISTA-DimAgnostic | vs ISTA |
|--------|------|-------------------|---------|
| 100 | 0.701 | 0.080 | 8.72× |
| 200 | 0.852 | 0.158 | 5.38× |
| 300 | 0.900 | 0.239 | 3.77× |

## 项目结构

```
project/
├── README.md                          # 本文件
├── requirements.txt                   # 依赖
├── report/                            # 报告
│   ├── main.md                        #   Markdown 版本
│   └── main.tex                       #   LaTeX 版本
├── common/                            # 公共工具
│   ├── __init__.py
│   ├── numerical.py                   #   数值计算
│   ├── visualization.py               #   可视化
│   ├── utils.py                       #   通用工具
│   ├── metrics.py                     #   评估指标
│   ├── theory.py                      #   理论分析
│   └── training.py                    #   训练框架
├── lasso/                             # LASSO 稀疏编码
│   ├── __init__.py
│   ├── problem.py                     #   问题定义
│   ├── classical.py                   #   ISTA/FISTA
│   ├── lista.py                       #   LISTA 变体
│   ├── lista_universal.py             #   LISTA-Momentum
│   ├── lista_dimension_agnostic.py    #   维度无关 LISTA
│   ├── train.py                       #   训练脚本
│   └── experiment.ipynb               #   实验
├── low_rank/                          # 低秩矩阵恢复
│   ├── __init__.py
│   ├── problem.py                     #   问题定义
│   ├── classical.py                   #   ADMM
│   ├── admm_net.py                    #   ADMM-Net
│   ├── train.py
│   └── experiment.ipynb
├── qp/                                # 二次规划
│   ├── __init__.py
│   ├── problem.py                     #   问题定义
│   ├── classical.py                   #   PGD
│   ├── pgd_net.py                     #   PGD-Net
│   ├── train.py
│   └── experiment.ipynb
└── notebooks/                         # 实验
    ├── overview.ipynb                 #   总览
    ├── ablation.ipynb                 #   消融实验
    └── experiments.ipynb              #   综合实验
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 运行实验

**LASSO 实验 (LISTA-Momentum)**:
```bash
cd lasso
python train.py
jupyter notebook experiment.ipynb
```

**维度无关 LISTA 实验**:
```bash
python train_dimension_agnostic_robust.py
```

**低秩矩阵恢复实验**:
```bash
cd low_rank
python train.py
jupyter notebook experiment.ipynb
```

**二次规划实验**:
```bash
cd qp
python train.py
jupyter notebook experiment.ipynb
```

### 3. 查看报告

```bash
cd report
# Markdown 版本
cat main.md

# LaTeX 版本 (需要 LaTeX 编译器)
pdflatex main.tex
```

## 依赖

- Python >= 3.9
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- SciPy >= 1.10.0
- Jupyter >= 1.0.0

## 参考文献

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. ICML.
2. Chen, X., et al. (2018). Theoretical linear convergence of unfolded ISTA. NeurIPS.
3. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm. SIAM.
4. Boyd, S., et al. (2011). Distributed optimization via ADMM. Foundations and Trends in ML.

## 许可证

本项目仅用于学术研究和教育目的。
