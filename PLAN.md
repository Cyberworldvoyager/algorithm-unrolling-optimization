# 算法展开求解连续优化问题 — 深度学习大作业

## Context

本项目探索**算法展开 (Algorithm Unrolling)** 技术，将经典迭代优化算法展开为可学习的神经网络，应用于三类常见连续优化问题。目标是兼具理论理解和实验验证。

## 一、项目结构

```
project/
├── README.md                    # 项目说明
├── PROGRESS.md                  # 进度记录 (Markdown)
├── PLAN.md                      # 项目计划 (本文件)
├── requirements.txt             # 依赖
├── report/                      # 报告
│   ├── main.tex                 #   LaTeX 版本
│   └── main.md                  #   Markdown 版本
├── common/                      # 公共工具
│   ├── __init__.py
│   ├── utils.py                 # 通用工具函数
│   └── metrics.py               # 评估指标
├── lasso/                       # LASSO 稀疏编码 — ISTA/FISTA 展开 (LISTA)
│   ├── problem.py               # 问题定义、数据生成
│   ├── classical.py             # 经典 ISTA/FISTA 求解器
│   ├── lista.py                 # LISTA 网络实现
│   ├── train.py                 # 训练脚本
│   └── experiment.ipynb         # 对比实验 + 可视化
├── low_rank/                    # 低秩矩阵恢复 — ADMM 展开
│   ├── problem.py               # 矩阵补全问题定义
│   ├── classical.py             # 经典 ADMM 求解器
│   ├── admm_net.py              # 展开的 ADMM-Net
│   ├── train.py
│   └── experiment.ipynb
├── qp/                          # 二次规划 — 投影梯度法展开
│   ├── problem.py               # QP 问题定义 (含 box/simplex 约束)
│   ├── classical.py             # 经典投影梯度下降
│   ├── pgd_net.py               # 展开的 PGD-Net
│   ├── train.py
│   └── experiment.ipynb
└── notebooks/
    └── overview.ipynb            # 总览/汇总可视化
```

## 二、三个问题的展开方案

### 问题 1：LASSO — ISTA/FISTA 展开 (LISTA)

**优化问题**: `min_x  1/2 ||Ax - b||² + λ||x||₁`

**经典算法**: ISTA 迭代公式
```
x_{k+1} = SoftThreshold(W₁b + W₂x_k, θ)
```
其中 `W₁ = I - ηAᵀA`, `W₂ = I - ηAᵀA`, `η` 为步长, `θ = ηλ`

**展开方案 (LISTA)**:
- 将 T 次 ISTA 迭代展开为 T 层网络
- 每层的 `W₁, W₂, θ` 作为可学习参数（而非固定为 A 的函数）
- 激活函数: 软阈值 (soft thresholding)，可学习阈值

**实验**:
- 合成数据: 随机 A ∈ R^{m×n} (m << n), 稀疏 x
- 对比: ISTA vs FISTA vs LISTA（收敛速度、重建精度）
- 可视化: 收敛曲线、支撑集恢复、阈值参数演化

### 问题 2：低秩矩阵恢复 — ADMM 展开 (ADMM-Net)

**优化问题**: `min_{X,Z}  ||X||_*  s.t.  P_Ω(X) = P_Ω(Z), Z = X`
(矩阵补全：从部分观测恢复低秩矩阵)

**经典 ADMM 迭代**:
```
X_{k+1} = D_τ(Z_k - Y_k/ρ)        # 奇异值阈值化
Z_{k+1} = P_Ω(M) + P_Ωᶜ(X_{k+1} + Y_k/ρ)  # 投影
Y_{k+1} = Y_k + ρ(X_{k+1} - Z_{k+1})        # 对偶更新
```

**展开方案 (ADMM-Net)**:
- 将 T 步 ADMM 展开为 T 层
- 学习参数: 阈值 τ_k（每层不同）、惩罚参数 ρ_k
- 用可学习的软阈值/近端算子替代固定奇异值阈值化

**实验**:
- MovieLens / 合成低秩矩阵
- 对比: 经典 ADMM vs ADMM-Net（不同观测比例）
- 可视化: 恢复误差曲线、奇异值分布对比

### 问题 3：二次规划 — 投影梯度法展开 (PGD-Net)

**优化问题**: `min_x  1/2 xᵀQx + cᵀx  s.t.  x ∈ C`
(C 为 box 约束或单纯形约束)

**经典 PGD 迭代**:
```
x_{k+1} = Proj_C(x_k - η_k(Qx_k + c))
```

**展开方案 (PGD-Net)**:
- 将 T 步 PGD 展开为 T 层
- 学习参数: 每层步长 η_k, 可选: 学习 Q 的修正
- 投影算子保持不动（保证可行性）

**实验**:
- 随机生成 QP 实例（保证 Q ≻ 0）
- 对比: 经典 PGD vs PGD-Net（固定迭代次数下的精度）
- 可视化: 目标函数值下降曲线、约束满足情况

## 三、技术实现要点

| 组件 | 实现方式 |
|------|---------|
| 深度学习框架 | PyTorch |
| 软阈值激活 | 自定义 `torch.autograd.Function` |
| 奇异值阈值化 | `torch.svd_lowrank` + 可微阈值 |
| 投影算子 | Box: `clamp`; Simplex: 排序投影 |
| 训练 | Adam optimizer, 学习率调度 |
| 数据 | 合成数据为主 + 1-2个真实数据集 |

## 四、报告结构 (约 15-20 页)

1. **引言** — 算法展开的动机和背景 (2页)
2. **预备知识** — 近端算子、软阈值、ADMM 简述 (2页)
3. **展开方法** — 三个问题的展开方案推导 (4页)
4. **实验设置** — 数据、超参、评估指标 (1页)
5. **实验结果与分析** — 三个问题的对比实验 (5页)
6. **消融实验** — 层数影响、参数初始化影响、泛化性 (2页)
7. **结论与讨论** (1页)

报告同时输出 **LaTeX** (`report/main.tex`) 和 **Markdown** (`report/main.md`) 两种格式。

## 五、实施步骤

1. **Step 1**: 搭建项目骨架和公共工具 (`common/`)
2. **Step 2**: 实现 LASSO 问题 — 经典求解器 + LISTA + 训练 + 实验
3. **Step 3**: 实现低秩矩阵恢复 — 经典 ADMM + ADMM-Net + 实验
4. **Step 4**: 实现 QP — 经典 PGD + PGD-Net + 实验
5. **Step 5**: 消融实验（层数、初始化、泛化到不同问题规模）
6. **Step 6**: 编写报告（LaTeX + Markdown 双版本）

## 六、验证方式

- 每个子问题的 `experiment.ipynb` 包含完整对比实验和可视化
- 确保经典求解器在足够迭代次数下收敛到已知最优值（作为 baseline 正确性验证）
- 展开网络在**少于**经典算法迭代次数时达到同等精度（核心卖点）
- 消融实验展示层数、初始化策略的影响
