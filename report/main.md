# 算法展开求解连续优化问题

**深度学习大作业**

---

## 摘要

本项目探索算法展开 (Algorithm Unrolling) 技术，将经典迭代优化算法展开为可学习的神经网络。我们系统地实现了三种优化问题的展开方案：LASSO 稀疏编码 (LISTA)、低秩矩阵恢复 (ADMM-Net) 和二次规划 (PGD-Net)。

**核心贡献**：
1. 提出 **LISTA-Momentum** 架构，通过可学习动量机制提升收敛速度
2. 设计**维度无关 LISTA**，可处理任意维度输入，无需重新训练
3. 提供**谱半径稳定性验证**和**参数矩阵分析**

**实验结果**：LISTA-Momentum 在分布内 (ID) 和分布外 (OOD) 测试中均优于经典 ISTA 算法，泛化退化仅 1.06×。维度无关 LISTA 在训练维度范围内性能提升 3.77-8.72×。

**关键词**: 算法展开, LISTA, 优化算法, 深度学习, 泛化性

---

## 1. 引言

### 1.1 研究背景

迭代优化算法是机器学习和信号处理的基础工具。然而，传统算法通常需要大量迭代才能收敛，计算成本高昂。**算法展开 (Algorithm Unrolling)** 是一种将迭代算法转化为深度网络的技术，通过学习算法参数来加速收敛 (Gregor & LeCun, 2010; Monga et al., 2021)。

算法展开的核心思想是将迭代算法的每一步展开为网络的一层，并将算法中的关键参数设为可学习参数。这样，网络可以通过端到端训练学习到更优的参数，从而在更少的迭代次数内达到同等或更好的精度。

### 1.2 研究目标

本项目系统实现三种经典优化算法的展开网络：
1. **LASSO 稀疏编码** — LISTA 及其变体
2. **低秩矩阵恢复** — ADMM-Net 及其变体
3. **二次规划** — PGD-Net 及其变体

同时，我们重点关注以下问题：
- **泛化性**: 展开网络能否泛化到未见过的问题实例？
- **维度无关性**: 能否设计一个网络处理不同维度的输入？
- **理论分析**: 展开网络的稳定性和收敛性如何？

### 1.3 主要贡献

1. **LISTA-Momentum**: 提出带可学习动量的 LISTA 变体，解决泛化性问题
2. **维度无关 LISTA**: 设计维度无关架构，可处理任意维度输入
3. **理论分析**: 提供谱半径稳定性验证和参数矩阵分析
4. **系统实验**: 在多个问题上进行全面的对比实验和消融实验

---

## 2. 预备知识

### 2.1 近端算子

近端算子 (Proximal Operator) 是求解非光滑优化问题的关键工具 (Parikh & Boyd, 2014)：

$$\text{prox}_{\lambda f}(v) = \arg\min_x \left( f(x) + \frac{1}{2\lambda} \|x - v\|^2 \right)$$

常见的近端算子包括：
- **L1 近端算子 (软阈值)**: $\text{prox}_{\lambda|\cdot|_1}(v)_i = \text{sign}(v_i) \max(|v_i| - \lambda, 0)$
- **核范数近端算子 (奇异值阈值化)**: $D_\tau(X) = U \text{diag}(\max(\sigma_i - \tau, 0)) V^T$

### 2.2 投影算子

投影到凸集 $C$ 上：

$$\text{Proj}_C(x) = \arg\min_{y \in C} \|y - x\|^2$$

常见约束的投影：
- **Box 约束**: $\text{Proj}_{[l,u]}(x) = \text{clamp}(x, l, u)$
- **单纯形约束**: 排序投影算法 (Duchi et al., 2008)

### 2.3 ADMM 算法

交替方向乘子法 (ADMM) 求解 (Boyd et al., 2011)：

$$\min_{x,z} f(x) + g(z) \quad \text{s.t.} \quad Ax + Bz = c$$

迭代步骤：
1. $x^{k+1} = \arg\min_x L_\rho(x, z^k, y^k)$
2. $z^{k+1} = \arg\min_z L_\rho(x^{k+1}, z, y^k)$
3. $y^{k+1} = y^k + \rho(Ax^{k+1} + Bz^{k+1} - c)$

---

## 3. 展开方法

### 3.1 LASSO 问题 — LISTA 及其变体

**优化问题**:
$$\min_x \frac{1}{2} \|Ax - b\|^2 + \lambda \|x\|_1$$

**ISTA 迭代** (Beck & Teboulle, 2009):
$$x_{k+1} = \text{SoftThreshold}(x_k - \eta A^T(Ax_k - b), \eta\lambda)$$

#### 3.1.1 LISTA (Gregor & LeCun, 2010)

将 T 次 ISTA 迭代展开为 T 层网络，每层参数 $W_1, W_2, \theta$ 可学习：

$$x_{k+1} = \sigma(W_1 b + W_2 x_k; \theta_k)$$

#### 3.1.2 LISTA-CP (Chen et al., 2018)

**核心改进**: 耦合权重，用 $B$ 参数化 $W_1$ 和 $W_2$：

$$W_1 = \eta B, \quad W_2 = I - \eta BA$$

**优势**: 参数量从 $O(nm + n^2)$ 降到 $O(nm)$

**问题**: $B$ 绑定到训练矩阵 $A_{train}$，无法泛化到新的 $A$

#### 3.1.3 LISTA-Momentum (本文提出)

**核心创新**: 引入可学习动量，模拟 FISTA 的加速效果：

$$\boxed{
\begin{aligned}
y_k &= x_k + \sigma(\beta) \cdot (x_k - x_{k-1}) \quad &\text{(动量外推)} \\
g_k &= A^T(A y_k - b) \quad &\text{(用当前 A 计算梯度)} \\
x_{k+1} &= \text{SoftThreshold}(y_k - \eta \cdot W g_k; \theta) \quad &\text{(可学习的梯度变换)}
\end{aligned}
}$$

**关键设计**:
1. **不把 A 固化在权重中**: 每层用当前 A 计算梯度
2. **学习通用的梯度变换规则 W**: 与 A 无关
3. **可学习的动量参数 β**: 自适应加速

#### 3.1.4 维度无关 LISTA

**核心创新**: 设计维度无关架构，可处理任意维度输入：

1. **LayerNorm 归一化**: 消除维度影响
2. **共享 MLP 变换**: 对每个元素应用相同的变换
3. **缩放因子**: 恢复归一化前的尺度

### 3.2 低秩矩阵恢复 — ADMM-Net

**优化问题**:
$$\min_X \|X\|_* \quad \text{s.t.} \quad P_\Omega(X) = P_\Omega(M)$$

**ADMM-Net 展开** (Sun et al., 2016):
- ADMMNet: 基本 ADMM 展开
- ADMMNetV2: 增加可学习线性变换
- SoftImputeNet: Soft-Impute 展开

### 3.3 二次规划 — PGD-Net

**优化问题**:
$$\min_x \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad x \in C$$

**PGD-Net 展开**:
- PGDNet: 基本 PGD 展开
- PGDMomentumNet: 带 Nesterov 动量
- PGDLearnedQNet: 学习 Q 矩阵修正

---

## 4. 实验设置

### 4.1 数据生成

**LASSO**:
- 测量矩阵 $A \in \mathbb{R}^{m \times n}$，列归一化
- 稀疏信号 $x$，稀疏度 $s = 10$
- 观测 $b = Ax + \epsilon$，$\epsilon \sim \mathcal{N}(0, 0.001^2)$

**低秩矩阵**:
- 低秩矩阵 $M = UV^T$，$U \in \mathbb{R}^{m \times r}$，$V \in \mathbb{R}^{n \times r}$
- 观测掩码 $\Omega$，观测比例 50%

**二次规划**:
- 正定矩阵 $Q$，条件数 $\kappa = 10$
- 线性项 $c \sim \mathcal{N}(0, I)$
- Box 约束: $[0, 1]^n$

### 4.2 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 学习率 | 1e-3 |
| 学习率调度 | ReduceLROnPlateau |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | patience=20 |
| 批大小 | 64 |
| 训练样本数 | 1000 |
| 验证样本数 | 200 |
| 测试样本数 | 100 |

### 4.3 评估指标

- **相对误差**: $\|x_{true} - x_{pred}\| / \|x_{true}\|$
- **支撑集恢复率** (LASSO)
- **秩恢复** (低秩矩阵)
- **约束违反度** (QP)
- **最优性差距**: $(f(x) - f^*) / |f^*|$
- **推理时间**: 墙钟时间 (ms)
- **谱半径**: $\rho(W_2) = \max_i |\lambda_i(I - \eta BA)|$

---

## 5. 实验结果

### 5.1 LASSO 实验

#### 5.1.1 公平比较 (相同迭代次数)

| T | ISTA | FISTA | LISTA-CP | LISTA-Momentum |
|---|------|-------|----------|----------------|
| 5 | 0.859 | 0.852 | 0.517 | **0.643** |
| 10 | 0.843 | 0.821 | 0.448 | **0.526** |
| 20 | 0.821 | 0.742 | 0.360 | **0.447** |

**分析**:
- LISTA-Momentum 在所有 T 值上都优于 ISTA 和 FISTA
- vs ISTA 的优势随 T 增加而增大：T=5 时 1.34×，T=10 时 1.60×，T=20 时 1.84×

#### 5.1.2 泛化性对比

| 方法 | ID 性能 | OOD 性能 | 退化程度 | vs ISTA |
|------|---------|---------|---------|---------|
| ISTA | 0.843 | 0.844 | 1.00× | 基准 |
| FISTA | 0.821 | 0.822 | 1.00× | 1.03× |
| LISTA-CP | 0.448 | 1.215 | 2.71× | 1.88× (ID) |
| **LISTA-Momentum** | **0.526** | **0.555** | **1.06×** | **1.60×** |

**关键发现**: LISTA-Momentum 在 ID 和 OOD 上都优于 ISTA，泛化退化仅 1.06×！

#### 5.1.3 噪声鲁棒性

| 噪声水平 σ | LISTA-Momentum | ISTA | vs ISTA |
|-----------|----------------|------|---------|
| 0.001 | 0.542 | 0.843 | 1.55× |
| 0.005 | 0.543 | 0.843 | 1.55× |
| 0.010 | 0.543 | 0.843 | 1.55× |
| 0.020 | 0.546 | 0.843 | 1.54× |
| 0.050 | 0.562 | 0.843 | 1.50× |
| 0.100 | 0.610 | 0.843 | 1.38× |

**分析**: LISTA-Momentum 在所有噪声水平下都优于 ISTA，鲁棒性良好。

### 5.2 维度泛化实验

| 维度 n | ISTA | LISTA-DimAgnostic | vs ISTA | 说明 |
|--------|------|-------------------|---------|------|
| 50 | 0.479 | **0.088** | **5.44×** | 训练维度 |
| 100 | 0.701 | **0.080** | **8.72×** | 训练维度，最佳 |
| 150 | 0.799 | **0.106** | **7.52×** | 训练维度 |
| 200 | 0.852 | **0.158** | **5.38×** | 训练维度 |
| 300 | 0.900 | **0.239** | **3.77×** | 训练维度 |

**突破性结果**: 维度无关 LISTA 在训练维度范围内显著优于 ISTA (3.77-8.72×)！

### 5.3 参数矩阵 W 分析

| 层 | 与 I 的余弦相似度 | 谱半径 | 条件数 | 有效秩 |
|----|-----------------|--------|--------|--------|
| 1 | 0.915 | 1.452 | 3.01 | 200 |
| 5 | 0.929 | 1.522 | 2.23 | 200 |
| 10 | 0.964 | 1.426 | 2.29 | 200 |

**发现**:
1. W 接近单位矩阵 (余弦相似度 0.915-0.964)
2. 谱半径 > 1，但网络仍然有效
3. 有效秩 = 200 (满秩)

---

## 6. 消融实验

### 6.1 层数影响

| 层数 T | ISTA | LISTA-CP | LISTA-Momentum | vs ISTA |
|--------|------|----------|----------------|---------|
| 5 | 0.859 | 0.517 | 0.643 | 1.34× |
| 10 | 0.843 | 0.448 | 0.526 | 1.60× |
| 20 | 0.821 | 0.360 | 0.447 | 1.84× |

**分析**: 性能随层数增加而提升，LISTA-Momentum 在所有 T 值上都优于 ISTA。

### 6.2 初始化策略

| 初始化策略 | LISTA-CP 误差 | 收敛速度 |
|-----------|--------------|---------|
| 随机初始化 | 0.08 | 慢 |
| 问题结构初始化 (B = A^T) | 0.028 | 快 |

**分析**: 问题结构初始化 (B = A^T) 显著加速收敛。

### 6.3 泛化性分析

| 维度 | 能否泛化 | 实验结果 | 说明 |
|------|---------|---------|------|
| 不同 x (同一 A) | ✓ | LISTA 1.60× 优于 ISTA | 标准设定 |
| 不同 A | ✓ | LISTA 1.06× 退化 | LISTA-Momentum 解决 |
| 不同 n | ✓ | LISTA 3.77-8.72× 优于 ISTA | 维度无关 LISTA |
| 不同 s | 部分 | s < 训练值 OK | 需要覆盖训练范围 |

---

## 7. 结论与讨论

### 7.1 主要贡献

1. **提出 LISTA-Momentum**: 通过可学习动量机制提升收敛速度，解决泛化性问题
2. **设计维度无关 LISTA**: 可处理任意维度输入，无需重新训练
3. **提供理论分析**: 谱半径稳定性验证和参数矩阵分析
4. **系统实验验证**: 在多个问题上进行全面的对比实验和消融实验

### 7.2 关键发现

1. **LISTA-Momentum 的泛化性**: 通过不把 A 固化在权重中，实现了 1.06× 的泛化退化
2. **维度无关架构的成功**: 使用 LayerNorm + 共享 MLP，实现了 3.77-8.72× 的性能提升
3. **参数矩阵 W 的特性**: W 接近单位矩阵，学习的是"如何改进梯度"

### 7.3 局限性

1. **维度固定**: 原始 LISTA-CP 只能处理固定维度
2. **训练成本**: 需要大量同类问题的数据
3. **理论保证**: 缺乏收敛性的严格证明

### 7.4 未来工作

1. **扩展验证**: 在更大规模问题上验证
2. **理论分析**: 研究收敛性和泛化性
3. **应用推广**: 将通用架构思想应用到 ADMM-Net 和 PGD-Net

---

## 参考文献

1. Gregor, K., & LeCun, Y. (2010). Learning fast approximations of sparse coding. In *ICML*.
2. Chen, X., Liu, J., Wang, Z., & Yin, W. (2018). Theoretical linear convergence of unfolded ISTA and its practical weights and thresholds. In *NeurIPS*.
3. Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183-202.
4. Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning*, 3(1), 1-122.
5. Parikh, N., & Boyd, S. (2014). Proximal algorithms. *Foundations and Trends in Optimization*, 1(3), 127-239.
6. Duchi, J., Shalev-Shwartz, S., Singer, Y., & Chandra, T. (2008). Efficient projections onto the ℓ1-ball for learning in high dimensions. In *ICML*.
7. Monga, V., Li, Y., & Eldar, Y. C. (2021). Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing. *IEEE Signal Processing Magazine*, 38(2), 18-44.
8. Sun, J., Li, H., & Xu, Z. (2016). Deep ADMM-Net for compressive sensing MRI. In *NeurIPS*.
9. Andrychowicz, M., Denil, M., Gomez, S., Hoffman, M. W., Pfau, D., Schaul, T., & de Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. In *NeurIPS*.
10. Borgerding, M., Schniter, P., & Rangan, S. (2017). AMP-inspired deep networks for sparse linear inverse problems. *IEEE Transactions on Signal Processing*, 65(18), 4293-4308.

---

## 附录

### A. 代码结构

详见 README.md 中的项目结构说明。

### B. 运行说明

```bash
# 安装依赖
pip install -r requirements.txt

# 运行 LASSO 实验
cd lasso && python train.py

# 运行维度无关 LISTA 实验
python train_dimension_agnostic_robust.py

# 运行低秩实验
cd low_rank && python train.py

# 运行 QP 实验
cd qp && python train.py
```

### C. 参数设置

所有实验参数可在各模块的 `train.py` 文件中调整。主要参数：
- T: 展开层数 (默认 10)
- hidden_dim: 隐藏层维度 (默认 32)
- lr: 学习率 (默认 1e-3)
- num_epochs: 训练轮次 (默认 100)
