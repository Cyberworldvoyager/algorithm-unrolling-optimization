"""
lasso/lista_dimension_agnostic.py — 维度无关的 LISTA-Momentum

核心思想: 用维度无关的变换替代固定的 n×n 矩阵 W

方法:
1. LayerNorm + 小 MLP: 先归一化，再用共享的 MLP 变换
2. 元素独立变换: 对每个元素应用相同的变换
3. 低秩分解: W = U @ V^T，U, V 可以适应不同维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class DimensionAgnosticLayer(nn.Module):
    """维度无关的 LISTA-Momentum 单层。

    核心设计:
    1. 用 LayerNorm 归一化梯度 (消除维度影响)
    2. 用共享的 MLP 变换 (维度无关)
    3. 用可学习的缩放因子恢复尺度

    这样，同一个网络可以处理不同维度的输入。
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()

        # 梯度变换网络 (维度无关)
        # 输入: 归一化后的梯度元素
        # 输出: 变换后的梯度元素
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习的步长和动量
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.1))

        # 可学习的缩放因子 (用于恢复尺度)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                x_prev: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        x : Tensor (batch, n) — 当前估计
        x_prev : Tensor (batch, n) — 上一步估计
        A : Tensor (batch, m, n) — 测量矩阵
        """
        # 动量外推
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)

        # 用当前 A 计算梯度: g = A^T (A y - b)
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)  # (batch, m)
        residual = Ay - b  # (batch, m)
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # (batch, n)

        # 维度无关的梯度变换
        # 1. 归一化 (消除维度影响)
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-8
        grad_normalized = (grad - grad_mean) / grad_std

        # 2. 元素独立变换 (共享权重)
        # 将 (batch, n) 变换为 (batch*n, 1)，应用 MLP，再变回 (batch, n)
        batch_size, n = grad.shape
        grad_flat = grad_normalized.reshape(-1, 1)  # (batch*n, 1)
        transformed_flat = self.transform(grad_flat)  # (batch*n, 1)
        transformed_normalized = transformed_flat.reshape(batch_size, n)  # (batch, n)

        # 3. 恢复尺度
        transformed_grad = transformed_normalized * grad_std * self.scale

        # 更新
        z = y - self.eta * transformed_grad

        # 软阈值化
        return torch.sign(z) * torch.maximum(
            torch.abs(z) - self.threshold, torch.zeros_like(z))


class DimensionAgnosticLISTA(nn.Module):
    """维度无关的 LISTA-Momentum。

    可以处理不同维度的 A 矩阵，无需重新训练。
    """

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([
            DimensionAgnosticLayer(hidden_dim) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        A : Tensor (batch, m, n) 或 (m, n) — 测量矩阵
        x0 : Tensor (batch, n) — 初始估计
        """
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        x_prev = x.clone()

        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new

        return x


# ============================================================
# 方法 2: 低秩分解版本
# ============================================================

class LowRankDimensionAgnosticLayer(nn.Module):
    """低秩分解的维度无关层。

    使用低秩分解 W = U @ V^T，其中 U, V 的秩是固定的，
    但可以适应不同维度。
    """

    def __init__(self, rank: int = 10):
        super().__init__()
        self.rank = rank

        # 低秩分解: W = U @ V^T
        # U 和 V 的第一维是 n (动态)，第二维是 rank (固定)
        self.U_generator = nn.Linear(1, rank)  # 从归一化索引生成 U
        self.V_generator = nn.Linear(1, rank)  # 从归一化索引生成 V

        # 可学习的步长和动量
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                x_prev: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        batch_size = b.shape[0]
        n = x.shape[1]

        # 动量外推
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)

        # 用当前 A 计算梯度
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

        # 归一化索引 (0 到 1)
        idx = torch.linspace(0, 1, n, device=x.device).unsqueeze(0).expand(batch_size, -1)  # (batch, n)

        # 生成低秩矩阵
        U = self.U_generator(idx.unsqueeze(-1)).squeeze(-2)  # (batch, n, rank)
        V = self.V_generator(idx.unsqueeze(-1)).squeeze(-2)  # (batch, n, rank)

        # 低秩变换: W @ g = U @ (V^T @ g)
        Vtgrad = torch.bmm(V.transpose(1, 2), grad.unsqueeze(-1)).squeeze(-1)  # (batch, rank)
        transformed_grad = torch.bmm(U, Vtgrad.unsqueeze(-1)).squeeze(-1)  # (batch, n)

        # 更新
        z = y - self.eta * transformed_grad

        return torch.sign(z) * torch.maximum(
            torch.abs(z) - self.threshold, torch.zeros_like(z))


class LowRankDimensionAgnosticLISTA(nn.Module):
    """低秩分解的维度无关 LISTA。"""

    def __init__(self, T: int = 10, rank: int = 10):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([
            LowRankDimensionAgnosticLayer(rank) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        x_prev = x.clone()

        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new

        return x


# ============================================================
# 工厂函数
# ============================================================

def create_dimension_agnostic_lista(variant: str = 'mlp', T: int = 10, **kwargs):
    """创建维度无关的 LISTA 变体。

    Parameters
    ----------
    variant : str
        'mlp' — LayerNorm + MLP 变换
        'lowrank' — 低秩分解
    """
    if variant == 'mlp':
        return DimensionAgnosticLISTA(T, kwargs.get('hidden_dim', 32))
    elif variant == 'lowrank':
        return LowRankDimensionAgnosticLISTA(T, kwargs.get('rank', 10))
    else:
        raise ValueError(f"Unknown variant: {variant}")
