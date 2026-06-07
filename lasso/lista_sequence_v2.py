"""
lasso/lista_sequence_v2.py — 改进的序列模型 LISTA

改进:
1. 使用归一化梯度作为输入 (降低维度)
2. 元素独立处理 (减少参数)
3. 更好的初始化
4. 支持动态迭代次数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


# ============================================================
# 改进的 RNN-LISTA
# ============================================================

class ImprovedRNNLayer(nn.Module):
    """改进的 RNN-LISTA 单层。

    核心改进:
    1. 输入: 归一化后的梯度 (n 维，而非 2n 维)
    2. 元素独立处理: 每个维度共享相同的 RNN
    3. 残差连接: 保持梯度流
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # RNN 单元 (输入: 归一化梯度)
        self.rnn = nn.GRUCell(input_size=1, hidden_size=hidden_dim)

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (batch, n) — 当前估计
        grad : Tensor (batch, n) — 梯度
        h : Tensor (batch, n, hidden_dim) — 隐藏状态
        """
        batch_size, n = x.shape

        # 归一化梯度
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_normalized = (grad - grad_mean) / grad_std

        # 元素独立处理
        grad_flat = grad_normalized.reshape(-1, 1)  # (batch*n, 1)
        h_flat = h.reshape(-1, self.hidden_dim)  # (batch*n, hidden_dim)

        # RNN 更新
        h_new_flat = self.rnn(grad_flat, h_flat)  # (batch*n, hidden_dim)

        # 计算更新量
        update_flat = self.hidden_to_update(h_new_flat)  # (batch*n, 1)
        update = update_flat.reshape(batch_size, n) * grad_std  # 恢复尺度

        # 更新 x
        x_new = x + self.eta * update

        # 重塑隐藏状态
        h_new = h_new_flat.reshape(batch_size, n, self.hidden_dim)

        return x_new, h_new


class ImprovedRNNLISTA(nn.Module):
    """改进的 RNN-LISTA。

    优势:
    - 输入维度降低 (n vs 2n)
    - 元素独立处理 (参数更少)
    - 支持动态迭代次数
    """

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.layer = ImprovedRNNLayer(hidden_dim)

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                T: Optional[int] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)

        T = T if T is not None else self.T

        for _ in range(T):
            # 计算梯度
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

            # RNN 更新
            x, h = self.layer(x, grad, h)

        return x


# ============================================================
# 改进的 LSTM-LISTA
# ============================================================

class ImprovedLSTMLayer(nn.Module):
    """改进的 LSTM-LISTA 单层。"""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # LSTM 单元
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, n = x.shape

        # 归一化梯度
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_normalized = (grad - grad_mean) / grad_std

        # 元素独立处理
        grad_flat = grad_normalized.reshape(-1, 1)
        h_flat = h.reshape(-1, self.hidden_dim)
        c_flat = c.reshape(-1, self.hidden_dim)

        # LSTM 更新
        h_new_flat, c_new_flat = self.lstm(grad_flat, (h_flat, c_flat))

        # 计算更新量
        update_flat = self.hidden_to_update(h_new_flat)
        update = update_flat.reshape(batch_size, n) * grad_std

        # 更新 x
        x_new = x + self.eta * update

        # 重塑隐藏状态
        h_new = h_new_flat.reshape(batch_size, n, self.hidden_dim)
        c_new = c_new_flat.reshape(batch_size, n, self.hidden_dim)

        return x_new, h_new, c_new


class ImprovedLSTMLISTA(nn.Module):
    """改进的 LSTM-LISTA。"""

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.layer = ImprovedLSTMLayer(hidden_dim)

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                T: Optional[int] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)

        T = T if T is not None else self.T

        for _ in range(T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
            x, h, c = self.layer(x, grad, h, c)

        return x


# ============================================================
# 改进的 Transformer-LISTA
# ============================================================

class ImprovedTransformerLayer(nn.Module):
    """改进的 Transformer-LISTA 单层。

    使用因果自注意力，只关注当前和过去的迭代。
    """

    def __init__(self, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 将归一化梯度映射到隐藏维度
        self.grad_to_hidden = nn.Linear(1, hidden_dim)

        # 因果自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                grad_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (batch, n)
        grad : Tensor (batch, n)
        grad_history : Tensor (batch, T, n)
        """
        batch_size, n = x.shape

        # 归一化梯度
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_normalized = (grad - grad_mean) / grad_std

        # 映射到隐藏维度
        grad_hidden = self.grad_to_hidden(grad_normalized.unsqueeze(-1))  # (batch, n, hidden_dim)

        # 更新历史
        grad_history_new = torch.cat([grad_history, grad_hidden.unsqueeze(1)], dim=1)

        # 因果自注意力
        attn_out, _ = self.self_attn(
            grad_history_new, grad_history_new, grad_history_new
        )  # (batch, T+1, n, hidden_dim)

        # 取最后一个位置
        attn_last = attn_out[:, -1, :, :]  # (batch, n, hidden_dim)

        # 计算更新量
        update = self.hidden_to_update(attn_last).squeeze(-1)  # (batch, n)
        update = update * grad_std  # 恢复尺度

        # 更新 x
        x_new = x + self.eta * update

        return x_new, grad_history_new


class ImprovedTransformerLISTA(nn.Module):
    """改进的 Transformer-LISTA。"""

    def __init__(self, T: int = 10, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.layer = ImprovedTransformerLayer(hidden_dim, num_heads)

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                T: Optional[int] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)

        T = T if T is not None else self.T

        # 初始化历史
        grad_history = torch.zeros(batch_size, 0, n, self.hidden_dim, device=b.device)

        for t in range(T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
            x, grad_history = self.layer(x, grad, grad_history)

        return x


# ============================================================
# 工厂函数
# ============================================================

def create_improved_sequence_lista(variant: str, n: int, T: int = 10, **kwargs):
    """创建改进的序列模型 LISTA。

    Parameters
    ----------
    variant : str — 'rnn', 'lstm', 'transformer'
    n : int — 信号维度 (仅用于兼容，实际不使用)
    T : int — 迭代次数
    """
    if variant == 'rnn':
        return ImprovedRNNLISTA(T, kwargs.get('hidden_dim', 32))
    elif variant == 'lstm':
        return ImprovedLSTMLISTA(T, kwargs.get('hidden_dim', 32))
    elif variant == 'transformer':
        return ImprovedTransformerLISTA(T, kwargs.get('hidden_dim', 32), kwargs.get('num_heads', 4))
    else:
        raise ValueError(f"Unknown variant: {variant}")
