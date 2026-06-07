"""
lasso/lista_sequence.py — 基于序列模型的 LISTA

使用 RNN/LSTM/Transformer 来建模优化求解过程：
1. RNN-LISTA: 用 RNN 学习迭代更新规则
2. LSTM-LISTA: 用 LSTM 处理长程依赖
3. Transformer-LISTA: 用自注意力捕捉迭代间关系
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


# ============================================================
# RNN-LISTA
# ============================================================

class RNNLISTALayer(nn.Module):
    """RNN-LISTA 单层。

    用 RNN 建模迭代更新：
    h_t = RNN(h_{t-1}, [x_t, grad_t])
    x_{t+1} = x_t + η * W * h_t
    """

    def __init__(self, n: int, hidden_dim: int = 64):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        # RNN 单元
        self.rnn = nn.GRUCell(input_size=2 * n, hidden_size=hidden_dim)

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Linear(hidden_dim, n)

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (batch, n) — 当前估计
        grad : Tensor (batch, n) — 梯度
        h : Tensor (batch, hidden_dim) — 隐藏状态

        Returns
        -------
        x_new : Tensor (batch, n) — 更新后的估计
        h_new : Tensor (batch, hidden_dim) — 新的隐藏状态
        """
        # 拼接 x 和 grad 作为 RNN 输入
        rnn_input = torch.cat([x, grad], dim=-1)

        # RNN 更新
        h_new = self.rnn(rnn_input, h)

        # 从隐藏状态计算更新量
        update = self.hidden_to_update(h_new)

        # 更新 x
        x_new = x + self.eta * update

        return x_new, h_new


class RNNLISTA(nn.Module):
    """RNN-LISTA: 用 RNN 建模优化过程。

    优势：
    - 可以处理任意迭代次数（动态展开）
    - 学习复杂的更新规则
    - 隐藏状态可以记住历史信息
    """

    def __init__(self, n: int, T: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.n = n
        self.T = T
        self.hidden_dim = hidden_dim

        self.layer = RNNLISTALayer(n, hidden_dim)

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                T: Optional[int] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        A : Tensor (batch, m, n) — 测量矩阵
        x0 : Tensor (batch, n) — 初始估计
        T : int — 迭代次数（可动态调整）
        """
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, self.hidden_dim, device=b.device)

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
# LSTM-LISTA
# ============================================================

class LSTMLISTALayer(nn.Module):
    """LSTM-LISTA 单层。

    用 LSTM 处理长程依赖：
    (h_t, c_t) = LSTM(h_{t-1}, c_{t-1}, [x_t, grad_t])
    x_{t+1} = x_t + η * W * h_t
    """

    def __init__(self, n: int, hidden_dim: int = 64):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        # LSTM 单元
        self.lstm = nn.LSTMCell(input_size=2 * n, hidden_size=hidden_dim)

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Linear(hidden_dim, n)

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_new : Tensor (batch, n)
        h_new : Tensor (batch, hidden_dim)
        c_new : Tensor (batch, hidden_dim)
        """
        # 拼接 x 和 grad
        lstm_input = torch.cat([x, grad], dim=-1)

        # LSTM 更新
        h_new, c_new = self.lstm(lstm_input, (h, c))

        # 从隐藏状态计算更新量
        update = self.hidden_to_update(h_new)

        # 更新 x
        x_new = x + self.eta * update

        return x_new, h_new, c_new


class LSTMLISTA(nn.Module):
    """LSTM-LISTA: 用 LSTM 建模优化过程。

    优势：
    - LSTM 可以更好地处理长程依赖
    - 细胞状态可以记住长期信息
    - 适合需要多步推理的优化问题
    """

    def __init__(self, n: int, T: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.n = n
        self.T = T
        self.hidden_dim = hidden_dim

        self.layer = LSTMLISTALayer(n, hidden_dim)

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                T: Optional[int] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=b.device)

        T = T if T is not None else self.T

        for _ in range(T):
            # 计算梯度
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

            # LSTM 更新
            x, h, c = self.layer(x, grad, h, c)

        return x


# ============================================================
# Transformer-LISTA
# ============================================================

class TransformerLISTALayer(nn.Module):
    """Transformer-LISTA 单层。

    用自注意力捕捉迭代间的关系：
    attn = SelfAttn([x_1, x_2, ..., x_t])
    x_{t+1} = x_t + η * W * attn
    """

    def __init__(self, n: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        # 将 x 映射到隐藏维度
        self.x_to_hidden = nn.Linear(n, hidden_dim)

        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 从隐藏状态到更新量
        self.hidden_to_update = nn.Linear(hidden_dim, n)

        # 可学习的步长
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, grad: torch.Tensor,
                x_history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (batch, n) — 当前估计
        grad : Tensor (batch, n) — 梯度
        x_history : Tensor (batch, T, n) — 历史估计

        Returns
        -------
        x_new : Tensor (batch, n)
        x_history_new : Tensor (batch, T+1, n)
        """
        batch_size = x.shape[0]

        # 映射到隐藏维度
        x_hidden = self.x_to_hidden(x_history)  # (batch, T, hidden_dim)

        # 自注意力
        attn_out, _ = self.self_attn(x_hidden, x_hidden, x_hidden)  # (batch, T, hidden_dim)

        # 取最后一个位置的输出
        attn_last = attn_out[:, -1, :]  # (batch, hidden_dim)

        # 计算更新量
        update = self.hidden_to_update(attn_last)  # (batch, n)

        # 更新 x
        x_new = x + self.eta * update

        # 更新历史
        x_history_new = torch.cat([x_history, x.unsqueeze(1)], dim=1)

        return x_new, x_history_new


class TransformerLISTA(nn.Module):
    """Transformer-LISTA: 用自注意力建模优化过程。

    优势：
    - 自注意力可以捕捉任意迭代间的关系
    - 并行计算效率高
    - 可以学习全局的优化策略
    """

    def __init__(self, n: int, T: int = 10, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.n = n
        self.T = T
        self.hidden_dim = hidden_dim

        self.layer = TransformerLISTALayer(n, hidden_dim, num_heads)

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
        x_history = x.unsqueeze(1)  # (batch, 1, n)

        for t in range(T):
            # 计算梯度
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

            # Transformer 更新
            x, x_history = self.layer(x, grad, x_history)

        return x


# ============================================================
# 工厂函数
# ============================================================

def create_sequence_lista(variant: str, n: int, T: int = 10, **kwargs):
    """创建序列模型 LISTA。

    Parameters
    ----------
    variant : str — 'rnn', 'lstm', 'transformer'
    n : int — 信号维度
    T : int — 迭代次数
    """
    if variant == 'rnn':
        return RNNLISTA(n, T, kwargs.get('hidden_dim', 64))
    elif variant == 'lstm':
        return LSTMLISTA(n, T, kwargs.get('hidden_dim', 64))
    elif variant == 'transformer':
        return TransformerLISTA(n, T, kwargs.get('hidden_dim', 64), kwargs.get('num_heads', 4))
    else:
        raise ValueError(f"Unknown variant: {variant}")
