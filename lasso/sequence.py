"""
lasso/sequence.py — 基于序列模型的 LISTA (RNN / LSTM / Transformer)

共同思路 (逐元素、维度无关):
  每步计算梯度 g = Aᵀ(Ax - b)，逐元素归一化后送入循环/注意力单元，
  输出每个坐标的更新量，乘回 std(g) 恢复尺度:
      x_{t+1} = x_t + η · update · std(g_t)
  每个坐标共享同一套单元参数，因此与维度 n 无关。

本模块是 RNN/LSTM/Transformer-LISTA 的唯一实现，由 run_all_experiments.py 导入。
"""

import torch
import torch.nn as nn
from typing import Optional


def _grad(A: torch.Tensor, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
    return torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)


def _expand_A(A: torch.Tensor, batch_size: int) -> torch.Tensor:
    return A.unsqueeze(0).expand(batch_size, -1, -1) if A.dim() == 2 else A


class RNNLISTA(nn.Module):
    """RNN-LISTA: 用 GRU 逐元素建模迭代更新规则。"""

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T, self.hidden_dim = T, hidden_dim
        self.rnn = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        A = _expand_A(A, batch_size)
        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        for _ in range(self.T):
            grad = _grad(A, x, b)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((grad - grad.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h = self.rnn(g_norm, h.reshape(-1, self.hidden_dim))
            update = self.head(h).reshape(batch_size, n) * g_std
            x = x + self.eta * update
            h = h.reshape(batch_size, n, self.hidden_dim)
        return x


class LSTMLISTA(nn.Module):
    """LSTM-LISTA: 用 LSTM 逐元素建模，细胞状态记忆长程信息。"""

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T, self.hidden_dim = T, hidden_dim
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        A = _expand_A(A, batch_size)
        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        for _ in range(self.T):
            grad = _grad(A, x, b)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((grad - grad.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h_flat, c_flat = self.lstm(
                g_norm, (h.reshape(-1, self.hidden_dim), c.reshape(-1, self.hidden_dim)))
            update = self.head(h_flat).reshape(batch_size, n) * g_std
            x = x + self.eta * update
            h = h_flat.reshape(batch_size, n, self.hidden_dim)
            c = c_flat.reshape(batch_size, n, self.hidden_dim)
        return x


class TransformerLISTA(nn.Module):
    """Transformer-LISTA: 收集各步逐元素 hidden，用自注意力在迭代维上聚合。

    注意力作用于"迭代步"序列 (每个坐标独立)，最后一步的注意力输出决定更新量，
    使更新可依赖全部历史迭代而非仅当前步。
    """

    def __init__(self, T: int = 10, hidden_dim: int = 32, num_heads: int = 4):
        super().__init__()
        self.T, self.hidden_dim = T, hidden_dim
        self.grad_to_hidden = nn.Linear(1, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        A = _expand_A(A, batch_size)
        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        hiddens, stds = [], []
        for _ in range(self.T):
            grad = _grad(A, x, b)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = (grad - grad.mean(dim=-1, keepdim=True)) / g_std
            h = self.grad_to_hidden(g_norm.reshape(-1, 1)).reshape(
                batch_size, n, self.hidden_dim)
            hiddens.append(h)
            stds.append(g_std)
            # 用历史 hidden 经注意力聚合得到当前更新
            seq = torch.stack(hiddens, dim=1)                        # (B, t, n, H)
            t = seq.shape[1]
            seq = seq.permute(0, 2, 1, 3).reshape(batch_size * n, t, self.hidden_dim)
            attn_out, _ = self.self_attn(seq, seq, seq)
            attn_last = attn_out[:, -1, :].reshape(batch_size, n, self.hidden_dim)
            update = self.head(attn_last.reshape(-1, self.hidden_dim)).reshape(
                batch_size, n) * g_std
            x = x + self.eta * update
        return x


def create_sequence_lista(variant: str, T: int = 10, **kwargs) -> nn.Module:
    if variant == 'rnn':
        return RNNLISTA(T, kwargs.get('hidden_dim', 32))
    if variant == 'lstm':
        return LSTMLISTA(T, kwargs.get('hidden_dim', 32))
    if variant == 'transformer':
        return TransformerLISTA(T, kwargs.get('hidden_dim', 32), kwargs.get('num_heads', 4))
    raise ValueError(f"Unknown variant: {variant}")
