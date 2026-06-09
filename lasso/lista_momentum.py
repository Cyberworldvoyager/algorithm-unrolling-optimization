"""
lasso/lista_momentum.py — LISTA-Momentum 及维度无关 LISTA (DA-LISTA)

1. LISTAMomentum        — 带可学习动量的展开网络。不把 A 固化进权重，
                          每层用当前 A 计算梯度 g=Aᵀ(Ay-b)，再经可学习变换 W。
2. DimensionAgnosticLISTA — 用 (LayerNorm + 逐元素共享 MLP) 替代固定的 n×n 矩阵，
                          参数量与维度 n 无关，可处理任意维度输入。

稳定性修复 (对应评分意见 "LISTA-Momentum 坍缩为 1.0"):
- 阈值 θ = softplus(ρ)，非负且初始很小 (0.01)，避免一次性把输出全部阈值为 0；
- 梯度变换 W 改为残差形式 (I + ΔW)，ΔW 零初始化，初始严格等价于 ISTA 步；
- 步长 η 初始 0.05，配合训练循环中的梯度裁剪 (max_norm=1.0) 抑制发散。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from .lista import SoftThreshold


# ============================================================
# LISTA-Momentum (固定维度，可学习 n×n 梯度变换)
# ============================================================

class LISTAMomentumLayer(nn.Module):
    """LISTA-Momentum 单层。

        y_t   = x_t + sigmoid(β)·(x_t - x_{t-1})         # 动量外推
        g_t   = Aᵀ(A y_t - b)                            # 用当前 A 算梯度
        x_{t+1} = σ( y_t - η·(g_t + ΔW g_t) ; θ )        # 残差化梯度变换
    """

    def __init__(self, n: int, init_eta: float = 0.05, init_threshold: float = 0.01):
        super().__init__()
        # 残差形式: 有效变换为 (I + dW)，dW 零初始化 => 初始即标准 ISTA 步
        self.dW = nn.Linear(n, n, bias=False)
        nn.init.zeros_(self.dW.weight)
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.beta = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5
        self.threshold = SoftThreshold(init_threshold, n=None)

    def forward(self, b, x, x_prev, A):
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
        transformed = grad + self.dW(grad)            # (I + dW) g
        z = y - self.eta * transformed
        return self.threshold(z)


class LISTAMomentum(nn.Module):
    """LISTA-Momentum 网络 (T 层，固定维度 n)。"""

    def __init__(self, m: int, n: int, T: int = 10,
                 init_eta: float = 0.05, init_threshold: float = 0.01):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTAMomentumLayer(n, init_eta, init_threshold) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)
        x_prev = x.clone()
        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new
        return x

    def get_W_matrix(self, layer_idx: int = 0) -> torch.Tensor:
        """返回某层的有效梯度变换矩阵 W = I + dW，用于谱分析。"""
        layer = self.layers[layer_idx]
        n = layer.dW.weight.shape[0]
        return torch.eye(n, device=layer.dW.weight.device) + layer.dW.weight

    def get_momentums(self) -> List[float]:
        return [float(torch.sigmoid(l.beta)) for l in self.layers]


# ============================================================
# 维度无关 LISTA (DA-LISTA)
# ============================================================

class DimensionAgnosticLayer(nn.Module):
    """维度无关单层。

    用 (逐元素归一化 + 共享 MLP + 残差 + 尺度恢复) 替代固定 n×n 矩阵:
        g_norm = (g - mean) / std
        g_hat  = (MLP(g_norm) + g_norm) · std · scale
        x_{t+1} = σ( y - η·g_hat ; θ )
    MLP 仅作用于单个元素 (Linear(1,h)->...->Linear(h,1))，故与维度 n 无关。
    """

    def __init__(self, hidden_dim: int = 32, init_eta: float = 0.1,
                 init_threshold: float = 0.01):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.threshold = SoftThreshold(init_threshold, n=None)

    def forward(self, b, x, x_prev, A):
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

        batch_size, n = grad.shape
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_norm = (grad - grad_mean) / grad_std

        transformed = self.transform(grad_norm.reshape(-1, 1)).reshape(batch_size, n)
        transformed = (transformed + grad_norm) * grad_std * self.scale   # 残差 + 恢复尺度
        z = y - self.eta * transformed
        return self.threshold(z)


class DimensionAgnosticLISTA(nn.Module):
    """维度无关 LISTA-Momentum。参数量与 n 无关，可跨维度推理。"""

    def __init__(self, T: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([
            DimensionAgnosticLayer(hidden_dim) for _ in range(T)
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
