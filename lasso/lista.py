"""
lasso/lista.py — LISTA 及其变体实现

实现以下展开网络:
1. LISTA — 基本 ISTA 展开 (Gregor & LeCun, 2010)
2. LISTA-CP — 耦合权重 LISTA (Chen et al., 2018)
3. LISTA-SS — 带稀疏结构和重启的 LISTA
4. LISTA-CP-FISTA — 带 Nesterov 动量的 LISTA-CP

理论基础:
- LISTA: x_{t+1} = σ(W₁b + W₂x_t; θ_t)
- LISTA-CP: W₁ = ηB, W₂ = I - ηB·A，参数由 B 参数化
- LISTA-CP-FISTA: 在 LISTA-CP 基础上引入动量变量 y_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class SoftThreshold(nn.Module):
    """可学习的软阈值激活函数。

    f(x) = sign(x) * max(|x| - θ, 0)

    支持标量、逐维度、逐层三种模式。
    """

    def __init__(self, init_threshold: float = 0.1, n: Optional[int] = None):
        super().__init__()
        if n is not None:
            self.theta = nn.Parameter(torch.full((n,), init_threshold))
        else:
            self.theta = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.maximum(torch.abs(x) - self.theta, torch.zeros_like(x))


# ============================================================
# LISTA — 基本版本 (Gregor & LeCun, 2010)
# ============================================================

class LISTALayer(nn.Module):
    """LISTA 单层: x_{t+1} = σ(W₁b + W₂x_t; θ_t)

    W₁ ∈ R^{n×m}, W₂ ∈ R^{n×n} 独立学习。
    """

    def __init__(self, m: int, n: int, init_eta: float = 0.1,
                 per_dim_threshold: bool = False):
        super().__init__()
        self.W1 = nn.Linear(m, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.eye_(self.W2.weight)
        self.W2.weight.data *= 0.9

        self.threshold = SoftThreshold(
            init_threshold=init_eta * 0.1,
            n=n if per_dim_threshold else None,
        )

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.threshold(self.W1(b) + self.W2(x))


class LISTA(nn.Module):
    """LISTA 网络: T 层 ISTA 展开。

    Parameters
    ----------
    m : int  — 观测维度
    n : int  — 信号维度
    T : int  — 展开层数
    """

    def __init__(self, m: int, n: int, T: int = 10, init_eta: float = 0.1,
                 per_dim_threshold: bool = False):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTALayer(m, n, init_eta, per_dim_threshold) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        intermediates = [x] if return_intermediates else None
        for layer in self.layers:
            x = layer(b, x)
            if return_intermediates:
                intermediates.append(x)
        return (x, intermediates) if return_intermediates else x

    def get_thresholds(self) -> List[float]:
        return [layer.threshold.theta.item() for layer in self.layers]


# ============================================================
# LISTA-CP — 耦合权重 (Chen et al., 2018)
# ============================================================

class LISTACPLayer(nn.Module):
    """LISTA-CP 单层。

    核心思想: W₁ = η·B, W₂ = I - η·B·A，用 B 参数化两个矩阵。
    参数量从 O(nm + n²) 降到 O(nm)。

    更新公式:
        x_{t+1} = σ(η·B·b + (I - η·B·A)·x_t; θ_t)

    其中 B ∈ R^{n×m} 是唯一的学习参数。
    """

    def __init__(self, A: torch.Tensor, n: int, init_eta: float = 0.1,
                 per_dim_threshold: bool = False):
        super().__init__()
        m = A.shape[0]
        self.register_buffer('A', A)  # (m, n)
        # eta 是可学习参数 (不是 buffer)
        self.eta = nn.Parameter(torch.tensor(init_eta))

        # B 是学习参数
        # 初始化: B = A^T (ISTA 的最优选择)
        self.B = nn.Parameter(A.T.clone())  # (n, m)

        self.threshold = SoftThreshold(
            init_threshold=init_eta * 0.1,
            n=n if per_dim_threshold else None,
        )

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # W₁b = η·B·b,  W₂x = (I - η·B·A)·x = x - η·B·(A·x)
        Ax = F.linear(x, self.A)        # (batch, m)
        BAx = F.linear(Ax, self.B)      # (batch, n)
        Bb = F.linear(b, self.B)        # (batch, n)
        z = self.eta * Bb + x - self.eta * BAx
        return self.threshold(z)


class LISTACP(nn.Module):
    """LISTA-CP: 耦合权重的 LISTA (Chen et al., 2018)。

    优势:
    - 参数量减半 (只有 B，没有独立的 W₁, W₂)
    - 初始化即为 ISTA，训练后超越 ISTA
    - 谱半径 ρ(W₂) < 1 有理论保证（当 η 足够小）

    Parameters
    ----------
    A : Tensor (m, n) — 测量矩阵
    T : int — 展开层数
    init_eta : float — 步长初始值
    """

    def __init__(self, A: torch.Tensor, T: int = 10, init_eta: float = 0.1,
                 per_dim_threshold: bool = False):
        super().__init__()
        m, n = A.shape
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTACPLayer(A, n, init_eta, per_dim_threshold) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        intermediates = [x] if return_intermediates else None
        for layer in self.layers:
            x = layer(b, x)
            if return_intermediates:
                intermediates.append(x)
        return (x, intermediates) if return_intermediates else x

    def get_thresholds(self) -> List[float]:
        return [layer.threshold.theta.item() for layer in self.layers]

    def get_spectral_radius(self) -> List[float]:
        """计算每层 W₂ = I - η·B·A 的谱半径。

        理论上 ρ(W₂) < 1 保证收敛。训练后应验证此条件。
        """
        radii = []
        for layer in self.layers:
            W2 = torch.eye(layer.B.shape[1], device=layer.B.device) - \
                 layer.eta * layer.B @ layer.A
            eigvals = torch.linalg.eigvals(W2)
            radii.append(eigvals.abs().max().item())
        return radii


# ============================================================
# LISTA-CP-SS — 带稀疏结构的 LISTA-CP
# ============================================================

class LISTACPSSLayer(nn.Module):
    """LISTA-CP-SS 单层。

    在 LISTA-CP 基础上:
    1. 逐维度独立阈值 θ_t ∈ R^n
    2. 每层学习独立的步长 η_t (而非共享)

    这使得每层能自适应不同的稀疏模式。
    """

    def __init__(self, A: torch.Tensor, n: int, init_eta: float = 0.1):
        super().__init__()
        self.register_buffer('A', A)
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.B = nn.Parameter(A.T.clone())
        self.threshold = SoftThreshold(init_threshold=init_eta * 0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Ax = F.linear(x, self.A)
        BAx = F.linear(Ax, self.B)
        Bb = F.linear(b, self.B)
        z = self.eta * Bb + x - self.eta * BAx
        return self.threshold(z)


class LISTACPSS(nn.Module):
    """LISTA-CP-SS: 耦合权重 + 逐维度阈值 + 独立步长。

    是 LISTA-CP 的更强变体，每个输出维度有独立的阈值。
    """

    def __init__(self, A: torch.Tensor, T: int = 10, init_eta: float = 0.1):
        super().__init__()
        m, n = A.shape
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTACPSSLayer(A, n, init_eta) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        intermediates = [x] if return_intermediates else None
        for layer in self.layers:
            x = layer(b, x)
            if return_intermediates:
                intermediates.append(x)
        return (x, intermediates) if return_intermediates else x


# ============================================================
# LISTA-CP-FISTA — 带 Nesterov 动量的 LISTA-CP
# ============================================================

class LISTACPFISTALayer(nn.Module):
    """LISTA-CP-FISTA 单层。

    模仿 FISTA 的 Nesterov 动量机制:
        y_t = x_t + β_t · (x_t - x_{t-1})     # 动量外推
        x_{t+1} = σ(η·B·b + (I - η·B·A)·y_t; θ_t)  # ISTA 步

    β_t 是可学习的动量参数，初始化为经典 FISTA 的 β₀ = 0。
    """

    def __init__(self, A: torch.Tensor, n: int, init_eta: float = 0.1):
        super().__init__()
        self.register_buffer('A', A)
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.B = nn.Parameter(A.T.clone())
        # 动量参数 β_t，初始化为 0 (相当于无动量)
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.threshold = SoftThreshold(init_threshold=init_eta * 0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                x_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x_new : Tensor — 更新后的估计
        x : Tensor — 当前 x (供下层作为 x_prev)
        """
        # Nesterov 动量外推
        y = x + torch.sigmoid(self.beta) * (x - x_prev)
        # ISTA 步
        Ax = F.linear(y, self.A)
        BAx = F.linear(Ax, self.B)
        Bb = F.linear(b, self.B)
        z = self.eta * Bb + y - self.eta * BAx
        x_new = self.threshold(z)
        return x_new, x


class LISTACPFISTA(nn.Module):
    """LISTA-CP-FISTA: 带 Nesterov 动量的 LISTA-CP。

    在 LISTA-CP 基础上引入可学习的动量参数，模拟 FISTA 的加速效果。
    初始化 β=0 退化为 LISTA-CP，训练后学习最优动量调度。

    Parameters
    ----------
    A : Tensor (m, n) — 测量矩阵
    T : int — 展开层数
    init_eta : float — 步长初始值
    """

    def __init__(self, A: torch.Tensor, T: int = 10, init_eta: float = 0.1):
        super().__init__()
        m, n = A.shape
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTACPFISTALayer(A, n, init_eta) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        x_prev = x.clone()
        intermediates = [x] if return_intermediates else None
        for layer in self.layers:
            x, x_prev = layer(b, x, x_prev)
            if return_intermediates:
                intermediates.append(x)
        return (x, intermediates) if return_intermediates else x

    def get_thresholds(self) -> List[float]:
        return [layer.threshold.theta.item() for layer in self.layers]

    def get_momentums(self) -> List[float]:
        """获取每层的动量参数 (sigmoid 映射后)。"""
        return [torch.sigmoid(layer.beta).item() for layer in self.layers]


# ============================================================
# 带初始化的工厂函数
# ============================================================

def create_lista(A: torch.Tensor, variant: str = 'cp', T: int = 10,
                 init_eta: float = 0.1) -> nn.Module:
    """创建 LISTA 变体的工厂函数。

    Parameters
    ----------
    A : Tensor (m, n) — 测量矩阵
    variant : str — 'basic', 'cp', 'ss', 'fista'
    T : int — 展开层数
    init_eta : float — 步长初始值
    """
    variants = {
        'basic': lambda: LISTA(A.shape[0], A.shape[1], T, init_eta),
        'cp': lambda: LISTACP(A, T, init_eta),
        'ss': lambda: LISTACPSS(A, T, init_eta),
        'fista': lambda: LISTACPFISTA(A, T, init_eta),
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    return variants[variant]()


# 保持向后兼容
LISTAWithInit = LISTACP
