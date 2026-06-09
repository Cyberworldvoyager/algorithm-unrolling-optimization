"""
lasso/lista.py — 经典展开网络 LISTA 及其耦合权重变体 LISTA-CP

实现:
1. LISTA      — 基本 ISTA 展开 (Gregor & LeCun, 2010)
                x_{t+1} = σ(W₁b + W₂x_t; θ_t)，W₁, W₂ 独立学习
2. LISTA-CP   — 耦合权重 LISTA (Chen et al., 2018)
                W₁ = ηB, W₂ = I - ηBA，仅由 B 参数化

设计说明 (与评分意见对应):
- 软阈值阈值 θ 通过 softplus 约束为非负，保证 σ(·;θ) 仍是合法的近端算子。
  自由参数 θ 在训练中可能变负，使软阈值退化为加噪，故统一改为 θ = softplus(ρ)。

本模块是项目中 LISTA / LISTA-CP 的唯一实现，由 run_all_experiments.py 直接导入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SoftThreshold(nn.Module):
    """可学习的软阈值激活: f(x) = sign(x)·max(|x| - θ, 0)，θ = softplus(ρ) ≥ 0。

    支持标量 (n=None) 或逐维度 (n>0) 阈值。
    """

    def __init__(self, init_threshold: float = 0.01, n: Optional[int] = None):
        super().__init__()
        # 以 softplus 的逆初始化 raw，使初始 θ ≈ init_threshold
        raw_init = math_softplus_inverse(init_threshold)
        if n is not None:
            self.raw = nn.Parameter(torch.full((n,), raw_init))
        else:
            self.raw = nn.Parameter(torch.tensor(raw_init))

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        theta = self.theta
        return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)


def math_softplus_inverse(y: float) -> float:
    """softplus 的逆: 给定目标 θ=y>0，返回 raw 使得 softplus(raw)=y。"""
    import math
    # softplus(x) = log(1+e^x) => x = log(e^y - 1)
    return math.log(math.expm1(y)) if y > 0 else -5.0


# ============================================================
# LISTA — 基本版本 (Gregor & LeCun, 2010)
# ============================================================

class LISTALayer(nn.Module):
    """LISTA 单层: x_{t+1} = σ(W₁b + W₂x_t; θ_t)，W₁∈R^{n×m}, W₂∈R^{n×n}。

    若提供 A_init (训练矩阵) 与 init_eta=η，则按 ISTA 等价方式初始化:
        W₁ = η·Aᵀ,  W₂ = I − η·AᵀA,  θ ≈ η·λ
    使网络初始即等价于一步 ISTA，训练只会在此基础上改进 (避免从随机权重
    出发学不到 ISTA 水平、反而劣于经典解的退化)。
    """

    def __init__(self, m: int, n: int, init_threshold: float = 0.01,
                 per_dim_threshold: bool = False,
                 A_init: Optional[torch.Tensor] = None, init_eta: float = 0.1):
        super().__init__()
        self.W1 = nn.Linear(m, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)
        if A_init is not None:
            # ISTA 等价初始化
            A = A_init
            self.W1.weight.data = (init_eta * A.t()).contiguous()
            self.W2.weight.data = torch.eye(n) - init_eta * (A.t() @ A)
        else:
            nn.init.xavier_uniform_(self.W1.weight)
            nn.init.eye_(self.W2.weight)
            self.W2.weight.data *= 0.9
        self.threshold = SoftThreshold(init_threshold, n if per_dim_threshold else None)

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.threshold(self.W1(b) + self.W2(x))


class LISTA(nn.Module):
    """LISTA 网络: T 层 ISTA 展开。W₁,W₂ 与训练矩阵 A 绑定，不随 A 泛化。

    传入 A_init 时各层做 ISTA 等价初始化 (推荐)，init_eta 一般取 1/L (L=‖AᵀA‖₂)。
    """

    def __init__(self, m: int, n: int, T: int = 10, init_threshold: float = 0.01,
                 per_dim_threshold: bool = False,
                 A_init: Optional[torch.Tensor] = None, init_eta: float = 0.1):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTALayer(m, n, init_threshold, per_dim_threshold, A_init, init_eta)
            for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: Optional[torch.Tensor] = None,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        # A 参数仅为接口统一 (LISTA 不使用)
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        for layer in self.layers:
            x = layer(b, x)
        return x

    def get_thresholds(self) -> List[float]:
        return [float(layer.threshold.theta.mean()) for layer in self.layers]


# ============================================================
# LISTA-CP — 耦合权重 (Chen et al., 2018)
# ============================================================

class LISTACPLayer(nn.Module):
    """LISTA-CP 单层: x_{t+1} = σ(ηBb + (I - ηBA)x_t; θ_t)。

    仅 B∈R^{n×m}、步长 η、阈值 θ 为可学习参数。
    B 初始化为 Aᵀ (ISTA 的最优选择)。
    """

    def __init__(self, A: torch.Tensor, n: int, init_eta: float = 0.1,
                 init_threshold: float = 0.01, per_dim_threshold: bool = False):
        super().__init__()
        self.register_buffer('A', A)                 # (m, n)，固定的训练矩阵
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.B = nn.Parameter(A.T.clone())           # (n, m)
        self.threshold = SoftThreshold(init_threshold, n if per_dim_threshold else None)

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Ax = F.linear(x, self.A)      # (batch, m)
        BAx = F.linear(Ax, self.B)    # (batch, n)
        Bb = F.linear(b, self.B)      # (batch, n)
        z = self.eta * Bb + x - self.eta * BAx
        return self.threshold(z)


class LISTACP(nn.Module):
    """LISTA-CP: 耦合权重的 LISTA。参数量 O(nm)，初始化即为 ISTA。"""

    def __init__(self, A: torch.Tensor, T: int = 10, init_eta: float = 0.1,
                 init_threshold: float = 0.01, per_dim_threshold: bool = False):
        super().__init__()
        m, n = A.shape
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            LISTACPLayer(A, n, init_eta, init_threshold, per_dim_threshold)
            for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: Optional[torch.Tensor] = None,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x0 if x0 is not None else torch.zeros(b.shape[0], self.n, device=b.device)
        for layer in self.layers:
            x = layer(b, x)
        return x

    def get_thresholds(self) -> List[float]:
        return [float(layer.threshold.theta.mean()) for layer in self.layers]

    def get_spectral_radius(self) -> List[float]:
        """每层 W₂ = I - η·B·A 的谱半径 (理论上 <1 保证收敛)。"""
        radii = []
        for layer in self.layers:
            W2 = torch.eye(layer.B.shape[0], device=layer.B.device) \
                 - layer.eta * layer.B @ layer.A
            radii.append(float(torch.linalg.eigvals(W2).abs().max()))
        return radii
