"""
qp/pgd_net.py — PGD-Net 及其变体

实现以下展开网络:
1. PGDNet — 基本 PGD 展开
2. PGDMomentumNet — 带 Nesterov 动量的 PGD 展开
3. PGDLearnedQNet — 学习 Q 矩阵修正的 PGD 展开

关键改进:
- 修复 box 约束支持向量值 lb/ub
- 增加 Nesterov 动量变体
- 统一投影算子接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union


class ProjectBox(nn.Module):
    """Box 约束投影层。

    支持标量或向量值的上下界。
    Proj(x) = clamp(x, lb, ub)
    """

    def __init__(self, lb: Union[float, torch.Tensor] = 0.0,
                 ub: Union[float, torch.Tensor] = 1.0):
        super().__init__()
        if isinstance(lb, torch.Tensor):
            self.register_buffer('lb', lb)
        else:
            self.lb = lb
        if isinstance(ub, torch.Tensor):
            self.register_buffer('ub', ub)
        else:
            self.ub = ub

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.lb, self.ub)


class ProjectSimplex(nn.Module):
    """单纯形约束投影层。

    Proj(x) = argmin_{y >= 0, sum(y) = s} ||y - x||^2
    使用 Duchi et al. (2008) O(n log n) 算法。
    """

    def __init__(self, s: float = 1.0):
        super().__init__()
        self.s = s

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        n = v.shape[-1]
        u, _ = torch.sort(v, dim=-1, descending=True)
        cssv = torch.cumsum(u, dim=-1) - self.s
        rho = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
        rho = torch.sum(u * rho > cssv, dim=-1, keepdim=True)
        theta = cssv.gather(-1, rho.long() - 1) / rho.float()
        return torch.maximum(v - theta, torch.zeros_like(v))


class ProjectNonNegative(nn.Module):
    """非负约束投影层。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, min=0.0)


def create_projector(constraint_type: str, constraint_params: Dict) -> nn.Module:
    """创建投影算子的工厂函数。"""
    if constraint_type == 'box':
        lb = constraint_params.get('lb', 0.0)
        ub = constraint_params.get('ub', 1.0)
        return ProjectBox(lb, ub)
    elif constraint_type == 'simplex':
        s = constraint_params.get('s', 1.0)
        return ProjectSimplex(s)
    elif constraint_type == 'nonneg':
        return ProjectNonNegative()
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


# ============================================================
# PGD-Net — 基本版本
# ============================================================

class PGDNetLayer(nn.Module):
    """PGD-Net 单层。

    x_{k+1} = Proj_C(x_k - η_k · (Qx_k + c))

    可选: 学习 Q 的修正 Q_correction
    """

    def __init__(self, n: int, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None,
                 learnable_Q: bool = False):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.learnable_Q = learnable_Q

        if learnable_Q:
            # Q 修正矩阵，初始化为 0
            self.Q_correction = nn.Parameter(torch.zeros(n, n))

        if constraint_params is None:
            constraint_params = {}
        self.projector = create_projector(constraint_type, constraint_params)

    def forward(self, x: torch.Tensor, Q: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        # 可选: 修正 Q
        if self.learnable_Q:
            Q = Q + self.Q_correction

        # 梯度
        if Q.dim() == 2:
            grad = x @ Q.T + c
        else:
            grad = torch.bmm(x.unsqueeze(1), Q).squeeze(1) + c

        # 梯度步 + 投影
        return self.projector(x - self.eta * grad)


class PGDNet(nn.Module):
    """PGD-Net: T 层 PGD 展开。

    Parameters
    ----------
    n : int — 变量维度
    T : int — 展开层数
    constraint_type : str — 约束类型
    constraint_params : dict — 约束参数
    learnable_Q : bool — 是否学习 Q 修正
    """

    def __init__(self, n: int, T: int = 10, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None,
                 learnable_Q: bool = False):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            PGDNetLayer(n, init_eta, constraint_type, constraint_params, learnable_Q)
            for _ in range(T)
        ])

    def forward(self, Q: torch.Tensor, c: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        batch_size = c.shape[0] if Q.dim() == 2 else Q.shape[0]
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=c.device)
        intermediates = [x] if return_intermediates else None

        for layer in self.layers:
            x = layer(x, Q, c)
            if return_intermediates:
                intermediates.append(x)

        return (x, intermediates) if return_intermediates else x

    def get_etas(self) -> List[float]:
        return [layer.eta.item() for layer in self.layers]


# ============================================================
# PGD-Momentum-Net — 带 Nesterov 动量
# ============================================================

class PGDMomentumLayer(nn.Module):
    """带 Nesterov 动量的 PGD-Net 单层。

    y_k = x_k + β_k · (x_k - x_{k-1})              # 动量外推
    x_{k+1} = Proj_C(y_k - η_k · (Qy_k + c))       # 梯度步 + 投影

    β_k 是可学习的动量参数。
    """

    def __init__(self, n: int, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(init_eta))
        # 动量参数，初始化为 0
        self.beta = nn.Parameter(torch.tensor(0.0))
        if constraint_params is None:
            constraint_params = {}
        self.projector = create_projector(constraint_type, constraint_params)

    def forward(self, x: torch.Tensor, x_prev: torch.Tensor,
                Q: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Nesterov 动量外推
        y = x + torch.sigmoid(self.beta) * (x - x_prev)

        # 梯度
        if Q.dim() == 2:
            grad = y @ Q.T + c
        else:
            grad = torch.bmm(y.unsqueeze(1), Q).squeeze(1) + c

        # 梯度步 + 投影
        x_new = self.projector(y - self.eta * grad)

        return x_new, x


class PGDMomentumNet(nn.Module):
    """带 Nesterov 动量的 PGD-Net。

    初始化 β=0 退化为基本 PGD-Net，训练后学习最优动量调度。

    Parameters
    ----------
    n : int — 变量维度
    T : int — 展开层数
    """

    def __init__(self, n: int, T: int = 10, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            PGDMomentumLayer(n, init_eta, constraint_type, constraint_params)
            for _ in range(T)
        ])

    def forward(self, Q: torch.Tensor, c: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        batch_size = c.shape[0] if Q.dim() == 2 else Q.shape[0]
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=c.device)
        x_prev = x.clone()
        intermediates = [x] if return_intermediates else None

        for layer in self.layers:
            x, x_prev = layer(x, x_prev, Q, c)
            if return_intermediates:
                intermediates.append(x)

        return (x, intermediates) if return_intermediates else x

    def get_etas(self) -> List[float]:
        return [layer.eta.item() for layer in self.layers]

    def get_momentums(self) -> List[float]:
        return [torch.sigmoid(layer.beta).item() for layer in self.layers]


# ============================================================
# PGD-LearnedQ-Net — 学习 Q 矩阵修正
# ============================================================

class PGDLearnedQLayer(nn.Module):
    """学习 Q 修正的 PGD-Net 单层。

    x_{k+1} = Proj_C(x_k - η_k · ((Q + ΔQ_k)x_k + c))

    ΔQ_k 是可学习的低秩修正矩阵。
    """

    def __init__(self, n: int, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None,
                 correction_rank: int = 5):
        super().__init__()
        self.eta = nn.Parameter(torch.tensor(init_eta))

        # 低秩修正: ΔQ = U @ V^T
        self.U = nn.Parameter(torch.randn(n, correction_rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n, correction_rank) * 0.01)

        if constraint_params is None:
            constraint_params = {}
        self.projector = create_projector(constraint_type, constraint_params)

    def forward(self, x: torch.Tensor, Q: torch.Tensor,
                c: torch.Tensor) -> torch.Tensor:
        # Q_eff = Q + ΔQ
        Q_eff = Q + self.U @ self.V.T

        # 梯度
        if Q.dim() == 2:
            grad = x @ Q_eff.T + c
        else:
            grad = torch.bmm(x.unsqueeze(1), Q_eff.unsqueeze(0)).squeeze(1) + c

        return self.projector(x - self.eta * grad)


class PGDLearnedQNet(nn.Module):
    """学习 Q 矩阵修正的 PGD-Net。

    通过低秩修正 ΔQ_k 自适应调整问题结构。
    """

    def __init__(self, n: int, T: int = 10, init_eta: float = 0.01,
                 constraint_type: str = 'box',
                 constraint_params: Optional[Dict] = None,
                 correction_rank: int = 5):
        super().__init__()
        self.n, self.T = n, T
        self.layers = nn.ModuleList([
            PGDLearnedQLayer(n, init_eta, constraint_type, constraint_params, correction_rank)
            for _ in range(T)
        ])

    def forward(self, Q: torch.Tensor, c: torch.Tensor,
                x0: Optional[torch.Tensor] = None,
                return_intermediates: bool = False) -> torch.Tensor:
        batch_size = c.shape[0] if Q.dim() == 2 else Q.shape[0]
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=c.device)
        intermediates = [x] if return_intermediates else None

        for layer in self.layers:
            x = layer(x, Q, c)
            if return_intermediates:
                intermediates.append(x)

        return (x, intermediates) if return_intermediates else x


# ============================================================
# 工厂函数
# ============================================================

def create_pgd_net(n: int, variant: str = 'basic', T: int = 10,
                   init_eta: float = 0.01, constraint_type: str = 'box',
                   constraint_params: Optional[Dict] = None,
                   correction_rank: int = 5) -> nn.Module:
    """创建 PGD-Net 变体的工厂函数。

    Parameters
    ----------
    n : int — 变量维度
    variant : str — 'basic', 'momentum', 'learned_q'
    """
    variants = {
        'basic': lambda: PGDNet(n, T, init_eta, constraint_type, constraint_params),
        'momentum': lambda: PGDMomentumNet(n, T, init_eta, constraint_type, constraint_params),
        'learned_q': lambda: PGDLearnedQNet(n, T, init_eta, constraint_type, constraint_params, correction_rank),
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")
    return variants[variant]()


# 向后兼容
PGDNetWithInit = PGDNet
