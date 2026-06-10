"""
socp/pdhg_cp.py — Stage 1: Learned-PDHG-CP

对应 LISTA-CP 的精神：保留 PDHG 骨架，每层仅学 4 个可学习标量
    η_k (主步长)    softplus 约束 > 0
    τ_k (对偶步长)  softplus 约束 > 0
    β_k (Halpern)   sigmoid 约束 (0, 1)
    θ_k (对偶外推)  sigmoid 约束 (0, 1)  (经典 PDHG = 1; PDQP 也是 1)

骨架 (同 socp/classical.py:pdqp_socp 但每层不同):
    x_md^k   = (1 - β_k) x̄^k + β_k x^k
    x^{k+1}  = Proj_K( x_md^k - η_k (c - A^T y^k) )
    x_bar    = θ_k (x^{k+1} - x^k) + x^{k+1}
    y^{k+1}  = y^k + τ_k (b - A x_bar)
    x̄^{k+1}  = (1 - β_k) x̄^k + β_k x^{k+1}

参数量: 4K 个标量 — 极轻量、完全可解释。
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import project_cone_torch


def _inv_softplus(y: float) -> float:
    """softplus 反函数: x = log(exp(y) - 1)"""
    import math
    return math.log(math.expm1(y))


def _inv_sigmoid(p: float) -> float:
    """sigmoid 反函数: x = log(p / (1-p))"""
    import math
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class LearnedPDHGCP(nn.Module):
    """K 层可学习 PDHG，每层 4 个标量超参。

    Parameters
    ----------
    K : int
        展开层数。
    init_eta : float
        η_0 的初值 (推荐 0.9/||A||)。
    init_tau : float
        τ_0 的初值。
    init_beta_schedule : str or callable
        β 的初值调度，可选:
            'halpern' (默认): β_k = 2/(k+2)
            'constant': β_k = init_beta (恒定)
            callable(k) -> float
    init_theta : float
        θ_0 的初值 (默认 1.0)。
    block_sizes : list of int
        锥结构 K = Π Q^{n_i}；在 forward 时也可覆盖。
    """

    def __init__(
        self,
        K: int = 30,
        init_eta: float = 0.5,
        init_tau: float = 0.5,
        init_beta_schedule='halpern',
        init_beta: float = 0.5,
        init_theta: float = 1.0,
        block_sizes: Optional[List[int]] = None,
    ):
        super().__init__()
        self.K = K
        self.block_sizes = block_sizes

        # 转成预激活的"原始参数" (softplus / sigmoid 之前)
        eta_raw = torch.full((K,), _inv_softplus(init_eta))
        tau_raw = torch.full((K,), _inv_softplus(init_tau))
        theta_raw = torch.full((K,), _inv_sigmoid(init_theta if init_theta < 1.0 else 0.999))

        if callable(init_beta_schedule):
            beta_vals = [float(init_beta_schedule(k)) for k in range(K)]
        elif init_beta_schedule == 'halpern':
            beta_vals = [2.0 / (k + 2.0) for k in range(K)]
        elif init_beta_schedule == 'constant':
            beta_vals = [init_beta] * K
        else:
            raise ValueError(f"unknown init_beta_schedule: {init_beta_schedule}")
        beta_raw = torch.tensor([_inv_sigmoid(v) for v in beta_vals], dtype=torch.float32)

        self.eta_raw = nn.Parameter(eta_raw)
        self.tau_raw = nn.Parameter(tau_raw)
        self.beta_raw = nn.Parameter(beta_raw)
        self.theta_raw = nn.Parameter(theta_raw)

    # ---- 物理参数（带约束） ----
    @property
    def eta(self):   return torch.nn.functional.softplus(self.eta_raw)
    @property
    def tau(self):   return torch.nn.functional.softplus(self.tau_raw)
    @property
    def beta(self):  return torch.sigmoid(self.beta_raw)
    @property
    def theta(self): return torch.sigmoid(self.theta_raw)

    def schedules(self) -> Dict[str, list]:
        return {
            'eta':   self.eta.detach().cpu().tolist(),
            'tau':   self.tau.detach().cpu().tolist(),
            'beta':  self.beta.detach().cpu().tolist(),
            'theta': self.theta.detach().cpu().tolist(),
        }

    def forward(
        self,
        A: torch.Tensor,             # (batch, m, N)
        b: torch.Tensor,             # (batch, m)
        c: torch.Tensor,             # (batch, N)
        block_sizes: Optional[List[int]] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向 = K 步 PDHG 展开。

        Returns
        -------
        x : (batch, N) 主变量
        y : (batch, m) 对偶变量
        traj (optional) : list of (x, y) of length K+1 包含初始 0
        """
        block = block_sizes if block_sizes is not None else self.block_sizes
        assert block is not None, "block_sizes must be provided in __init__ or forward"

        batch, m, N = A.shape
        device = A.device

        x = torch.zeros(batch, N, device=device, dtype=A.dtype)
        y = torch.zeros(batch, m, device=device, dtype=A.dtype)
        x_avg = torch.zeros_like(x)

        traj = [(x.clone(), y.clone())] if return_trajectory else None

        eta = self.eta
        tau = self.tau
        beta = self.beta
        theta = self.theta

        for k in range(self.K):
            # 主步: x^{k+1} = Proj_K( x_md - η_k (c - A^T y) )
            Aty = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)  # (batch, N)
            x_md = (1.0 - beta[k]) * x_avg + beta[k] * x
            x_new = project_cone_torch(x_md - eta[k] * (c - Aty), block)

            # 对偶外推 + 对偶步
            x_bar = theta[k] * (x_new - x) + x_new
            Ax_bar = torch.bmm(A, x_bar.unsqueeze(-1)).squeeze(-1)  # (batch, m)
            y = y + tau[k] * (b - Ax_bar)

            # 平均更新
            x_avg = (1.0 - beta[k]) * x_avg + beta[k] * x_new
            x = x_new

            if return_trajectory:
                traj.append((x.clone(), y.clone()))

        return (x, y, traj) if return_trajectory else (x, y)


# ============================================================
# 损失函数
# ============================================================

def supervised_mse(x_pred: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
    """监督 MSE: ||x_hat - x*||^2 / batch."""
    return ((x_pred - x_star) ** 2).sum(dim=-1).mean()


def kkt_loss(
    x: torch.Tensor, y: torch.Tensor,
    A: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
    block_sizes: List[int],
    w_primal: float = 1.0, w_dual: float = 1.0, w_gap: float = 0.1,
) -> torch.Tensor:
    """无监督 KKT 损失:
        w_p ||Ax-b||^2 + w_d ||s - Proj_K(s)||^2 + w_g |<s, x>|
    其中 s = c - A^T y。
    """
    Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
    Aty = torch.bmm(A.transpose(1, 2), y.unsqueeze(-1)).squeeze(-1)
    s = c - Aty
    s_proj = project_cone_torch(s, block_sizes)
    primal = ((Ax - b) ** 2).sum(dim=-1).mean()
    dual = ((s - s_proj) ** 2).sum(dim=-1).mean()
    gap = ((s * x).sum(dim=-1).abs()).mean()
    return w_primal * primal + w_dual * dual + w_gap * gap
