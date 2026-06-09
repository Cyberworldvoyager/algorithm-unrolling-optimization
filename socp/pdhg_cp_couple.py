"""
socp/pdhg_cp_couple.py — Stage 2: Learned-PDHG-CP-Couple

在 Stage 1 (4K 标量超参) 之上，进一步引入"耦合矩阵" Δ_p, Δ_d 残差化:
    B_p^k = A^T + Δ_p^k       (用于主步 -A^T y -> -B_p y)
    B_d^k = A   + Δ_d^k       (用于对偶步 A x_tilde -> B_d x_tilde)

初始化 Δ_p = 0, Δ_d = 0  =>  Stage 2 退化为 Stage 1，正向数值完全一致。
这复刻了 LISTA-CP 的精神: "开局即经典算法，训练后超越"。

迭代式 (与 PDQP 模板同构):
    x_md^k   = (1 - β_k) x̄^k + β_k x^k
    g_x      = c - B_p^k @ y^k                                # 主梯度
    x^{k+1}  = Proj_K( x_md^k - η_k g_x )
    x_bar    = θ_k (x^{k+1} - x^k) + x^{k+1}
    g_y      = b - B_d^k @ x_bar                              # 对偶残差
    y^{k+1}  = y^k + τ_k g_y
    x̄^{k+1}  = (1 - β_k) x̄^k + β_k x^{k+1}

参数共享 mode:
    'independent' (默认): 每层独立 (Δ_p^k, Δ_d^k), 参数量 2K·m·N
    'shared':       全部层共享 (Δ_p, Δ_d),           参数量 2·m·N
    'symmetric':    每层独立, 但强制 Δ_d^k = (Δ_p^k)^T (对偶对称), 参数量 K·m·N
    'shared_symmetric': 全部层共享 + 对偶对称,        参数量 m·N

正则:
    coupling_reg() 返回 Σ_k (||Δ_p^k||_F^2 + ||Δ_d^k||_F^2) 用于训练时加入损失。
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import project_cone_torch
from socp.pdhg_cp import _inv_softplus, _inv_sigmoid


class LearnedPDHGCPCouple(nn.Module):
    """K 层可学习 PDHG，每层 4 个标量 + 一对耦合矩阵残差 (Δ_p, Δ_d)。

    Parameters
    ----------
    K : int                       展开层数
    A_template : torch.Tensor    (m, N) 用于初始化的"代表性" A，决定 Δ 的尺寸；
                                  当训练/推理时传入的 A 可与之不同（推荐相同尺寸）。
    init_eta, init_tau : float    标量步长初值
    init_beta_schedule : str      'halpern' / 'constant'
    init_beta : float             constant 时的取值
    init_theta : float            θ 初值 (默认 0.999, sigmoid 域可学)
    coupling_mode : str           'independent' | 'shared' | 'symmetric' | 'shared_symmetric'
    block_sizes : list of int     锥结构
    """

    def __init__(
        self,
        K: int,
        A_template: torch.Tensor,
        block_sizes: List[int],
        init_eta: float = 0.5,
        init_tau: float = 0.5,
        init_beta_schedule: str = 'halpern',
        init_beta: float = 0.5,
        init_theta: float = 0.999,
        coupling_mode: str = 'independent',
    ):
        super().__init__()
        assert A_template.dim() == 2
        m, N = A_template.shape
        self.K = K
        self.m, self.N = m, N
        self.block_sizes = block_sizes
        self.coupling_mode = coupling_mode

        # ---------- 标量超参 (复用 Stage 1 的参数化) ----------
        self.eta_raw = nn.Parameter(torch.full((K,), _inv_softplus(init_eta)))
        self.tau_raw = nn.Parameter(torch.full((K,), _inv_softplus(init_tau)))
        self.theta_raw = nn.Parameter(torch.full(
            (K,), _inv_sigmoid(init_theta if init_theta < 1.0 else 0.999)))

        if init_beta_schedule == 'halpern':
            beta_vals = [2.0 / (k + 2.0) for k in range(K)]
        elif init_beta_schedule == 'constant':
            beta_vals = [init_beta] * K
        elif callable(init_beta_schedule):
            beta_vals = [float(init_beta_schedule(k)) for k in range(K)]
        else:
            raise ValueError(f"unknown init_beta_schedule: {init_beta_schedule}")
        self.beta_raw = nn.Parameter(torch.tensor(
            [_inv_sigmoid(v) for v in beta_vals], dtype=torch.float32))

        # ---------- 耦合矩阵残差 Δ_p (形状 N×m), Δ_d (形状 m×N) ----------
        # 初值为 0 => B_p = A^T, B_d = A，正向等价于 Stage 1
        if coupling_mode == 'independent':
            self.delta_p = nn.Parameter(torch.zeros(K, N, m))
            self.delta_d = nn.Parameter(torch.zeros(K, m, N))
        elif coupling_mode == 'shared':
            self.delta_p = nn.Parameter(torch.zeros(N, m))
            self.delta_d = nn.Parameter(torch.zeros(m, N))
        elif coupling_mode == 'symmetric':
            # 只学 Δ_p，Δ_d = (Δ_p)^T (按层独立)
            self.delta_p = nn.Parameter(torch.zeros(K, N, m))
            self.delta_d = None  # tied
        elif coupling_mode == 'shared_symmetric':
            self.delta_p = nn.Parameter(torch.zeros(N, m))
            self.delta_d = None
        else:
            raise ValueError(f"unknown coupling_mode: {coupling_mode}")

    # ---------- 物理参数访问 ----------
    @property
    def eta(self):   return torch.nn.functional.softplus(self.eta_raw)
    @property
    def tau(self):   return torch.nn.functional.softplus(self.tau_raw)
    @property
    def beta(self):  return torch.sigmoid(self.beta_raw)
    @property
    def theta(self): return torch.sigmoid(self.theta_raw)

    def get_delta(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回第 k 层使用的 (Δ_p, Δ_d)，自动按 mode 处理共享/对称。"""
        if self.coupling_mode == 'independent':
            return self.delta_p[k], self.delta_d[k]
        if self.coupling_mode == 'shared':
            return self.delta_p, self.delta_d
        if self.coupling_mode == 'symmetric':
            dp = self.delta_p[k]
            return dp, dp.transpose(-1, -2)
        if self.coupling_mode == 'shared_symmetric':
            dp = self.delta_p
            return dp, dp.transpose(-1, -2)
        raise RuntimeError(self.coupling_mode)

    def coupling_reg(self) -> torch.Tensor:
        """Σ_k (||Δ_p^k||_F^2 + ||Δ_d^k||_F^2)。"""
        if self.coupling_mode == 'independent':
            return (self.delta_p ** 2).sum() + (self.delta_d ** 2).sum()
        if self.coupling_mode == 'shared':
            return self.K * ((self.delta_p ** 2).sum() + (self.delta_d ** 2).sum())
        if self.coupling_mode == 'symmetric':
            return 2.0 * (self.delta_p ** 2).sum()
        if self.coupling_mode == 'shared_symmetric':
            return 2.0 * self.K * (self.delta_p ** 2).sum()
        raise RuntimeError(self.coupling_mode)

    def delta_norms(self) -> Dict[str, list]:
        """每层 (||Δ_p||_F, ||Δ_d||_F) 用于诊断 / 可视化几何漂移。"""
        norms_p, norms_d = [], []
        for k in range(self.K):
            dp, dd = self.get_delta(k)
            norms_p.append(float(torch.linalg.norm(dp).item()))
            norms_d.append(float(torch.linalg.norm(dd).item()))
        return {'delta_p': norms_p, 'delta_d': norms_d}

    def schedules(self) -> Dict[str, list]:
        return {
            'eta':   self.eta.detach().cpu().tolist(),
            'tau':   self.tau.detach().cpu().tolist(),
            'beta':  self.beta.detach().cpu().tolist(),
            'theta': self.theta.detach().cpu().tolist(),
        }

    # ---------- 前向 ----------
    def forward(
        self,
        A: torch.Tensor,             # (batch, m, N)
        b: torch.Tensor,             # (batch, m)
        c: torch.Tensor,             # (batch, N)
        block_sizes: Optional[List[int]] = None,
        return_trajectory: bool = False,
    ):
        block = block_sizes if block_sizes is not None else self.block_sizes
        batch, m, N = A.shape
        assert m == self.m and N == self.N, \
            f"input A shape ({m},{N}) mismatch model ({self.m},{self.N})"
        device = A.device

        x = torch.zeros(batch, N, device=device, dtype=A.dtype)
        y = torch.zeros(batch, m, device=device, dtype=A.dtype)
        x_avg = torch.zeros_like(x)
        traj = [(x.clone(), y.clone())] if return_trajectory else None

        eta, tau, beta, theta = self.eta, self.tau, self.beta, self.theta

        # A^T: (batch, N, m)
        At = A.transpose(1, 2)

        for k in range(self.K):
            dp, dd = self.get_delta(k)  # (N,m), (m,N)
            # B_p y = (A^T + Δ_p) y  -> 按 batch 计算
            Bpy = torch.bmm(At + dp.unsqueeze(0).expand(batch, -1, -1),
                            y.unsqueeze(-1)).squeeze(-1)               # (batch, N)

            # 主步
            x_md = (1.0 - beta[k]) * x_avg + beta[k] * x
            x_new = project_cone_torch(x_md - eta[k] * (c - Bpy), block)

            # 对偶外推
            x_bar = theta[k] * (x_new - x) + x_new

            # B_d x_bar = (A + Δ_d) x_bar
            Bdx = torch.bmm(A + dd.unsqueeze(0).expand(batch, -1, -1),
                            x_bar.unsqueeze(-1)).squeeze(-1)            # (batch, m)
            y = y + tau[k] * (b - Bdx)

            # Halpern 平均
            x_avg = (1.0 - beta[k]) * x_avg + beta[k] * x_new
            x = x_new

            if return_trajectory:
                traj.append((x.clone(), y.clone()))

        if return_trajectory:
            return x, y, traj
        return x, y


# ============================================================
# 训练损失 (含耦合正则)
# ============================================================

def supervised_mse_with_coupling(
    x_pred: torch.Tensor, x_star: torch.Tensor,
    model: LearnedPDHGCPCouple, lam_couple: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (total, mse, reg)。"""
    mse = ((x_pred - x_star) ** 2).sum(dim=-1).mean()
    if lam_couple > 0:
        reg = lam_couple * model.coupling_reg()
        return mse + reg, mse, reg
    z = torch.zeros((), device=x_pred.device, dtype=x_pred.dtype)
    return mse, mse, z


def count_params(model: LearnedPDHGCPCouple) -> Dict[str, int]:
    n_scalar = model.eta_raw.numel() + model.tau_raw.numel() + \
               model.beta_raw.numel() + model.theta_raw.numel()
    n_couple = model.delta_p.numel()
    if model.delta_d is not None:
        n_couple += model.delta_d.numel()
    return {'scalar': n_scalar, 'coupling': n_couple, 'total': n_scalar + n_couple}
