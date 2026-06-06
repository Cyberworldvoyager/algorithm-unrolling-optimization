"""
low_rank/admm_net.py — ADMM-Net 及其变体

实现以下展开网络:
1. ADMMNet — 基本 ADMM 展开 (Sun et al., 2016)
2. ADMMNet-v2 — 带可学习线性变换的 ADMM-Net
3. SoftImputeNet — Soft-Impute 展开 (作为替代 baseline)

关键改进:
- 修复原版 alpha 参数未使用的问题
- 增加可学习的线性变换层
- 支持低秩近似 SVD (torch.svd_lowrank)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict


class SVDThreshold(nn.Module):
    """可学习的奇异值软阈值化层。

    D_τ(X) = U diag(max(σ_i - τ, 0)) V^T

    Parameters
    ----------
    init_tau : float — 阈值初始值
    use_lowrank : bool — 是否使用低秩近似 SVD
    max_rank : int — 低秩近似的最大秩
    """

    def __init__(self, init_tau: float = 0.1, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau))
        self.use_lowrank = use_lowrank
        self.max_rank = max_rank

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_lowrank and X.shape[-1] > self.max_rank:
            # 使用低秩近似 (更快但近似)
            k = min(self.max_rank, min(X.shape[-2:]) - 1)
            U, S, Vh = torch.svd_lowrank(X, q=k)
            S_thresh = torch.maximum(S - self.tau, torch.zeros_like(S))
            return (U * S_thresh.unsqueeze(-2)) @ Vh
        else:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            S_thresh = torch.maximum(S - self.tau, torch.zeros_like(S))
            return U @ torch.diag_embed(S_thresh) @ Vh


class SVDThresholdPerSingular(nn.Module):
    """逐奇异值可学习的阈值化。"""

    def __init__(self, min_dim: int, init_tau: float = 0.1):
        super().__init__()
        self.tau = nn.Parameter(torch.full((min_dim,), init_tau))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        S_thresh = torch.maximum(S - self.tau, torch.zeros_like(S))
        return U @ torch.diag_embed(S_thresh) @ Vh


# ============================================================
# ADMM-Net v1 — 基本版本 (Sun et al., 2016)
# ============================================================

class ADMMNetLayer(nn.Module):
    """ADMM-Net 单层。

    展开一步 ADMM:
        X_{k+1} = D_τ(Z_k - Y_k/ρ)           # 近端步
        Z_{k+1} = P_Ω(M) + P_Ωᶜ(X_{k+1} + Y_k/ρ)  # 投影
        Y_{k+1} = Y_k + ρ(X_{k+1} - Z_{k+1})        # 对偶更新

    可学习参数: τ (阈值), ρ (惩罚), α (残差权重)
    """

    def __init__(self, m: int, n: int, init_tau: float = 0.1,
                 init_rho: float = 1.0, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.threshold = SVDThreshold(init_tau, use_lowrank, max_rank)
        self.rho = nn.Parameter(torch.tensor(init_rho))
        # α 是对偶更新步长的缩放因子
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                X: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_c = 1.0 - mask

        # X 更新: 奇异值软阈值化
        X_new = self.threshold(Z - Y / self.rho)

        # Z 更新: 投影到观测约束
        Z_new = M_observed * mask + (X_new + Y / self.rho) * mask_c

        # Y 更新: 对偶变量 (使用可学习的 α)
        Y_new = Y + self.alpha * self.rho * (X_new - Z_new)

        return X_new, Z_new, Y_new


class ADMMNet(nn.Module):
    """ADMM-Net: T 层 ADMM 展开。

    Parameters
    ----------
    m, n : int — 矩阵维度
    T : int — 展开层数
    use_lowrank : bool — 是否使用低秩近似 SVD
    max_rank : int — 低秩近似的最大秩
    """

    def __init__(self, m: int, n: int, T: int = 10, init_tau: float = 0.1,
                 init_rho: float = 1.0, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            ADMMNetLayer(m, n, init_tau, init_rho, use_lowrank, max_rank)
            for _ in range(T)
        ])

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                return_intermediates: bool = False) -> torch.Tensor:
        X = torch.zeros_like(M_observed)
        Z = torch.zeros_like(M_observed)
        Y = torch.zeros_like(M_observed)
        intermediates = [(X, Z, Y)] if return_intermediates else None

        for layer in self.layers:
            X, Z, Y = layer(M_observed, mask, X, Z, Y)
            if return_intermediates:
                intermediates.append((X, Z, Y))

        return (X, intermediates) if return_intermediates else X

    def get_parameters(self) -> Dict[str, List[float]]:
        taus, rhos, alphas = [], [], []
        for layer in self.layers:
            taus.append(layer.threshold.tau.item())
            rhos.append(layer.rho.item())
            alphas.append(layer.alpha.item())
        return {'tau': taus, 'rho': rhos, 'alpha': alphas}


# ============================================================
# ADMM-Net v2 — 带可学习线性变换
# ============================================================

class ADMMNetV2Layer(nn.Module):
    """ADMM-Net v2 单层。

    在基本 ADMM-Net 基础上增加可学习的线性变换:
        X_{k+1} = D_τ(L₁(Z_k) - L₂(Y_k/ρ))

    其中 L₁, L₂ 是可学习的线性变换，提供更强的表达能力。
    """

    def __init__(self, m: int, n: int, init_tau: float = 0.1,
                 init_rho: float = 1.0, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.threshold = SVDThreshold(init_tau, use_lowrank, max_rank)
        self.rho = nn.Parameter(torch.tensor(init_rho))
        self.alpha = nn.Parameter(torch.tensor(1.0))

        # 可学习的线性变换 (用 1x1 卷积实现矩阵逐元素变换)
        self.L1 = nn.Conv2d(1, 1, 1, bias=False)
        self.L2 = nn.Conv2d(1, 1, 1, bias=False)
        nn.init.eye_(self.L1.weight.view(m, n)[:min(m,n), :min(m,n)])
        nn.init.eye_(self.L2.weight.view(m, n)[:min(m,n), :min(m,n)])

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                X: torch.Tensor, Z: torch.Tensor, Y: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_c = 1.0 - mask

        # 可学习线性变换
        Z_t = self.L1(Z.unsqueeze(1)).squeeze(1)
        Y_rho_t = self.L2((Y / self.rho).unsqueeze(1)).squeeze(1)

        # X 更新
        X_new = self.threshold(Z_t - Y_rho_t)

        # Z 更新
        Z_new = M_observed * mask + (X_new + Y / self.rho) * mask_c

        # Y 更新
        Y_new = Y + self.alpha * self.rho * (X_new - Z_new)

        return X_new, Z_new, Y_new


class ADMMNetV2(nn.Module):
    """ADMM-Net v2: 带可学习线性变换的 ADMM-Net。

    相比 v1 增加了 L₁, L₂ 线性变换，表达能力更强。
    """

    def __init__(self, m: int, n: int, T: int = 10, init_tau: float = 0.1,
                 init_rho: float = 1.0, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            ADMMNetV2Layer(m, n, init_tau, init_rho, use_lowrank, max_rank)
            for _ in range(T)
        ])

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                return_intermediates: bool = False) -> torch.Tensor:
        X = torch.zeros_like(M_observed)
        Z = torch.zeros_like(M_observed)
        Y = torch.zeros_like(M_observed)
        intermediates = [(X, Z, Y)] if return_intermediates else None

        for layer in self.layers:
            X, Z, Y = layer(M_observed, mask, X, Z, Y)
            if return_intermediates:
                intermediates.append((X, Z, Y))

        return (X, intermediates) if return_intermediates else X

    def get_parameters(self) -> Dict[str, List[float]]:
        taus, rhos, alphas = [], [], []
        for layer in self.layers:
            taus.append(layer.threshold.tau.item())
            rhos.append(layer.rho.item())
            alphas.append(layer.alpha.item())
        return {'tau': taus, 'rho': rhos, 'alpha': alphas}


# ============================================================
# SoftImputeNet — Soft-Impute 展开
# ============================================================

class SoftImputeLayer(nn.Module):
    """Soft-Impute 单层。

    X_{k+1} = D_λ(P_Ω(M) + P_Ωᶜ(X_k))

    只有一个可学习参数: 阈值 λ。
    """

    def __init__(self, init_lambda: float = 0.1, use_lowrank: bool = False,
                 max_rank: int = 20):
        super().__init__()
        self.threshold = SVDThreshold(init_lambda, use_lowrank, max_rank)

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                X: torch.Tensor) -> torch.Tensor:
        # 填充: 用观测值替换已知位置
        Z = M_observed * mask + X * (1.0 - mask)
        # 软阈值化
        return self.threshold(Z)


class SoftImputeNet(nn.Module):
    """Soft-Impute 展开网络。

    作为 ADMM-Net 的替代 baseline，只展开 Soft-Impute 算法。
    """

    def __init__(self, m: int, n: int, T: int = 10, init_lambda: float = 0.1,
                 use_lowrank: bool = False, max_rank: int = 20):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            SoftImputeLayer(init_lambda, use_lowrank, max_rank)
            for _ in range(T)
        ])

    def forward(self, M_observed: torch.Tensor, mask: torch.Tensor,
                return_intermediates: bool = False) -> torch.Tensor:
        X = torch.zeros_like(M_observed)
        intermediates = [X] if return_intermediates else None

        for layer in self.layers:
            X = layer(M_observed, mask, X)
            if return_intermediates:
                intermediates.append(X)

        return (X, intermediates) if return_intermediates else X


# 向后兼容
ADMMNetWithInit = ADMMNet
