"""
low_rank/classical.py — 经典 ADMM 求解器

ADMM 迭代:
X_{k+1} = D_τ(Z_k - Y_k/ρ)        # 奇异值阈值化
Z_{k+1} = P_Ω(M) + P_Ωᶜ(X_{k+1} + Y_k/ρ)  # 投影
Y_{k+1} = Y_k + ρ(X_{k+1} - Z_{k+1})        # 对偶更新
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import svd_threshold, nuclear_norm


def admm_matrix_completion(
    M_observed: np.ndarray,
    mask: np.ndarray,
    rho: float = 1.0,
    tau: Optional[float] = None,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """ADMM 求解矩阵补全问题。

    min ||X||_*  s.t.  P_Ω(X) = P_Ω(M)

    Parameters
    ----------
    M_observed : np.ndarray
        观测矩阵 (m, n)，未观测位置为 0。
    mask : np.ndarray
        观测掩码 (m, n)。
    rho : float
        ADMM 惩罚参数。
    tau : float, optional
        奇异值阈值化阈值。若为 None 则使用 1/ρ。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    Returns
    -------
    X : np.ndarray
        恢复的低秩矩阵。
    history : list
        迭代历史记录。
    """
    m, n = M_observed.shape
    if tau is None:
        tau = 1.0 / rho

    # 初始化
    X = np.zeros((m, n))
    Z = np.zeros((m, n))
    Y = np.zeros((m, n))

    history = []
    mask_c = ~mask  # 未观测位置

    for k in range(max_iter):
        # X 更新: 奇异值软阈值化
        X_new = svd_threshold(Z - Y / rho, tau)

        # Z 更新: 投影
        # Z = P_Ω(M) + P_Ωᶜ(X + Y/ρ)
        Z_new = M_observed * mask + (X_new + Y / rho) * mask_c

        # Y 更新: 对偶变量
        Y_new = Y + rho * (X_new - Z_new)

        # 计算残差
        primal_res = np.linalg.norm(X_new - Z_new)
        dual_res = np.linalg.norm(rho * (Z_new - Z))

        # 记录历史
        obj = nuclear_norm(X_new)
        history.append({
            'iteration': k + 1,
            'objective': obj,
            'primal_residual': primal_res,
            'dual_residual': dual_res,
            'rank': np.linalg.matrix_rank(X_new),
        })

        # 收敛判断
        if primal_res < tol and dual_res < tol:
            X = X_new
            Z = Z_new
            Y = Y_new
            break

        X = X_new
        Z = Z_new
        Y = Y_new

    return X, history


def admm_with_params(
    M_observed: np.ndarray,
    mask: np.ndarray,
    rho_list: List[float],
    tau_list: List[float],
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """使用每层不同参数的 ADMM (用于对比 ADMM-Net)。

    Parameters
    ----------
    M_observed : np.ndarray
        观测矩阵。
    mask : np.ndarray
        观测掩码。
    rho_list : list
        每层的 ρ 参数。
    tau_list : list
        每层的 τ 参数。

    Returns
    -------
    X : np.ndarray
        恢复的低秩矩阵。
    history : list
        迭代历史。
    """
    m, n = M_observed.shape
    T = len(rho_list)

    # 初始化
    X = np.zeros((m, n))
    Z = np.zeros((m, n))
    Y = np.zeros((m, n))

    history = []
    mask_c = ~mask

    for t in range(T):
        rho = rho_list[t]
        tau = tau_list[t]

        # X 更新
        X_new = svd_threshold(Z - Y / rho, tau)

        # Z 更新
        Z_new = M_observed * mask + (X_new + Y / rho) * mask_c

        # Y 更新
        Y_new = Y + rho * (X_new - Z_new)

        # 记录
        obj = nuclear_norm(X_new)
        history.append({
            'iteration': t + 1,
            'objective': obj,
            'rank': np.linalg.matrix_rank(X_new),
        })

        X = X_new
        Z = Z_new
        Y = Y_new

    return X, history


def soft_impute(
    M_observed: np.ndarray,
    mask: np.ndarray,
    lam: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """Soft-Impute 算法 (作为 baseline)。

    迭代: X_{k+1} = S_λ(P_Ω(M) + P_Ωᶜ(X_k))

    Parameters
    ----------
    M_observed : np.ndarray
        观测矩阵。
    mask : np.ndarray
        观测掩码。
    lam : float
        正则化参数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    Returns
    -------
    X : np.ndarray
        恢复矩阵。
    objectives : list
        目标函数值历史。
    """
    m, n = M_observed.shape
    mask_c = ~mask
    X = np.zeros((m, n))

    objectives = []
    for k in range(max_iter):
        # 填充
        Z = M_observed * mask + X * mask_c

        # 软阈值化
        X_new = svd_threshold(Z, lam)

        # 记录目标函数值
        obj = nuclear_norm(X_new) + 0.5 * np.sum((apply_mask(X_new, mask) - M_observed) ** 2)
        objectives.append(obj)

        # 收敛判断
        if np.linalg.norm(X_new - X) < tol:
            X = X_new
            break
        X = X_new

    return X, objectives


def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """应用观测掩码。"""
    return X * mask
