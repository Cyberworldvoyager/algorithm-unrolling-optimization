"""
lasso/classical.py — 经典 ISTA/FISTA 求解器

ISTA: 近端梯度下降法
FISTA: 快速近端梯度下降法 (Nesterov 加速)
"""

import numpy as np
from typing import Tuple, Optional, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import proximal_l1


def ista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """ISTA 求解 LASSO 问题。

    迭代公式: x_{k+1} = SoftThreshold(x_k - η A^T(Ax_k - b), ηλ)

    Parameters
    ----------
    A : np.ndarray
        测量矩阵 (m, n)。
    b : np.ndarray
        观测向量 (m,)。
    lam : float
        正则化参数 λ。
    x0 : np.ndarray, optional
        初始点。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。
    eta : float, optional
        步长。若为 None 则使用 1/L (L = ||A^T A||_2)。

    Returns
    -------
    x : np.ndarray
        最优解。
    objectives : list
        每步的目标函数值。
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # 计算 Lipschitz 常数 L = ||A^T A||_2
    if eta is None:
        L = np.linalg.norm(A.T @ A, ord=2)
        eta = 1.0 / L

    # 预计算 A^T A 和 A^T b
    AtA = A.T @ A
    Atb = A.T @ b

    objectives = []
    for k in range(max_iter):
        # 梯度步
        grad = AtA @ x - Atb
        x_new = proximal_l1(x - eta * grad, eta * lam)

        # 记录目标函数值
        obj = 0.5 * np.linalg.norm(A @ x_new - b) ** 2 + lam * np.sum(np.abs(x_new))
        objectives.append(obj)

        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x, objectives


def fista(
    A: np.ndarray,
    b: np.ndarray,
    lam: float,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """FISTA 求解 LASSO 问题 (Nesterov 加速)。

    Parameters
    ----------
    A : np.ndarray
        测量矩阵 (m, n)。
    b : np.ndarray
        观测向量 (m,)。
    lam : float
        正则化参数 λ。
    x0 : np.ndarray, optional
        初始点。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。
    eta : float, optional
        步长。若为 None 则使用 1/L。

    Returns
    -------
    x : np.ndarray
        最优解。
    objectives : list
        每步的目标函数值。
    """
    m, n = A.shape
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    # 计算 Lipschitz 常数
    if eta is None:
        L = np.linalg.norm(A.T @ A, ord=2)
        eta = 1.0 / L

    # 预计算
    AtA = A.T @ A
    Atb = A.T @ b

    # FISTA 初始化
    y = x.copy()
    t = 1.0

    objectives = []
    for k in range(max_iter):
        # 梯度步
        grad = AtA @ y - Atb
        x_new = proximal_l1(y - eta * grad, eta * lam)

        # Nesterov 更新
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)

        # 记录目标函数值
        obj = 0.5 * np.linalg.norm(A @ x_new - b) ** 2 + lam * np.sum(np.abs(x_new))
        objectives.append(obj)

        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break

        x = x_new
        t = t_new

    return x, objectives


def ista_unrolled(
    A: np.ndarray,
    b: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    theta: np.ndarray,
    T: int,
) -> Tuple[np.ndarray, List[float]]:
    """展开的 ISTA (LISTA 的前向传播，用于对比)。

    使用可学习参数 W1, W2, theta 进行 T 步迭代。

    Parameters
    ----------
    A : np.ndarray
        测量矩阵 (m, n)。
    b : np.ndarray
        观测向量 (m,)。
    W1 : np.ndarray
        可学习参数 (n, m)。
    W2 : np.ndarray
        可学习参数 (n, n)。
    theta : np.ndarray
        可学习阈值参数 (n,) 或标量。
    T : int
        展开层数。

    Returns
    -------
    x : np.ndarray
        最终解。
    intermediates : list
        每层的中间解。
    """
    n = W2.shape[0]
    x = np.zeros(n)
    intermediates = [x.copy()]

    for t in range(T):
        # x_{t+1} = SoftThreshold(W1 @ b + W2 @ x_t, theta)
        z = W1 @ b + W2 @ x
        x = np.sign(z) * np.maximum(np.abs(z) - theta, 0)
        intermediates.append(x.copy())

    return x, intermediates
