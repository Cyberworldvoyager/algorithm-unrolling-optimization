"""
qp/classical.py — 经典投影梯度下降法 (PGD)

PGD 迭代: x_{k+1} = Proj_C(x_k - η_k(Qx_k + c))
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import project_box, project_simplex
from qp.problem import qp_objective, project_to_constraint


def pgd_box(
    Q: np.ndarray,
    c: np.ndarray,
    lb: float = 0.0,
    ub: float = 1.0,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """投影梯度下降法求解 box 约束 QP。

    min  1/2 x^T Q x + c^T x  s.t.  lb <= x <= ub

    Parameters
    ----------
    Q : np.ndarray
        正定矩阵 (n, n)。
    c : np.ndarray
        线性项 (n,)。
    lb, ub : float
        box 约束上下界。
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
    n = Q.shape[0]
    if x0 is None:
        x = np.clip(np.zeros(n), lb, ub)
    else:
        x = x0.copy()

    # 计算 Lipschitz 常数
    if eta is None:
        L = np.linalg.norm(Q, ord=2)
        eta = 1.0 / L

    objectives = []
    for k in range(max_iter):
        # 梯度
        grad = Q @ x + c

        # 梯度步 + 投影
        x_new = project_box(x - eta * grad, lb, ub)

        # 记录目标函数值
        obj = qp_objective(Q, c, x_new)
        objectives.append(obj)

        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x, objectives


def pgd_simplex(
    Q: np.ndarray,
    c: np.ndarray,
    s: float = 1.0,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """投影梯度下降法求解单纯形约束 QP。

    min  1/2 x^T Q x + c^T x  s.t.  x >= 0, sum(x) = s

    Parameters
    ----------
    Q : np.ndarray
        正定矩阵 (n, n)。
    c : np.ndarray
        线性项 (n,)。
    s : float
        单纯形约束的和。
    x0 : np.ndarray, optional
        初始点。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。
    eta : float, optional
        步长。

    Returns
    -------
    x : np.ndarray
        最优解。
    objectives : list
        每步的目标函数值。
    """
    n = Q.shape[0]
    if x0 is None:
        # 初始化为均匀分布
        x = np.ones(n) / n * s
    else:
        x = x0.copy()

    # 计算 Lipschitz 常数
    if eta is None:
        L = np.linalg.norm(Q, ord=2)
        eta = 1.0 / L

    objectives = []
    for k in range(max_iter):
        # 梯度
        grad = Q @ x + c

        # 梯度步 + 投影
        x_new = project_simplex(x - eta * grad, s)

        # 记录目标函数值
        obj = qp_objective(Q, c, x_new)
        objectives.append(obj)

        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x, objectives


def pgd(
    Q: np.ndarray,
    c: np.ndarray,
    constraint_type: str = 'box',
    constraint_params: Optional[Dict] = None,
    x0: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
    eta: Optional[float] = None,
) -> Tuple[np.ndarray, List[float]]:
    """通用投影梯度下降法。

    Parameters
    ----------
    Q : np.ndarray
        正定矩阵。
    c : np.ndarray
        线性项。
    constraint_type : str
        约束类型: 'box' 或 'simplex'。
    constraint_params : dict, optional
        约束参数。
    x0 : np.ndarray, optional
        初始点。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。
    eta : float, optional
        步长。

    Returns
    -------
    x : np.ndarray
        最优解。
    objectives : list
        目标函数值历史。
    """
    if constraint_params is None:
        constraint_params = {}

    if constraint_type == 'box':
        lb = constraint_params.get('lb', 0.0)
        ub = constraint_params.get('ub', 1.0)
        return pgd_box(Q, c, lb, ub, x0, max_iter, tol, eta)
    elif constraint_type == 'simplex':
        s = constraint_params.get('s', 1.0)
        return pgd_simplex(Q, c, s, x0, max_iter, tol, eta)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def pgd_with_params(
    Q: np.ndarray,
    c: np.ndarray,
    eta_list: List[float],
    constraint_type: str = 'box',
    constraint_params: Optional[Dict] = None,
) -> Tuple[np.ndarray, List[float]]:
    """使用每层不同步长的 PGD (用于对比 PGD-Net)。

    Parameters
    ----------
    Q : np.ndarray
        正定矩阵。
    c : np.ndarray
        线性项。
    eta_list : list
        每层的步长参数。
    constraint_type : str
        约束类型。
    constraint_params : dict, optional
        约束参数。

    Returns
    -------
    x : np.ndarray
        最终解。
    objectives : list
        目标函数值历史。
    """
    if constraint_params is None:
        constraint_params = {}

    n = Q.shape[0]
    x = np.zeros(n)
    T = len(eta_list)

    objectives = []
    for t in range(T):
        eta = eta_list[t]

        # 梯度
        grad = Q @ x + c

        # 梯度步 + 投影
        x_new = project_to_constraint(x - eta * grad, constraint_type, constraint_params)

        # 记录目标函数值
        obj = qp_objective(Q, c, x_new)
        objectives.append(obj)

        x = x_new

    return x, objectives
