"""
qp/problem.py — 二次规划问题定义与数据生成

QP 问题: min_x  1/2 x^T Q x + c^T x  s.t.  x ∈ C
C 为 box 约束或单纯形约束
"""

import numpy as np
from typing import Tuple, Optional, Dict, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import project_box, project_simplex, generate_spd_matrix


def generate_qp_data(
    n: int = 50,
    constraint_type: str = 'box',
    kappa: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """生成二次规划问题数据。

    Parameters
    ----------
    n : int
        变量维度。
    constraint_type : str
        约束类型: 'box' 或 'simplex'。
    kappa : float, optional
        矩阵条件数。
    seed : int, optional
        随机种子。

    Returns
    -------
    data : dict
        包含 Q, c, constraint_type 等。
    """
    rng = np.random.RandomState(seed)

    # 生成对称正定矩阵 Q
    Q = generate_spd_matrix(n, kappa, seed)

    # 生成线性项 c
    c = rng.randn(n)

    # 约束参数
    if constraint_type == 'box':
        lb = rng.uniform(-1, 0, n)
        ub = rng.uniform(0, 1, n)
        constraint_params = {'lb': lb, 'ub': ub}
    elif constraint_type == 'simplex':
        constraint_params = {'s': 1.0}
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    return {
        'Q': Q,
        'c': c,
        'n': n,
        'constraint_type': constraint_type,
        'constraint_params': constraint_params,
    }


def qp_objective(Q: np.ndarray, c: np.ndarray, x: np.ndarray) -> float:
    """计算 QP 目标函数值: 1/2 x^T Q x + c^T x。"""
    return float(0.5 * x @ Q @ x + c @ x)


def project_to_constraint(
    x: np.ndarray,
    constraint_type: str,
    constraint_params: Dict,
) -> np.ndarray:
    """投影到约束集。

    Parameters
    ----------
    x : np.ndarray
        输入点。
    constraint_type : str
        约束类型。
    constraint_params : dict
        约束参数。

    Returns
    -------
    x_proj : np.ndarray
        投影后的点。
    """
    if constraint_type == 'box':
        lb = constraint_params.get('lb', 0.0)
        ub = constraint_params.get('ub', 1.0)
        return np.clip(x, lb, ub)
    elif constraint_type == 'simplex':
        s = constraint_params.get('s', 1.0)
        return project_simplex(x, s)
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def compute_optimal_solution(
    Q: np.ndarray,
    c: np.ndarray,
    constraint_type: str,
    constraint_params: Dict,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """计算 QP 问题的近似最优解 (使用投影梯度法)。

    Parameters
    ----------
    Q : np.ndarray
        正定矩阵。
    c : np.ndarray
        线性项。
    constraint_type : str
        约束类型。
    constraint_params : dict
        约束参数。
    max_iter : int
        最大迭代次数。
    tol : float
        收敛容忍度。

    Returns
    -------
    x_opt : np.ndarray
        近似最优解。
    """
    n = Q.shape[0]
    x = np.zeros(n)

    # 计算步长
    L = np.linalg.norm(Q, ord=2)
    eta = 1.0 / L

    for k in range(max_iter):
        # 梯度
        grad = Q @ x + c

        # 梯度步 + 投影
        x_new = project_to_constraint(x - eta * grad, constraint_type, constraint_params)

        # 收敛判断
        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new

    return x


def generate_batch_data(
    batch_size: int,
    n: int = 50,
    constraint_type: str = 'box',
    kappa: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """生成一批 QP 数据。

    Returns
    -------
    Q_batch : np.ndarray
        正定矩阵 (batch_size, n, n)。
    c_batch : np.ndarray
        线性项 (batch_size, n)。
    x_opt_batch : np.ndarray
        最优解 (batch_size, n)。
    constraint_params_batch : dict
        约束参数。
    """
    rng = np.random.RandomState(seed)
    Q_list, c_list, x_opt_list = [], [], []

    for _ in range(batch_size):
        data = generate_qp_data(n, constraint_type, kappa, seed=rng.randint(1e6))
        Q = data['Q']
        c = data['c']
        constraint_params = data['constraint_params']

        # 计算最优解
        x_opt = compute_optimal_solution(Q, c, constraint_type, constraint_params)

        Q_list.append(Q)
        c_list.append(c)
        x_opt_list.append(x_opt)

    return np.array(Q_list), np.array(c_list), np.array(x_opt_list), data['constraint_params']
