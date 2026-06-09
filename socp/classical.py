"""
socp/classical.py — SOCP 的裸 PDHG 与 PDQP 风格基线求解器 (无学习)

问题: min c^T x  s.t.  A x = b,  x ∈ K (SOC 锥积)
Lagrangian: L(x, y) = c^T x - y^T (A x - b) + δ_K(x)
KKT:  s := c - A^T y ∈ K* = K,  <s, x> = 0

PDHG (Chambolle-Pock) 迭代:
    x^{k+1} = Proj_K( x^k - η (c - A^T y^k) )           # 主步
    x_bar   = θ (x^{k+1} - x^k) + x^{k+1}                # 对偶外推
    y^{k+1} = y^k + τ (b - A x_bar)                      # 对偶步
收敛条件: η τ ||A||^2 ≤ 1。

PDQP 风格 (Halpern 动量 + 平均):
    x_md^k  = (1-β^k) x̄^k + β^k x^k
    x^{k+1} = Proj_K( x_md^k - η (c - A^T y^k) )
    x_bar   = θ (x^{k+1} - x^k) + x^{k+1}
    y^{k+1} = y^k + τ (b - A x_bar)
    x̄^{k+1} = (1-β^k) x̄^k + β^k x^{k+1}
β^k 默认采用 2/(k+2) (PDQP 常用 Halpern 调度)。
"""

import numpy as np
from typing import Optional, Tuple, List, Callable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.numerical import project_cone
from socp.problem import kkt_residuals


def estimate_op_norm(A: np.ndarray, n_iter: int = 50) -> float:
    """幂迭代估计 ||A||_2 (谱范数)。"""
    m, n = A.shape
    rng = np.random.RandomState(0)
    v = rng.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        u = A @ v
        un = np.linalg.norm(u)
        if un < 1e-12:
            break
        u /= un
        v = A.T @ u
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            break
        v /= vn
    return float(np.linalg.norm(A @ v))


def _record(history: dict, data: dict, x: np.ndarray, y: np.ndarray):
    res = kkt_residuals(data, x, y)
    for k, v in res.items():
        history.setdefault(k, []).append(v)


def pdhg_socp(
    data: dict,
    K_iters: int = 1000,
    eta: Optional[float] = None,
    tau: Optional[float] = None,
    theta: float = 1.0,
    record_every: int = 1,
    return_history: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """裸 PDHG (Chambolle-Pock) 求解 SOCP。

    Parameters
    ----------
    data : dict
        来自 generate_socp_data。
    K_iters : int
        迭代步数。
    eta, tau : float, optional
        主/对偶步长。默认 0.9 / ||A|| (使 η τ ||A||^2 ≈ 0.81)。
    theta : float
        对偶外推系数 (经典 θ = 1)。
    """
    A, b, c = data['A'], data['b'], data['c']
    block = data['block_sizes']
    N, m = data['N'], data['m']

    if eta is None or tau is None:
        L = estimate_op_norm(A)
        if eta is None:
            eta = 0.9 / max(L, 1e-8)
        if tau is None:
            tau = 0.9 / max(L, 1e-8)

    x = np.zeros(N)
    y = np.zeros(m)
    history = {'iter': []}

    for k in range(K_iters):
        x_new = project_cone(x - eta * (c - A.T @ y), block)
        x_bar = theta * (x_new - x) + x_new
        y = y + tau * (b - A @ x_bar)
        x = x_new
        if return_history and (k % record_every == 0 or k == K_iters - 1):
            history['iter'].append(k + 1)
            _record(history, data, x, y)

    return x, y, history


def pdqp_socp(
    data: dict,
    K_iters: int = 1000,
    eta: Optional[float] = None,
    tau: Optional[float] = None,
    theta: float = 1.0,
    beta_schedule: Optional[Callable[[int], float]] = None,
    record_every: int = 1,
    return_history: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """PDQP 风格 (Halpern 平均 + 动量) 求解 SOCP。"""
    A, b, c = data['A'], data['b'], data['c']
    block = data['block_sizes']
    N, m = data['N'], data['m']

    if eta is None or tau is None:
        L = estimate_op_norm(A)
        if eta is None:
            eta = 0.9 / max(L, 1e-8)
        if tau is None:
            tau = 0.9 / max(L, 1e-8)

    if beta_schedule is None:
        beta_schedule = lambda k: 2.0 / (k + 2.0)  # Halpern 经典

    x = np.zeros(N)
    y = np.zeros(m)
    x_avg = x.copy()
    history = {'iter': []}

    for k in range(K_iters):
        beta = float(beta_schedule(k))
        x_md = (1.0 - beta) * x_avg + beta * x
        x_new = project_cone(x_md - eta * (c - A.T @ y), block)
        x_bar = theta * (x_new - x) + x_new
        y = y + tau * (b - A @ x_bar)
        x_avg = (1.0 - beta) * x_avg + beta * x_new
        x = x_new
        if return_history and (k % record_every == 0 or k == K_iters - 1):
            history['iter'].append(k + 1)
            _record(history, data, x, y)

    return x, y, history


def summarize(history: dict, every: int = 100) -> str:
    """打印简要收敛信息。"""
    lines = []
    iters = history['iter']
    keys = [k for k in history.keys() if k != 'iter']
    header = "iter  " + "  ".join(f"{k:>10s}" for k in keys)
    lines.append(header)
    for i, it in enumerate(iters):
        if it % every == 0 or i == len(iters) - 1:
            row = f"{it:5d}  " + "  ".join(f"{history[k][i]:10.3e}" for k in keys)
            lines.append(row)
    return "\n".join(lines)
