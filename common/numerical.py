"""
common/numerical.py — 数值计算工具
"""

import numpy as np
from typing import Optional, Tuple


def condition_number(A: np.ndarray) -> float:
    """计算矩阵条件数 κ(A) = σ_max / σ_min。"""
    sv = np.linalg.svd(A, compute_uv=False)
    return float(sv[0] / sv[-1])


def svd_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    """奇异值软阈值化: D_τ(X) = U diag(max(σ_i - τ, 0)) V^T。"""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s_thresh = np.maximum(s - tau, 0.0)
    return U @ np.diag(s_thresh) @ Vt


def nuclear_norm(X: np.ndarray) -> float:
    """计算核范数 ||X||_* = Σ σ_i。"""
    return float(np.sum(np.linalg.svd(X, compute_uv=False)))


def proximal_l1(v: np.ndarray, lam: float) -> np.ndarray:
    """L1 近端算子 (软阈值)。"""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def proximal_l2(v: np.ndarray, lam: float) -> np.ndarray:
    """L2 近端算子 (缩放)。"""
    return v / (1.0 + lam)


def proximal_nuclear(X: np.ndarray, tau: float) -> np.ndarray:
    """核范数近端算子 (奇异值软阈值化)。"""
    return svd_threshold(X, tau)


def project_box(x: np.ndarray, lb: float = 0.0, ub: float = 1.0) -> np.ndarray:
    """投影到 box 约束 [lb, ub]^n。"""
    return np.clip(x, lb, ub)


def project_simplex(v: np.ndarray, s: float = 1.0) -> np.ndarray:
    """投影到单纯形 {x : x >= 0, sum(x) = s}。

    Duchi et al. (2008) O(n log n) 算法。
    """
    n = v.shape[0]
    if n == 0:
        return v.copy()

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def project_affine(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """投影到仿射集 {x : Ax = b}。

    x_proj = x - A^T (AA^T)^{-1} (Ax - b)
    """
    r = A @ x - b
    return x - A.T @ np.linalg.solve(A @ A.T, r)


def power_iteration(A: np.ndarray, n_iter: int = 100, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
    """幂迭代法求最大特征值和对应特征向量。"""
    n = A.shape[0]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)

    for _ in range(n_iter):
        Av = A @ v
        lam = np.linalg.norm(Av)
        v_new = Av / lam
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new

    return lam, v


def generate_spd_matrix(n: int, kappa: Optional[float] = None, seed: Optional[int] = None) -> np.ndarray:
    """生成随机对称正定矩阵。

    Parameters
    ----------
    n : int
        矩阵维度。
    kappa : float, optional
        期望条件数。若为 None 则随机生成。
    seed : int, optional
        随机种子。
    """
    rng = np.random.RandomState(seed)
    Q, _ = np.linalg.qr(rng.randn(n, n))

    if kappa is not None:
        d = np.linspace(1.0, 1.0 / kappa, n)
    else:
        d = rng.uniform(0.1, 10.0, n)

    return Q @ np.diag(d) @ Q.T


def gaussian_random_matrix(m: int, n: int, normalize: bool = True, seed: Optional[int] = None) -> np.ndarray:
    """生成高斯随机矩阵 A ∈ R^{m×n}。"""
    rng = np.random.RandomState(seed)
    A = rng.randn(m, n)
    if normalize:
        A /= np.sqrt(m)
    return A
