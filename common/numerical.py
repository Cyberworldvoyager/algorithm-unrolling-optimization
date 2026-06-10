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


def project_soc(z: np.ndarray) -> np.ndarray:
    """单个二阶锥的闭式投影。

    二阶锥 K = {(t, u) ∈ R × R^{k-1} : ||u||_2 <= t}。
    设 z = (t, u)，则
        - 若 ||u|| <= t       : Proj(z) = z
        - 若 ||u|| <= -t      : Proj(z) = 0
        - 否则                : Proj(z) = ((t + ||u||) / 2) * (1, u / ||u||)
    """
    z = np.asarray(z, dtype=np.float64)
    t = z[0]
    u = z[1:]
    nu = float(np.linalg.norm(u))
    if nu <= t:
        return z.copy()
    if nu <= -t:
        return np.zeros_like(z)
    s = 0.5 * (t + nu)
    out = np.empty_like(z)
    out[0] = s
    out[1:] = s * (u / nu)
    return out


def project_cone(x: np.ndarray, block_sizes) -> np.ndarray:
    """到锥积 K = K_1 × K_2 × ... × K_p 的投影。

    Parameters
    ----------
    x : np.ndarray
        待投影向量，长度等于 sum(block_sizes)。
    block_sizes : Iterable[int]
        每个二阶锥块的维度。维度为 1 时退化为 R_+ 上的投影 (max(0, x))。
    """
    out = np.empty_like(x, dtype=np.float64)
    offset = 0
    for k in block_sizes:
        end = offset + k
        if k == 1:
            out[offset:end] = np.maximum(x[offset:end], 0.0)
        else:
            out[offset:end] = project_soc(x[offset:end])
        offset = end
    return out


def cone_dist(x: np.ndarray, block_sizes) -> float:
    """计算 x 到锥积 K 的距离 ||x - Proj_K(x)||_2 (锥可行性残差)。"""
    return float(np.linalg.norm(x - project_cone(x, block_sizes)))


def project_soc_torch(z):
    """torch 版 SOC 投影 (支持 batch，沿最后一维)。

    z shape: (..., k)
    """
    import torch
    t = z[..., :1]
    u = z[..., 1:]
    nu = torch.linalg.norm(u, dim=-1, keepdim=True)
    inside = (nu <= t).float()
    outside_neg = ((nu <= -t) & (nu > t)).float()  # nu <= -t 且 nu > t
    middle = 1.0 - inside - outside_neg
    s = 0.5 * (t + nu)
    eps = 1e-12
    u_proj = s * (u / (nu + eps))
    z_proj_middle = torch.cat([s, u_proj], dim=-1)
    z_zero = torch.zeros_like(z)
    return inside * z + middle * z_proj_middle + outside_neg * z_zero


def project_cone_torch(x, block_sizes):
    """torch 版锥积投影。x shape: (..., N)，N = sum(block_sizes)。"""
    import torch
    pieces = []
    offset = 0
    for k in block_sizes:
        end = offset + k
        seg = x[..., offset:end]
        if k == 1:
            pieces.append(torch.clamp(seg, min=0.0))
        else:
            pieces.append(project_soc_torch(seg))
        offset = end
    return torch.cat(pieces, dim=-1)


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
