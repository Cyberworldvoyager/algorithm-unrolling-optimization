"""
low_rank/problem.py — 低秩矩阵恢复问题定义与数据生成

矩阵补全问题: 从部分观测 P_Ω(M) 恢复低秩矩阵 M
优化形式: min_{X,Z} ||X||_*  s.t.  P_Ω(X) = P_Ω(M), Z = X
"""

import numpy as np
from typing import Tuple, Optional, Dict


def generate_low_rank_matrix(
    m: int = 50,
    n: int = 50,
    rank: int = 5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成低秩矩阵 M = U @ V^T。

    Parameters
    ----------
    m : int
        矩阵行数。
    n : int
        矩阵列数。
    rank : int
        矩阵秩。
    seed : int, optional
        随机种子。

    Returns
    -------
    M : np.ndarray
        低秩矩阵 (m, n)。
    U : np.ndarray
        左因子 (m, rank)。
    V : np.ndarray
        右因子 (n, rank)。
    """
    rng = np.random.RandomState(seed)
    U = rng.randn(m, rank)
    V = rng.randn(n, rank)
    M = U @ V.T
    return M, U, V


def generate_observation_mask(
    m: int,
    n: int,
    ratio: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """生成观测掩码 Ω。

    Parameters
    ----------
    m : int
        矩阵行数。
    n : int
        矩阵列数。
    ratio : float
        观测比例 (0, 1]。
    seed : int, optional
        随机种子。

    Returns
    -------
    mask : np.ndarray
        布尔掩码 (m, n)，True 表示已观测。
    """
    rng = np.random.RandomState(seed)
    total = m * n
    num_obs = int(total * ratio)
    indices = rng.choice(total, num_obs, replace=False)
    mask = np.zeros(total, dtype=bool)
    mask[indices] = True
    return mask.reshape(m, n)


def apply_mask(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """应用观测掩码: P_Ω(X)。

    Parameters
    ----------
    X : np.ndarray
        输入矩阵。
    mask : np.ndarray
        布尔掩码。

    Returns
    -------
    X_masked : np.ndarray
        仅保留观测位置的值，其余为 0。
    """
    return X * mask


def generate_matrix_completion_data(
    m: int = 50,
    n: int = 50,
    rank: int = 5,
    ratio: float = 0.5,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """生成矩阵补全问题数据。

    Returns
    -------
    data : dict
        包含 M, mask, M_observed, rank 等。
    """
    M, U, V = generate_low_rank_matrix(m, n, rank, seed)
    mask = generate_observation_mask(m, n, ratio, seed)

    # 添加噪声
    if noise_std > 0:
        rng = np.random.RandomState(seed)
        M = M + noise_std * rng.randn(m, n)

    M_observed = apply_mask(M, mask)

    return {
        'M': M,
        'M_observed': M_observed,
        'mask': mask,
        'U': U,
        'V': V,
        'rank': rank,
        'ratio': ratio,
    }


def nuclear_norm_objective(X: np.ndarray, mask: np.ndarray, M_observed: np.ndarray) -> float:
    """计算矩阵补全目标函数值: ||X||_* + (ρ/2) ||P_Ω(X) - P_Ω(M)||²。

    这里简化为核范数 + 数据拟合项。
    """
    nuclear = np.sum(np.linalg.svd(X, compute_uv=False))
    data_fit = 0.5 * np.sum((apply_mask(X, mask) - M_observed) ** 2)
    return float(nuclear + data_fit)


def generate_batch_data(
    batch_size: int,
    m: int = 50,
    n: int = 50,
    rank: int = 5,
    ratio: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """生成一批矩阵补全数据。

    Returns
    -------
    M_batch : np.ndarray
        真实矩阵 (batch_size, m, n)。
    M_obs_batch : np.ndarray
        观测矩阵 (batch_size, m, n)。
    mask_batch : np.ndarray
        观测掩码 (batch_size, m, n)。
    rank_batch : np.ndarray
        秩 (batch_size,)。
    """
    rng = np.random.RandomState(seed)
    M_list, M_obs_list, mask_list, rank_list = [], [], [], []

    for _ in range(batch_size):
        data = generate_matrix_completion_data(m, n, rank, ratio, seed=rng.randint(1e6))
        M_list.append(data['M'])
        M_obs_list.append(data['M_observed'])
        mask_list.append(data['mask'])
        rank_list.append(data['rank'])

    return np.array(M_list), np.array(M_obs_list), np.array(mask_list), np.array(rank_list)
