"""
lasso/problem.py — LASSO 稀疏编码问题定义与数据生成

LASSO 问题: min_x  1/2 ||Ax - b||² + λ||x||₁
"""

import numpy as np
from typing import Tuple, Optional


def generate_lasso_data(
    m: int = 50,
    n: int = 200,
    sparsity: int = 10,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成 LASSO 问题的合成数据。

    Parameters
    ----------
    m : int
        观测维度 (行数)。
    n : int
        信号维度 (列数)，通常 m << n。
    sparsity : int
        稀疏度 (非零元素个数)。
    noise_std : float
        噪声标准差。
    seed : int, optional
        随机种子。

    Returns
    -------
    A : np.ndarray
        测量矩阵 (m, n)。
    b : np.ndarray
        观测向量 (m,)。
    x_true : np.ndarray
        真实稀疏信号 (n,)。
    """
    rng = np.random.RandomState(seed)

    # 生成测量矩阵 A，列归一化
    A = rng.randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)

    # 生成稀疏信号 x_true
    x_true = np.zeros(n)
    support = rng.choice(n, sparsity, replace=False)
    x_true[support] = rng.randn(sparsity)

    # 生成含噪观测 b = A x + noise
    b = A @ x_true + noise_std * rng.randn(m)

    return A, b, x_true


def lasso_objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:
    """计算 LASSO 目标函数值: 1/2 ||Ax - b||² + λ||x||₁。"""
    residual = A @ x - b
    return 0.5 * np.sum(residual ** 2) + lam * np.sum(np.abs(x))


def compute_optimal_lambda(A: np.ndarray, b: np.ndarray) -> float:
    """计算使 LASSO 解为零向量的最小 λ (λ_max)。

    λ_max = ||A^T b||_∞
    """
    return float(np.max(np.abs(A.T @ b)))


def generate_batch_data(
    batch_size: int,
    m: int = 50,
    n: int = 200,
    sparsity: int = 10,
    noise_std: float = 0.01,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成一批 LASSO 数据用于训练。

    Returns
    -------
    A_batch : np.ndarray
        测量矩阵 (batch_size, m, n)。
    b_batch : np.ndarray
        观测向量 (batch_size, m)。
    x_batch : np.ndarray
        真实稀疏信号 (batch_size, n)。
    """
    rng = np.random.RandomState(seed)
    A_batch = []
    b_batch = []
    x_batch = []

    for _ in range(batch_size):
        A, b, x = generate_lasso_data(m, n, sparsity, noise_std, seed=rng.randint(1e6))
        A_batch.append(A)
        b_batch.append(b)
        x_batch.append(x)

    return np.array(A_batch), np.array(b_batch), np.array(x_batch)
