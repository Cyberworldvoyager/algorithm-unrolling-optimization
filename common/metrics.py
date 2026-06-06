"""
common/metrics.py — 评估指标
"""

import numpy as np
from typing import Dict, List, Optional


def mse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """均方误差 (MSE)。"""
    return float(np.mean((x_true - x_pred) ** 2))


def rmse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """均方根误差 (RMSE)。"""
    return float(np.sqrt(mse(x_true, x_pred)))


def relative_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """相对误差 ||x_true - x_pred||_2 / ||x_true||_2。"""
    return float(np.linalg.norm(x_true - x_pred) / (np.linalg.norm(x_true) + 1e-10))


def psnr(x_true: np.ndarray, x_pred: np.ndarray, max_val: float = 1.0) -> float:
    """峰值信噪比 (PSNR)。"""
    mse_val = mse(x_true, x_pred)
    if mse_val < 1e-10:
        return float('inf')
    return float(20 * np.log10(max_val) - 10 * np.log10(mse_val))


def support_recovery(x_true: np.ndarray, x_pred: np.ndarray, tol: float = 1e-3) -> float:
    """支撑集恢复率 (用于稀疏信号)。"""
    support_true = set(np.where(np.abs(x_true) > tol)[0])
    support_pred = set(np.where(np.abs(x_pred) > tol)[0])
    if len(support_true) == 0:
        return 1.0 if len(support_pred) == 0 else 0.0
    return len(support_true & support_pred) / len(support_true)


def rank_recovery(X_true: np.ndarray, X_pred: np.ndarray, tol: float = 1e-3) -> bool:
    """秩恢复判断 (用于低秩矩阵)。"""
    rank_true = np.linalg.matrix_rank(X_true, tol=tol)
    rank_pred = np.linalg.matrix_rank(X_pred, tol=tol)
    return rank_pred <= rank_true


def constraint_violation(x: np.ndarray, lb: float = 0.0, ub: float = 1.0) -> float:
    """约束违反程度 (box 约束)。"""
    violation = np.maximum(x - ub, 0) + np.maximum(lb - x, 0)
    return float(np.max(violation))


def optimality_gap(obj_val: float, obj_optimal: float) -> float:
    """最优性差距 (f(x) - f*) / |f*|。"""
    if abs(obj_optimal) < 1e-10:
        return abs(obj_val)
    return abs(obj_val - obj_optimal) / abs(obj_optimal)


class MetricsTracker:
    """训练过程指标跟踪器。"""

    def __init__(self):
        self.history: Dict[str, List[float]] = {}

    def update(self, metrics: Dict[str, float]):
        """更新指标。"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get(self, key: str) -> List[float]:
        """获取指定指标的历史记录。"""
        return self.history.get(key, [])

    def get_latest(self, key: str) -> Optional[float]:
        """获取指定指标的最新值。"""
        history = self.get(key)
        return history[-1] if history else None

    def summary(self) -> Dict[str, float]:
        """返回所有指标的最新值。"""
        return {key: values[-1] for key, values in self.history.items() if values}
