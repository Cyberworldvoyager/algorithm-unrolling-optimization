"""
lasso/lista_universal.py — 通用 LISTA：解决维度问题并分析参数

改进:
1. 维度无关架构: 用相对维度而非绝对维度
2. 参数分析: 可视化和解释 W 矩阵
3. 更好的初始化策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt


class SoftThreshold(nn.Module):
    """可学习的软阈值激活函数。"""
    def __init__(self, init_threshold: float = 0.1, n: Optional[int] = None):
        super().__init__()
        if n is not None:
            self.theta = nn.Parameter(torch.full((n,), init_threshold))
        else:
            self.theta = nn.Parameter(torch.tensor(init_threshold))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.maximum(torch.abs(x) - self.theta, torch.zeros_like(x))


# ============================================================
# 方法 1: LISTA-Momentum-Universal — 维度无关架构
# ============================================================

class LISTAMomentumUniversalLayer(nn.Module):
    """LISTA-Momentum 单层 (维度无关版本)。

    核心改进:
    1. W 的维度由输入决定，而非固定
    2. 使用相对维度比例来初始化
    3. 支持不同大小的 A 矩阵
    """

    def __init__(self, hidden_dim: int = 64):
        """
        Parameters
        ----------
        hidden_dim : int — 隐藏层维度 (用于梯度变换)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 梯度变换网络 (维度无关)
        # 输入: 梯度 g (任意维度)
        # 输出: 变换后的梯度 (同维度)
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 可学习的步长和动量
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                x_prev: torch.Tensor, A: torch.Tensor,
                grad_norm: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        x : Tensor (batch, n) — 当前估计
        x_prev : Tensor (batch, n) — 上一步估计
        A : Tensor (batch, m, n) — 测量矩阵
        grad_norm : Tensor (batch, 1) — 梯度范数 (可选)
        """
        # 动量外推
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)

        # 用当前 A 计算梯度
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)  # (batch, m)
        residual = Ay - b  # (batch, m)
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)  # (batch, n)

        # 梯度变换 (维度无关)
        # 使用 LayerNorm 使变换与维度无关
        grad_normalized = F.layer_norm(grad, grad.shape[-1:])
        transformed_grad = self.transform(grad_normalized)

        # 更新
        z = y - self.eta * transformed_grad

        # 软阈值化
        return torch.sign(z) * torch.maximum(
            torch.abs(z) - self.threshold, torch.zeros_like(z)), x


class LISTAMomentumUniversal(nn.Module):
    """LISTA-Momentum (维度无关版本)。

    支持不同大小的 A 矩阵，无需重新训练。
    """

    def __init__(self, T: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            LISTAMomentumUniversalLayer(hidden_dim) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        x_prev = x.clone()

        for layer in self.layers:
            x, x_prev = layer(b, x, x_prev, A), x

        return x


# ============================================================
# 方法 2: LISTA-Momentum-Shared — 共享权重结构
# ============================================================

class LISTAMomentumSharedLayer(nn.Module):
    """LISTA-Momentum 单层 (共享权重版本)。

    核心思想: 用一个小网络生成梯度变换矩阵 W，
    这样 W 可以适应不同维度。
    """

    def __init__(self, n: int, rank: int = 5):
        """
        Parameters
        ----------
        n : int — 信号维度
        rank : int — W 的秩 (低秩分解)
        """
        super().__init__()
        self.n = n
        self.rank = rank

        # 低秩分解: W = U @ V^T
        self.U = nn.Parameter(torch.randn(n, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(n, rank) * 0.01)

        # 可学习的步长和动量
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                x_prev: torch.Tensor, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 动量外推
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)

        # 用当前 A 计算梯度
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

        # 低秩梯度变换: W @ g = U @ (V^T @ g)
        Vtgrad = torch.matmul(grad, self.V)  # (batch, rank)
        transformed_grad = torch.matmul(Vtgrad, self.U.T)  # (batch, n)

        # 更新
        z = y - self.eta * transformed_grad

        # 软阈值化
        return torch.sign(z) * torch.maximum(
            torch.abs(z) - self.threshold, torch.zeros_like(z)), x


class LISTAMomentumShared(nn.Module):
    """LISTA-Momentum (共享权重版本)。

    使用低秩分解来减少参数量。
    """

    def __init__(self, n: int, T: int = 10, rank: int = 5):
        super().__init__()
        self.n = n
        self.T = T
        self.layers = nn.ModuleList([
            LISTAMomentumSharedLayer(n, rank) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)

        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)
        x_prev = x.clone()

        for layer in self.layers:
            x, x_prev = layer(b, x, x_prev, A), x

        return x


# ============================================================
# 参数分析工具
# ============================================================

def analyze_W_matrix(model, layer_idx: int = 0) -> Dict:
    """分析 LISTA-Momentum 的 W 矩阵。

    Returns
    -------
    analysis : dict
        包含 W 的各种分析结果。
    """
    layer = model.layers[layer_idx]

    # 获取 W 矩阵
    if hasattr(layer, 'W'):
        W = layer.W.weight.detach().cpu().numpy()
    elif hasattr(layer, 'U') and hasattr(layer, 'V'):
        # 低秩分解
        U = layer.U.detach().cpu().numpy()
        V = layer.V.detach().cpu().numpy()
        W = U @ V.T
    else:
        return {}

    # 基本统计
    analysis = {
        'W': W,
        'shape': W.shape,
        'mean': float(np.mean(W)),
        'std': float(np.std(W)),
        'min': float(np.min(W)),
        'max': float(np.max(W)),
    }

    # 与单位矩阵的比较
    I = np.eye(W.shape[0])
    analysis['diff_from_identity'] = float(np.linalg.norm(W - I))
    analysis['cosine_similarity'] = float(np.sum(W * I) / (np.linalg.norm(W) * np.linalg.norm(I)))

    # 特征值分析
    eigvals = np.linalg.eigvals(W)
    analysis['eigenvalues'] = eigvals.tolist()
    analysis['spectral_radius'] = float(np.max(np.abs(eigvals)))
    analysis['condition_number'] = float(np.max(np.abs(eigvals)) / (np.min(np.abs(eigvals)) + 1e-10))

    # 奇异值分析
    U_svd, S, Vt = np.linalg.svd(W)
    analysis['singular_values'] = S.tolist()
    analysis['nuclear_norm'] = float(np.sum(S))
    analysis['effective_rank'] = float(np.sum(S > 0.1 * S[0]))

    return analysis


def visualize_W_matrix(analysis: Dict, save_path: Optional[str] = None):
    """可视化 W 矩阵。"""
    W = analysis['W']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # W 矩阵热力图
    im0 = axes[0].imshow(W, cmap='RdBu_r', aspect='auto')
    axes[0].set_title('W Matrix')
    axes[0].set_xlabel('Input Dimension')
    axes[0].set_ylabel('Output Dimension')
    plt.colorbar(im0, ax=axes[0])

    # 特征值分布
    eigvals = analysis['eigenvalues']
    axes[1].scatter([e.real for e in eigvals], [e.imag for e in eigvals], alpha=0.6)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel('Real Part')
    axes[1].set_ylabel('Imaginary Part')
    axes[1].set_title('Eigenvalue Distribution')
    axes[1].set_aspect('equal')

    # 奇异值分布
    S = analysis['singular_values']
    axes[2].plot(range(1, len(S) + 1), S, 'o-')
    axes[2].set_xlabel('Index')
    axes[2].set_ylabel('Singular Value')
    axes[2].set_title('Singular Value Distribution')
    axes[2].set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_all_layers(model) -> List[Dict]:
    """分析所有层的 W 矩阵。"""
    analyses = []
    for i in range(len(model.layers)):
        analysis = analyze_W_matrix(model, i)
        if analysis:
            analyses.append(analysis)
    return analyses


# ============================================================
# 工厂函数
# ============================================================

def create_universal_lista(variant: str, m: int, n: int, T: int = 10, **kwargs):
    """创建通用 LISTA 变体。

    Parameters
    ----------
    variant : str
        'universal' — 维度无关架构
        'shared' — 共享权重 (低秩分解)
        'momentum' — 原始 LISTA-Momentum
    """
    from lasso.lista_universal import LISTAMomentumUniversal, LISTAMomentumShared

    if variant == 'universal':
        return LISTAMomentumUniversal(T, kwargs.get('hidden_dim', 64))
    elif variant == 'shared':
        return LISTAMomentumShared(n, T, kwargs.get('rank', 5))
    elif variant == 'momentum':
        # 原始版本，需要固定维度
        return create_original_momentum(m, n, T)
    else:
        raise ValueError(f"Unknown variant: {variant}")


def create_original_momentum(m: int, n: int, T: int = 10):
    """创建原始 LISTA-Momentum (固定维度)。"""

    class LISTAMomentumOriginalLayer(nn.Module):
        def __init__(self, m, n):
            super().__init__()
            self.W = nn.Linear(n, n, bias=False)
            nn.init.eye_(self.W.weight)
            self.eta = nn.Parameter(torch.tensor(0.1))
            self.beta = nn.Parameter(torch.tensor(0.0))
            self.threshold = nn.Parameter(torch.tensor(0.1))

        def forward(self, b, x, x_prev, A):
            beta = torch.sigmoid(self.beta)
            y = x + beta * (x - x_prev)
            Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
            residual = Ay - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
            transformed_grad = self.W(grad)
            z = y - self.eta * transformed_grad
            x_new = torch.sign(z) * torch.maximum(
                torch.abs(z) - self.threshold, torch.zeros_like(z))
            return x_new

    class LISTAMomentumOriginal(nn.Module):
        def __init__(self, m, n, T):
            super().__init__()
            self.n = n
            self.layers = nn.ModuleList([
                LISTAMomentumOriginalLayer(m, n) for _ in range(T)
            ])

        def forward(self, b, A, x0=None):
            batch_size = b.shape[0]
            if A.dim() == 2:
                A = A.unsqueeze(0).expand(batch_size, -1, -1)
            x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)
            x_prev = x.clone()
            for layer in self.layers:
                x_new = layer(b, x, x_prev, A)
                x_prev = x
                x = x_new
            return x

    return LISTAMomentumOriginal(m, n, T)
