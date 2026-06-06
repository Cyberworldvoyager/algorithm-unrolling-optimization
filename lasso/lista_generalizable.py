"""
lasso/lista_generalizable.py — 提升泛化性的 LISTA 变体

解决核心问题：如何让展开网络泛化到不同的 A 矩阵？

方法：
1. ConditionalLISTA — A 作为输入，动态计算 W1, W2
2. HyperLISTA — 用小网络从 A 生成权重
3. AdaLISTA — 元学习风格，快速适应新 A
4. SharedLISTA — 共享权重，不依赖 A
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


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
# 方法 1: Conditional LISTA — A 作为额外输入
# ============================================================

class ConditionalLISTALayer(nn.Module):
    """条件 LISTA 单层。

    核心思想：不将 A 固化在权重中，而是将 A 作为额外输入。
    这样网络可以处理不同的 A 矩阵。

    更新公式：
        W1 = f_theta(A)  # 从 A 动态生成 W1
        W2 = g_theta(A)  # 从 A 动态生成 W2
        x_{t+1} = σ(W1 @ b + W2 @ x_t; θ_t)
    """

    def __init__(self, m: int, n: int, hidden_dim: int = 64):
        super().__init__()
        self.m, self.n = m, n

        # 从 A 生成 W1 的网络
        self.W1_generator = nn.Sequential(
            nn.Linear(m * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n * m),
        )

        # 从 A 生成 W2 的网络
        self.W2_generator = nn.Sequential(
            nn.Linear(m * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n * n),
        )

        self.threshold = SoftThreshold(init_threshold=0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                A_flat: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        x : Tensor (batch, n) — 当前估计
        A_flat : Tensor (batch, m*n) — 展平的 A 矩阵
        """
        batch_size = b.shape[0]

        # 动态生成权重
        W1 = self.W1_generator(A_flat).view(batch_size, self.n, self.m)  # (batch, n, m)
        W2 = self.W2_generator(A_flat).view(batch_size, self.n, self.n)  # (batch, n, n)

        # 计算更新
        # W1 @ b: (batch, n, m) @ (batch, m, 1) -> (batch, n)
        W1b = torch.bmm(W1, b.unsqueeze(-1)).squeeze(-1)
        # W2 @ x: (batch, n, n) @ (batch, n, 1) -> (batch, n)
        W2x = torch.bmm(W2, x.unsqueeze(-1)).squeeze(-1)

        return self.threshold(W1b + W2x)


class ConditionalLISTA(nn.Module):
    """条件 LISTA：将 A 作为输入，动态生成权重。

    优势：
    - 理论上可以泛化到任何 A
    - 不需要为每个 A 重新训练

    劣势：
    - 参数量更大
    - 需要学习 A 到权重的映射
    """

    def __init__(self, m: int, n: int, T: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            ConditionalLISTALayer(m, n, hidden_dim) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Parameters
        ----------
        b : Tensor (batch, m) — 观测
        A : Tensor (batch, m, n) 或 (m, n) — 测量矩阵
        """
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        A_flat = A.view(batch_size, -1)  # (batch, m*n)

        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)

        for layer in self.layers:
            x = layer(b, x, A_flat)

        return x


# ============================================================
# 方法 2: Hypernetwork LISTA — 用小网络生成权重
# ============================================================

class HyperLISTALayer(nn.Module):
    """Hypernetwork LISTA 单层。

    核心思想：用一个小型超网络 (hypernetwork) 从 A 生成主网络的权重。
    超网络学习 A -> 权重 的映射，主网络执行 LISTA 更新。

    与 Conditional LISTA 的区别：
    - Conditional: 直接从 A 生成 W1, W2
    - Hyper: 先编码 A，再从编码生成权重（更紧凑）
    """

    def __init__(self, m: int, n: int, latent_dim: int = 32):
        super().__init__()
        self.m, self.n = m, n

        # A 编码器：将 A 压缩为低维表示
        self.A_encoder = nn.Sequential(
            nn.Linear(m * n, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # 从 latent 生成 W1
        self.W1_gen = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, n * m),
        )

        # 从 latent 生成 W2
        self.W2_gen = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, n * n),
        )

        self.threshold = SoftThreshold(init_threshold=0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                A_flat: torch.Tensor) -> torch.Tensor:
        batch_size = b.shape[0]

        # 编码 A
        z = self.A_encoder(A_flat)  # (batch, latent_dim)

        # 生成权重
        W1 = self.W1_gen(z).view(batch_size, self.n, self.m)
        W2 = self.W2_gen(z).view(batch_size, self.n, self.n)

        # LISTA 更新
        W1b = torch.bmm(W1, b.unsqueeze(-1)).squeeze(-1)
        W2x = torch.bmm(W2, x.unsqueeze(-1)).squeeze(-1)

        return self.threshold(W1b + W2x)


class HyperLISTA(nn.Module):
    """Hypernetwork LISTA。

    用超网络从 A 的紧凑表示生成 LISTA 权重。
    """

    def __init__(self, m: int, n: int, T: int = 10, latent_dim: int = 32):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            HyperLISTALayer(m, n, latent_dim) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        A_flat = A.view(batch_size, -1)

        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)

        for layer in self.layers:
            x = layer(b, x, A_flat)

        return x


# ============================================================
# 方法 3: Shared LISTA — 共享权重，不依赖 A
# ============================================================

class SharedLISTALayer(nn.Module):
    """共享权重 LISTA 单层。

    核心思想：完全移除对 A 的依赖，只学习通用的稀疏恢复规则。

    更新公式：
        x_{t+1} = σ(W1 @ b + W2 @ x_t; θ_t)

    其中 W1, W2 是与 A 无关的共享权重。
    这是最极端的泛化尝试——如果有效，说明网络学到了通用的稀疏恢复能力。
    """

    def __init__(self, m: int, n: int):
        super().__init__()
        self.W1 = nn.Linear(m, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)

        # 随机初始化（不使用 A）
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.eye_(self.W2.weight)
        self.W2.weight.data *= 0.5

        self.threshold = SoftThreshold(init_threshold=0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.threshold(self.W1(b) + self.W2(x))


class SharedLISTA(nn.Module):
    """共享权重 LISTA。

    完全不依赖 A，学习通用的稀疏恢复规则。
    forward 接受 A 参数以保持接口一致，但不使用它。
    """

    def __init__(self, m: int, n: int, T: int = 10):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            SharedLISTALayer(m, n) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: Optional[torch.Tensor] = None,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)

        for layer in self.layers:
            x = layer(b, x)

        return x


# ============================================================
# 方法 4: AdaLISTA — 元学习风格，快速适应
# ============================================================

class AdaLISTALayer(nn.Module):
    """AdaLISTA 单层。

    核心思想：用元学习 (MAML 风格) 训练，使得网络可以用少量梯度步适应新 A。

    结构：基本 LISTA，但训练时在多个 A 上做内循环更新。
    """

    def __init__(self, m: int, n: int, init_eta: float = 0.1):
        super().__init__()
        self.register_buffer('A', torch.zeros(m, n))
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.B = nn.Parameter(torch.randn(n, m) * 0.01)
        self.threshold = SoftThreshold(init_threshold=0.1, n=n)

    def set_A(self, A: torch.Tensor):
        """设置当前 A 矩阵。"""
        self.A = A

    def forward(self, b: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        Ax = F.linear(x, self.A)
        BAx = F.linear(Ax, self.B)
        Bb = F.linear(b, self.B)
        z = self.eta * Bb + x - self.eta * BAx
        return self.threshold(z)


class AdaLISTA(nn.Module):
    """AdaLISTA：元学习风格的 LISTA。

    训练时：在多个 A 上做内循环适应
    测试时：用少量梯度步适应新 A
    """

    def __init__(self, m: int, n: int, T: int = 10, init_eta: float = 0.1):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            AdaLISTALayer(m, n, init_eta) for _ in range(T)
        ])

    def set_A(self, A: torch.Tensor):
        """设置当前 A 矩阵。"""
        for layer in self.layers:
            layer.set_A(A)

    def forward(self, b: torch.Tensor, x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)

        for layer in self.layers:
            x = layer(b, x)

        return x

    def adapt_to_A(self, A: torch.Tensor, b_support: torch.Tensor,
                   x_support: torch.Tensor, num_steps: int = 5, lr: float = 0.01):
        """用少量支持样本适应新 A。

        Parameters
        ----------
        A : Tensor (m, n) — 新的测量矩阵
        b_support : Tensor (K, m) — 支持样本的观测
        x_support : Tensor (K, n) — 支持样本的真实值
        num_steps : int — 适应步数
        lr : float — 适应学习率
        """
        self.set_A(A)

        # 保存原始权重
        original_params = {name: param.clone() for name, param in self.named_parameters()
                          if 'B' in name or 'eta' in name}

        # 内循环适应
        adapt_params = {name: param.clone().requires_grad_(True)
                       for name, param in original_params.items()}

        for _ in range(num_steps):
            # 前向传播
            x_pred = self.forward(b_support)
            loss = F.mse_loss(x_pred, x_support)

            # 计算梯度
            grads = torch.autograd.grad(loss, self.parameters(),
                                        create_graph=False, allow_unused=True)

            # 更新可适应参数
            for (name, param), grad in zip(self.named_parameters(), grads):
                if grad is not None and name in adapt_params:
                    adapt_params[name] = adapt_params[name] - lr * grad

        # 应用适应后的权重
        for name, param in self.named_parameters():
            if name in adapt_params:
                param.data = adapt_params[name].data


# ============================================================
# 方法 5: LISTA with A-input — A 作为额外特征输入
# ============================================================

class LISTAWithAInputLayer(nn.Module):
    """LISTA with A-input 单层。

    核心思想：将 A 的特征作为额外输入拼接到每层。
    这样网络可以在推理时感知到当前的 A。

    更新公式：
        z = [b; flatten(A)]  # 拼接观测和 A 的特征
        x_{t+1} = σ(W1 @ z + W2 @ x_t; θ_t)
    """

    def __init__(self, m: int, n: int, A_feature_dim: int = 32):
        super().__init__()
        self.m, self.n = m, n

        # A 特征提取器
        self.A_encoder = nn.Sequential(
            nn.Linear(m * n, A_feature_dim),
            nn.ReLU(),
        )

        # 输入维度：m (b) + A_feature_dim
        input_dim = m + A_feature_dim

        self.W1 = nn.Linear(input_dim, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)

        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.eye_(self.W2.weight)
        self.W2.weight.data *= 0.5

        self.threshold = SoftThreshold(init_threshold=0.1, n=n)

    def forward(self, b: torch.Tensor, x: torch.Tensor,
                A_flat: torch.Tensor) -> torch.Tensor:
        # 提取 A 特征
        A_feat = self.A_encoder(A_flat)  # (batch, A_feature_dim)

        # 拼接 b 和 A 特征
        z = torch.cat([b, A_feat], dim=-1)  # (batch, m + A_feature_dim)

        return self.threshold(self.W1(z) + self.W2(x))


class LISTAWithAInput(nn.Module):
    """LISTA with A-input：将 A 特征作为额外输入。

    每层都能感知到当前的 A 矩阵。
    """

    def __init__(self, m: int, n: int, T: int = 10, A_feature_dim: int = 32):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([
            LISTAWithAInputLayer(m, n, A_feature_dim) for _ in range(T)
        ])

    def forward(self, b: torch.Tensor, A: torch.Tensor,
                x0: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        A_flat = A.view(batch_size, -1)

        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)

        for layer in self.layers:
            x = layer(b, x, A_flat)

        return x


# ============================================================
# 工厂函数
# ============================================================

def create_generalizable_lista(variant: str, m: int, n: int, T: int = 10, **kwargs):
    """创建可泛化的 LISTA 变体。

    Parameters
    ----------
    variant : str
        'conditional' — 条件 LISTA，A 作为输入
        'hyper' — 超网络 LISTA
        'shared' — 共享权重，不依赖 A
        'adainput' — A 特征作为额外输入
        'ada' — 元学习风格
    """
    variants = {
        'conditional': lambda: ConditionalLISTA(m, n, T, kwargs.get('hidden_dim', 64)),
        'hyper': lambda: HyperLISTA(m, n, T, kwargs.get('latent_dim', 32)),
        'shared': lambda: SharedLISTA(m, n, T),
        'adainput': lambda: LISTAWithAInput(m, n, T, kwargs.get('A_feature_dim', 32)),
        'ada': lambda: AdaLISTA(m, n, T, kwargs.get('init_eta', 0.1)),
    }

    if variant not in variants:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(variants.keys())}")

    return variants[variant]()
