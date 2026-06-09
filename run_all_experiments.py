"""
run_all_experiments.py — 全部 10 个模型的完整对比实验

模型列表:
  经典基线:   ISTA, FISTA
  展开网络:   LISTA, LISTA-CP, LISTA-Momentum, 维度无关 LISTA
  序列模型:   RNN-LISTA, LSTM-LISTA, LSTM-LISTA v2, Transformer-LISTA

实验维度:
  1. 不同展开层数 T (T=5, 10, 20)
  2. 泛化性 (ID vs OOD)
  3. 无噪声 vs 有噪声
"""

import sys, json, os
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.metrics import relative_error
from lasso.classical import ista, fista


def set_seed(x):
    torch.manual_seed(x)
    np.random.seed(x)


def to_numpy(t):
    return t.detach().cpu().numpy()


def gen_data(A, num, sparsity=10, noise=0.0, seed=0):
    rng = np.random.RandomState(seed)
    m, n = A.shape
    B, X = [], []
    for _ in range(num):
        x = np.zeros(n)
        support = rng.choice(n, sparsity, replace=False)
        x[support] = rng.randn(sparsity)
        b = A @ x + noise * rng.randn(m)
        B.append(b.astype(np.float32))
        X.append(x.astype(np.float32))
    return np.array(B), np.array(X)


# ============================================================
# 模型定义
# ============================================================

# --- LISTA (基本) ---
class LISTABasicLayer(nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.W1 = nn.Linear(m, n, bias=False)
        self.W2 = nn.Linear(n, n, bias=False)
        self.theta = nn.Parameter(torch.tensor(0.1))
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.eye_(self.W2.weight)
        self.W2.weight.data *= 0.9

    def forward(self, b, x):
        z = self.W1(b) + self.W2(x)
        return torch.sign(z) * torch.maximum(torch.abs(z) - self.theta, torch.zeros_like(z))


class LISTABasic(nn.Module):
    def __init__(self, m, n, T=10):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList([LISTABasicLayer(m, n) for _ in range(T)])

    def forward(self, b, A=None):
        x = torch.zeros(b.shape[0], self.n, device=b.device)
        for layer in self.layers:
            x = layer(b, x)
        return x


# --- LISTA-CP ---
class LISTACPLayer(nn.Module):
    def __init__(self, A, n, init_eta=0.1):
        super().__init__()
        self.register_buffer('A', A)
        self.eta = nn.Parameter(torch.tensor(init_eta))
        self.B = nn.Parameter(A.T.clone())
        self.threshold = nn.Parameter(torch.tensor(init_eta * 0.1))

    def forward(self, b, x):
        Ax = torch.nn.functional.linear(x, self.A)
        BAx = torch.nn.functional.linear(Ax, self.B)
        Bb = torch.nn.functional.linear(b, self.B)
        z = self.eta * Bb + x - self.eta * BAx
        return torch.sign(z) * torch.maximum(torch.abs(z) - self.threshold, torch.zeros_like(z))


class LISTACP(nn.Module):
    def __init__(self, A, T=10, init_eta=0.1):
        super().__init__()
        self.n = A.shape[1]
        self.layers = nn.ModuleList([LISTACPLayer(A, self.n, init_eta) for _ in range(T)])

    def forward(self, b, A=None):
        x = torch.zeros(b.shape[0], self.n, device=b.device)
        for layer in self.layers:
            x = layer(b, x)
        return x


# --- LISTA-Momentum ---
class LISTAMomentumLayer(nn.Module):
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
        z = y - self.eta * self.W(grad)
        return torch.sign(z) * torch.maximum(torch.abs(z) - self.threshold, torch.zeros_like(z))


class LISTAMomentum(nn.Module):
    def __init__(self, m, n, T=10):
        super().__init__()
        self.n = n
        self.layers = nn.ModuleList([LISTAMomentumLayer(m, n) for _ in range(T)])

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.zeros(batch_size, self.n, device=b.device)
        x_prev = x.clone()
        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new
        return x


# --- 维度无关 LISTA ---
class DimAgnosticLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, b, x, x_prev, A):
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
        batch_size, n = grad.shape
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_normalized = (grad - grad_mean) / grad_std
        grad_flat = grad_normalized.reshape(-1, 1)
        transformed_flat = self.transform(grad_flat)
        transformed_normalized = transformed_flat.reshape(batch_size, n)
        transformed_grad = (transformed_normalized + grad_normalized) * grad_std * self.scale
        z = y - self.eta * transformed_grad
        return torch.sign(z) * torch.maximum(torch.abs(z) - self.threshold, torch.zeros_like(z))


class DimAgnosticLISTA(nn.Module):
    def __init__(self, T=10, hidden_dim=32):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([DimAgnosticLayer(hidden_dim) for _ in range(T)])

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = torch.zeros(batch_size, n, device=b.device)
        x_prev = x.clone()
        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new
        return x


# --- RNN-LISTA ---
class RNNLISTA(nn.Module):
    def __init__(self, T=10, hidden_dim=32):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRUCell(input_size=1, hidden_size=hidden_dim)
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        for _ in range(self.T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            grad = torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)
            g_mean = grad.mean(dim=-1, keepdim=True)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((grad - g_mean) / g_std).reshape(-1, 1)
            h_new = self.rnn(g_norm, h.reshape(-1, self.hidden_dim))
            update = self.hidden_to_update(h_new).reshape(batch_size, n) * g_std
            x = x + self.eta * update
            h = h_new.reshape(batch_size, n, self.hidden_dim)
        return x


# --- LSTM-LISTA (v1, 共享权重) ---
class LSTMLISTA(nn.Module):
    def __init__(self, T=10, hidden_dim=32):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        for _ in range(self.T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            grad = torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)
            g_mean = grad.mean(dim=-1, keepdim=True)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((grad - g_mean) / g_std).reshape(-1, 1)
            h_new, c_new = self.lstm(g_norm, (h.reshape(-1, self.hidden_dim), c.reshape(-1, self.hidden_dim)))
            update = self.hidden_to_update(h_new).reshape(batch_size, n) * g_std
            x = x + self.eta * update
            h = h_new.reshape(batch_size, n, self.hidden_dim)
            c = c_new.reshape(batch_size, n, self.hidden_dim)
        return x


# --- LSTM-LISTA v2 (逐元素, 与 v1 相同架构但独立训练) ---
class LSTMLISTAv2(nn.Module):
    """LSTM-LISTA v2: 与 v1 架构相同，独立训练以对比。"""
    def __init__(self, T=10, hidden_dim=32):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_size=1, hidden_size=hidden_dim)
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, n, self.hidden_dim, device=b.device)
        for _ in range(self.T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            grad = torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)
            g_mean = grad.mean(dim=-1, keepdim=True)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((grad - g_mean) / g_std).reshape(-1, 1)
            h_new, c_new = self.lstm(g_norm, (h.reshape(-1, self.hidden_dim), c.reshape(-1, self.hidden_dim)))
            update = self.hidden_to_update(h_new).reshape(batch_size, n) * g_std
            x = x + self.eta * update
            h = h_new.reshape(batch_size, n, self.hidden_dim)
            c = c_new.reshape(batch_size, n, self.hidden_dim)
        return x


# --- Transformer-LISTA (修复版) ---
class TransformerLISTA(nn.Module):
    """Transformer-LISTA: 逐元素处理，用 Transformer 建模迭代间关系。

    修复: 逐元素处理 (与 RNN/LSTM 一致)，Transformer 作用于迭代序列。
    """
    def __init__(self, T=10, hidden_dim=32, num_heads=4):
        super().__init__()
        self.T = T
        self.hidden_dim = hidden_dim
        self.grad_to_hidden = nn.Linear(1, hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.hidden_to_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, A):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = torch.zeros(batch_size, n, device=b.device)

        # 收集每步的 hidden state
        hiddens = []  # list of (batch, n, hidden_dim)

        for t in range(self.T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            grad = torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)
            g_mean = grad.mean(dim=-1, keepdim=True)
            g_std = grad.std(dim=-1, keepdim=True) + 1e-6
            g_norm = (grad - g_mean) / g_std

            # 映射到 hidden_dim: (batch, n) -> (batch*n, 1) -> (batch*n, hidden_dim)
            h = self.grad_to_hidden(g_norm.reshape(-1, 1)).reshape(batch_size, n, self.hidden_dim)
            hiddens.append(h)

            # 用当前 hidden 的更新量
            update = self.hidden_to_update(h.reshape(-1, self.hidden_dim)).reshape(batch_size, n) * g_std
            x = x + self.eta * update

        # Transformer 处理整个迭代序列
        # hiddens: list of (batch, n, hidden_dim) -> (batch*n, T, hidden_dim)
        seq = torch.stack(hiddens, dim=1)  # (batch, T, n, hidden_dim)
        seq = seq.permute(0, 2, 1, 3).reshape(batch_size * n, self.T, self.hidden_dim)

        attn_out, _ = self.self_attn(seq, seq, seq)  # (batch*n, T, hidden_dim)
        attn_last = attn_out[:, -1, :]  # (batch*n, hidden_dim)

        # 最终更新
        final_update = self.hidden_to_update(attn_last).reshape(batch_size, n)

        # 重新计算最终梯度
        Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
        grad = torch.bmm(A.transpose(1, 2), (Ax - b).unsqueeze(-1)).squeeze(-1)
        g_std = grad.std(dim=-1, keepdim=True) + 1e-6

        x = x + self.eta * final_update * g_std
        return x


# ============================================================
# W 矩阵可视化
# ============================================================

def visualize_W_matrix(model, save_path='report/W_matrix_analysis.png'):
    """可视化 LISTA-Momentum 的 W 矩阵。"""
    if not hasattr(model, 'layers'):
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, layer_idx in enumerate([0, 4, 9]):
        if layer_idx >= len(model.layers):
            break
        layer = model.layers[layer_idx]

        # 获取 W 矩阵
        if hasattr(layer, 'W'):
            W = layer.W.weight.detach().cpu().numpy()
        else:
            continue

        # W 矩阵热力图
        ax = axes[0, idx]
        im = ax.imshow(W, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Layer {layer_idx+1} W Matrix', fontsize=12)
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        plt.colorbar(im, ax=ax)

        # 特征值分布
        ax = axes[1, idx]
        eigvals = np.linalg.eigvals(W)
        ax.scatter(eigvals.real, eigvals.imag, alpha=0.5, s=10)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
        # 画单位圆
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'r--', linewidth=1, label='Unit Circle')
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f'Layer {layer_idx+1} Eigenvalues', fontsize=12)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  W 矩阵可视化已保存到 {save_path}")


def visualize_training_loss(noiseless_results, noisy_results, save_path='report/training_comparison.png'):
    """可视化无噪声 vs 有噪声的对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 全模型对比
    methods = ['ISTA', 'FISTA', 'LISTA', 'CP', 'Mom', 'DA', 'RNN', 'LSTM', 'LSTM2', 'Trans']
    noiseless_vals = [
        noiseless_results['ista'], noiseless_results['fista'],
        noiseless_results['lista'], noiseless_results['lista_cp'],
        noiseless_results['lista_momentum'], noiseless_results['da_lista'],
        noiseless_results['rnn_lista'], noiseless_results['lstm_lista'],
        noiseless_results['lstm_lista_v2'], noiseless_results['transformer_lista'],
    ]
    noisy_vals = [
        noisy_results['ista'], noisy_results['fista'],
        noisy_results['lista'], noisy_results['lista_cp'],
        noisy_results['lista_momentum'], noisy_results['da_lista'],
        noisy_results['rnn_lista'], noisy_results['lstm_lista'],
        noisy_results['lstm_lista_v2'], noisy_results['transformer_lista'],
    ]

    x = np.arange(len(methods))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width/2, noiseless_vals, width, label='Noiseless', color='steelblue')
    bars2 = ax.bar(x + width/2, noisy_vals, width, label='Noisy', color='coral')
    ax.set_xlabel('Method')
    ax.set_ylabel('Relative Error')
    ax.set_title('All Models: Noiseless vs Noisy')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # 不同 T 对比 (LSTM-LISTA)
    ax = axes[1]
    T_vals = [5, 10, 20]
    if 'different_T' in noiseless_results:
        lstm_noiseless = [noiseless_results['different_T'][str(t)]['lstm_lista'] for t in T_vals]
        lstm_noisy = [noisy_results['different_T'][str(t)]['lstm_lista'] for t in T_vals]
        ista_noiseless = [noiseless_results['different_T'][str(t)]['ista'] for t in T_vals]

        ax.plot(T_vals, ista_noiseless, 'k--', marker='s', label='ISTA', linewidth=2)
        ax.plot(T_vals, lstm_noiseless, 'b-o', label='LSTM (noiseless)', linewidth=2)
        ax.plot(T_vals, lstm_noisy, 'r-s', label='LSTM (noisy)', linewidth=2)
        ax.set_xlabel('Number of Layers T')
        ax.set_ylabel('Relative Error')
        ax.set_title('LSTM-LISTA: Different T')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图表已保存到 {save_path}")


# ============================================================
# 训练函数
# ============================================================

def train_model(model, train_fn, T, noise, num_epochs=50, lr=1e-3, verbose=False, fixed_n=None):
    """通用训练函数。fixed_n: 若指定则只生成该维度的数据。"""
    set_seed(42)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 生成训练数据
    train_samples = []
    rng = np.random.RandomState(42)
    for i in range(1000):
        n = fixed_n if fixed_n else rng.choice([100, 150, 200])
        A = rng.randn(50, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        x = np.zeros(n)
        support = rng.choice(n, 10, replace=False)
        x[support] = rng.randn(10)
        b = A @ x + noise * rng.randn(50)
        train_samples.append((b.astype(np.float32), x.astype(np.float32), A))

    for epoch in range(num_epochs):
        np.random.shuffle(train_samples)
        total_loss = 0
        for b, x, A in train_samples:
            loss = train_fn(model, b, x, A, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/len(train_samples):.6f}")
    return model


def train_lista_basic(m, n, T, noise, num_epochs=50, verbose=False):
    set_seed(42)
    model = LISTABasic(m, n, T)
    def train_fn(model, b, x, A, criterion):
        b_t = torch.FloatTensor(b).unsqueeze(0)
        x_t = torch.FloatTensor(x).unsqueeze(0)
        return criterion(model(b_t), x_t)
    return train_model(model, train_fn, T, noise, num_epochs, verbose=verbose, fixed_n=n)


def train_lista_cp(A, T, noise, num_epochs=50, verbose=False):
    set_seed(42)
    A_tensor = torch.FloatTensor(A)
    model = LISTACP(A_tensor, T)
    n = A.shape[1]
    def train_fn(model, b, x, A_np, criterion):
        b_t = torch.FloatTensor(b).unsqueeze(0)
        x_t = torch.FloatTensor(x).unsqueeze(0)
        return criterion(model(b_t), x_t)
    return train_model(model, train_fn, T, noise, num_epochs, verbose=verbose, fixed_n=n)


def train_generic(model, T, noise, num_epochs=50, verbose=False, fixed_n=200):
    """训练需要 A 矩阵的模型。"""
    def train_fn(model, b, x, A, criterion):
        b_t = torch.FloatTensor(b).unsqueeze(0)
        x_t = torch.FloatTensor(x).unsqueeze(0)
        A_t = torch.FloatTensor(A).unsqueeze(0)
        return criterion(model(b_t, A_t), x_t)
    return train_model(model, train_fn, T, noise, num_epochs, verbose=verbose, fixed_n=fixed_n)


# ============================================================
# 评估函数
# ============================================================

def evaluate(model, A_test, B_test, X_test, needs_A=False):
    """评估模型。"""
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(len(B_test)):
            b = torch.FloatTensor(B_test[i:i+1])
            if needs_A:
                A_t = torch.FloatTensor(A_test).unsqueeze(0)
                x_pred = model(b, A_t)
            else:
                x_pred = model(b)
            errs.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))
    return float(np.mean(errs))


def evaluate_ista_fista(A_test, B_test, X_test, T):
    """评估 ISTA 和 FISTA。"""
    ista_errs, fista_errs = [], []
    for i in range(len(B_test)):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        x_fista, _ = fista(A_test, B_test[i], lam, max_iter=T)
        ista_errs.append(relative_error(X_test[i], x_ista))
        fista_errs.append(relative_error(X_test[i], x_fista))
    return float(np.mean(ista_errs)), float(np.mean(fista_errs))


# ============================================================
# 主实验
# ============================================================

def run_all_experiments():
    results = {}
    sparsity = 10

    for noise_label, noise_std in [("noiseless", 0.0), ("noisy", 0.01)]:
        print(f"\n{'='*70}")
        print(f"  {noise_label} (sigma={noise_std})")
        print(f"{'='*70}")

        results[noise_label] = {}

        # 测试矩阵
        set_seed(42)
        A_test = np.random.randn(50, 200).astype(np.float32)
        A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
        B_test, X_test = gen_data(A_test, 30, sparsity, noise_std, seed=9999)
        A_tensor = torch.FloatTensor(A_test)

        # ---- 实验 1: T=10 的全模型对比 ----
        print(f"\n--- 实验 1: 全模型对比 (T=10, n=200) ---")
        T = 10

        # 经典基线
        ista_err, fista_err = evaluate_ista_fista(A_test, B_test, X_test, T)
        print(f"  ISTA:           {ista_err:.4f}")
        print(f"  FISTA:          {fista_err:.4f}")

        # LISTA (基本)
        print(f"  训练 LISTA...")
        model_lista = train_lista_basic(50, 200, T, noise_std, num_epochs=50)
        lista_err = evaluate(model_lista, A_test, B_test, X_test, needs_A=False)
        print(f"  LISTA:          {lista_err:.4f}")

        # LISTA-CP
        print(f"  训练 LISTA-CP...")
        model_cp = train_lista_cp(A_test, T, noise_std, num_epochs=50)
        cp_err = evaluate(model_cp, A_test, B_test, X_test, needs_A=False)
        print(f"  LISTA-CP:       {cp_err:.4f}")

        # LISTA-Momentum
        print(f"  训练 LISTA-Momentum...")
        model_mom = LISTAMomentum(50, 200, T)
        model_mom = train_generic(model_mom, T, noise_std, num_epochs=50)
        mom_err = evaluate(model_mom, A_test, B_test, X_test, needs_A=True)
        print(f"  LISTA-Momentum: {mom_err:.4f}")

        # 维度无关 LISTA
        print(f"  训练 DA-LISTA...")
        model_da = DimAgnosticLISTA(T)
        model_da = train_generic(model_da, T, noise_std, num_epochs=50)
        da_err = evaluate(model_da, A_test, B_test, X_test, needs_A=True)
        print(f"  DA-LISTA:       {da_err:.4f}")

        # RNN-LISTA
        print(f"  训练 RNN-LISTA...")
        model_rnn = RNNLISTA(T)
        model_rnn = train_generic(model_rnn, T, noise_std, num_epochs=50)
        rnn_err = evaluate(model_rnn, A_test, B_test, X_test, needs_A=True)
        print(f"  RNN-LISTA:      {rnn_err:.4f}")

        # LSTM-LISTA
        print(f"  训练 LSTM-LISTA...")
        model_lstm = LSTMLISTA(T)
        model_lstm = train_generic(model_lstm, T, noise_std, num_epochs=50)
        lstm_err = evaluate(model_lstm, A_test, B_test, X_test, needs_A=True)
        print(f"  LSTM-LISTA:     {lstm_err:.4f}")

        # LSTM-LISTA v2
        print(f"  训练 LSTM-LISTA v2...")
        model_lstm2 = LSTMLISTAv2(T)
        model_lstm2 = train_generic(model_lstm2, T, noise_std, num_epochs=50)
        lstm2_err = evaluate(model_lstm2, A_test, B_test, X_test, needs_A=True)
        print(f"  LSTM-LISTA v2:  {lstm2_err:.4f}")

        # Transformer-LISTA
        print(f"  训练 Transformer-LISTA...")
        model_trans = TransformerLISTA(T)
        model_trans = train_generic(model_trans, T, noise_std, num_epochs=50)
        trans_err = evaluate(model_trans, A_test, B_test, X_test, needs_A=True)
        print(f"  Transformer:    {trans_err:.4f}")

        results[noise_label]['full_comparison'] = {
            'ista': ista_err, 'fista': fista_err,
            'lista': lista_err, 'lista_cp': cp_err,
            'lista_momentum': mom_err, 'da_lista': da_err,
            'rnn_lista': rnn_err, 'lstm_lista': lstm_err,
            'lstm_lista_v2': lstm2_err, 'transformer_lista': trans_err,
        }

        # ---- 实验 2: 不同 T (全部模型) ----
        print(f"\n--- 实验 2: 不同展开层数 ---")
        results[noise_label]['different_T'] = {}
        for T_val in [5, 10, 20]:
            ista_t, fista_t = evaluate_ista_fista(A_test, B_test, X_test, T_val)
            lista_t = evaluate(train_lista_basic(50, 200, T_val, noise_std, 5), A_test, B_test, X_test)
            cp_t = evaluate(train_lista_cp(A_test, T_val, noise_std, 5), A_test, B_test, X_test)
            mom_model = LISTAMomentum(50, 200, T_val)
            mom_t = evaluate(train_generic(mom_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            da_model = DimAgnosticLISTA(T_val)
            da_t = evaluate(train_generic(da_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            rnn_model = RNNLISTA(T_val)
            rnn_t = evaluate(train_generic(rnn_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            lstm_model = LSTMLISTA(T_val)
            lstm_t = evaluate(train_generic(lstm_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            lstm2_model = LSTMLISTAv2(T_val)
            lstm2_t = evaluate(train_generic(lstm2_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            trans_model = TransformerLISTA(T_val)
            trans_t = evaluate(train_generic(trans_model, T_val, noise_std, 5), A_test, B_test, X_test, needs_A=True)
            results[noise_label]['different_T'][str(T_val)] = {
                'ista': ista_t, 'fista': fista_t,
                'lista': lista_t, 'lista_cp': cp_t,
                'lista_momentum': mom_t, 'da_lista': da_t,
                'rnn_lista': rnn_t, 'lstm_lista': lstm_t,
                'lstm_lista_v2': lstm2_t, 'transformer_lista': trans_t,
            }
            print(f"  T={T_val}: ISTA={ista_t:.4f} FISTA={fista_t:.4f} LISTA={lista_t:.4f} CP={cp_t:.4f} Mom={mom_t:.4f} DA={da_t:.4f} RNN={rnn_t:.4f} LSTM={lstm_t:.4f} LSTM2={lstm2_t:.4f} Trans={trans_t:.4f}")

        # ---- 实验 3: 泛化性 ----
        print(f"\n--- 实验 3: 泛化性 ---")
        # ID
        id_results = {}
        for name, model, needs_A in [
            ('ista', None, False), ('fista', None, False),
            ('lista', model_lista, False), ('lista_cp', model_cp, False),
            ('lista_momentum', model_mom, True), ('da_lista', model_da, True),
            ('rnn_lista', model_rnn, True), ('lstm_lista', model_lstm, True),
            ('lstm_lista_v2', model_lstm2, True), ('transformer_lista', model_trans, True),
        ]:
            if model is None:
                if name == 'ista':
                    id_results[name] = results[noise_label]['full_comparison']['ista']
                else:
                    id_results[name] = results[noise_label]['full_comparison']['fista']
            else:
                id_results[name] = evaluate(model, A_test, B_test[:30], X_test[:30], needs_A)

        # OOD
        ood_results = {k: [] for k in id_results}
        for trial in range(3):
            A_ood = np.random.randn(50, 200).astype(np.float32)
            A_ood /= np.linalg.norm(A_ood, axis=0, keepdims=True)
            B_ood, X_ood = gen_data(A_ood, 20, sparsity, noise_std, seed=5000+trial)
            for name, model, needs_A in [
                ('ista', None, False), ('fista', None, False),
                ('lista', model_lista, False), ('lista_cp', model_cp, False),
                ('lista_momentum', model_mom, True), ('da_lista', model_da, True),
                ('rnn_lista', model_rnn, True), ('lstm_lista', model_lstm, True),
                ('lstm_lista_v2', model_lstm2, True), ('transformer_lista', model_trans, True),
            ]:
                if model is None:
                    errs = []
                    for i in range(20):
                        lam = 0.01 * np.max(np.abs(A_ood.T @ B_ood[i]))
                        if name == 'ista':
                            x_pred, _ = ista(A_ood, B_ood[i], lam, max_iter=T)
                        else:
                            x_pred, _ = fista(A_ood, B_ood[i], lam, max_iter=T)
                        errs.append(relative_error(X_ood[i], x_pred))
                    ood_results[name].append(np.mean(errs))
                else:
                    ood_results[name].append(evaluate(model, A_ood, B_ood, X_ood, needs_A))

        results[noise_label]['generalization'] = {}
        for name in id_results:
            ood_mean = float(np.mean(ood_results[name]))
            degradation = ood_mean / (id_results[name] + 1e-10)
            results[noise_label]['generalization'][name] = {
                'id': id_results[name], 'ood': ood_mean, 'degradation': degradation,
            }
            print(f"  {name:20s}: ID={id_results[name]:.4f} OOD={ood_mean:.4f} degrad={degradation:.2f}x")

    # 生成可视化
    print(f"\n--- 生成可视化图表 ---")
    # W 矩阵可视化 (用 noiseless 的 LISTA-Momentum 模型)
    if 'noiseless' in results and 'model_mom' in dir():
        visualize_W_matrix(model_mom)

    # 对比图表
    if 'noiseless' in results and 'noisy' in results:
        visualize_training_loss(
            results['noiseless']['full_comparison'],
            results['noisy']['full_comparison'],
        )

    # 保存
    with open('all_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到 all_experiment_results.json")
    return results


if __name__ == '__main__':
    run_all_experiments()
