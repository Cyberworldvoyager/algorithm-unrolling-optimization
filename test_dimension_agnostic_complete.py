"""
test_dimension_agnostic_complete.py — 补全维度无关 LISTA 的所有实验
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.metrics import relative_error
from lasso.classical import ista, fista

set_seed = lambda x: (torch.manual_seed(x), np.random.seed(x))
set_seed(42)

def gen_data(A, num, sparsity=10, noise=0.001, seed=0):
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

def to_numpy(t):
    return t.detach().cpu().numpy()

# 维度无关 LISTA
class DimensionAgnosticLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
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

class DimensionAgnosticLISTA(nn.Module):
    def __init__(self, T=10, hidden_dim=32):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([DimensionAgnosticLayer(hidden_dim) for _ in range(T)])

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        x_prev = x.clone()
        for layer in self.layers:
            x_new = layer(b, x, x_prev, A)
            x_prev = x
            x = x_new
        return x

def train_model(T=5, hidden_dim=32, num_epochs=40, lr=1e-3):
    """训练维度无关 LISTA"""
    sparsity = 10
    train_samples = []
    rng = np.random.RandomState(42)

    for n in [100, 200, 300]:
        for i in range(10):
            A = rng.randn(50, n).astype(np.float32)
            A /= np.linalg.norm(A, axis=0, keepdims=True)
            B, X = gen_data(A, 10, sparsity, seed=1000+len(train_samples))
            for j in range(10):
                train_samples.append((B[j:j+1], X[j:j+1], A))

    model = DimensionAgnosticLISTA(T=T, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_samples)
        for b, x, A in train_samples:
            b_t = torch.FloatTensor(b)
            x_t = torch.FloatTensor(x)
            A_t = torch.FloatTensor(A).unsqueeze(0)
            x_pred = model(b_t, A_t)
            loss = criterion(x_pred, x_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

# ============================================================
# 实验 1: 不同 T 的对比
# ============================================================
print("="*60)
print("Experiment 1: Fair Comparison at Different T")
print("="*60)

sparsity = 10
T_values = [5, 10, 20]
A_test = np.random.randn(50, 200).astype(np.float32)
A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
B_test, X_test = gen_data(A_test, 50, sparsity, seed=9999)

# ISTA/FISTA baseline
ista_errors = {T: [] for T in T_values}
fista_errors = {T: [] for T in T_values}

for T in T_values:
    for i in range(50):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        x_fista, _ = fista(A_test, B_test[i], lam, max_iter=T)
        ista_errors[T].append(relative_error(X_test[i], x_ista))
        fista_errors[T].append(relative_error(X_test[i], x_fista))

# 训练并测试不同 T 的维度无关 LISTA
lista_errors = {}
for T in T_values:
    print(f"\nTraining LISTA-DimAgnostic T={T}...")
    set_seed(42)
    model = train_model(T=T, hidden_dim=32, num_epochs=40)

    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(50):
            b = torch.FloatTensor(B_test[i:i+1])
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            errors.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))
    lista_errors[T] = errors

# 打印结果
print("\n" + "="*60)
print("Results: Fair Comparison at Different T")
print("="*60)
print(f"{'T':<5} {'ISTA':<10} {'FISTA':<10} {'LISTA-DimAgn':<15} {'vs ISTA':<10}")
print("-"*50)
for T in T_values:
    ista_mean = np.mean(ista_errors[T])
    fista_mean = np.mean(fista_errors[T])
    lista_mean = np.mean(lista_errors[T])
    vs_ista = ista_mean / (lista_mean + 1e-10)
    print(f"{T:<5} {ista_mean:<10.4f} {fista_mean:<10.4f} {lista_mean:<15.4f} {vs_ista:<10.2f}x")

# ============================================================
# 实验 2: 泛化性对比 (ID vs OOD)
# ============================================================
print("\n" + "="*60)
print("Experiment 2: Generalization (ID vs OOD)")
print("="*60)

# 训练模型 (T=10)
print("\nTraining LISTA-DimAgnostic T=10...")
set_seed(42)
model = train_model(T=10, hidden_dim=32, num_epochs=40)

# ID 测试 (训练维度 n=200)
A_id = np.random.randn(50, 200).astype(np.float32)
A_id /= np.linalg.norm(A_id, axis=0, keepdims=True)
B_id, X_id = gen_data(A_id, 50, sparsity, seed=9999)

# OOD 测试 (不同 A，相同维度)
ood_errors = []
for trial in range(10):
    A_ood = np.random.randn(50, 200).astype(np.float32)
    A_ood /= np.linalg.norm(A_ood, axis=0, keepdims=True)
    B_ood, X_ood = gen_data(A_ood, 20, sparsity, seed=5000+trial)

    model.eval()
    with torch.no_grad():
        for i in range(20):
            b = torch.FloatTensor(B_ood[i:i+1])
            A = torch.FloatTensor(A_ood).unsqueeze(0)
            x_pred = model(b, A)
            ood_errors.append(relative_error(X_ood[i], to_numpy(x_pred.squeeze())))

# ID 测试
model.eval()
id_errors = []
with torch.no_grad():
    for i in range(50):
        b = torch.FloatTensor(B_id[i:i+1])
        A = torch.FloatTensor(A_id).unsqueeze(0)
        x_pred = model(b, A)
        id_errors.append(relative_error(X_id[i], to_numpy(x_pred.squeeze())))

# ISTA baseline
ista_id_errors = []
ista_ood_errors = []
for i in range(50):
    lam = 0.01 * np.max(np.abs(A_id.T @ B_id[i]))
    x_ista, _ = ista(A_id, B_id[i], lam, max_iter=10)
    ista_id_errors.append(relative_error(X_id[i], x_ista))

for trial in range(10):
    A_ood = np.random.randn(50, 200).astype(np.float32)
    A_ood /= np.linalg.norm(A_ood, axis=0, keepdims=True)
    B_ood, X_ood = gen_data(A_ood, 20, sparsity, seed=5000+trial)
    for i in range(20):
        lam = 0.01 * np.max(np.abs(A_ood.T @ B_ood[i]))
        x_ista, _ = ista(A_ood, B_ood[i], lam, max_iter=10)
        ista_ood_errors.append(relative_error(X_ood[i], x_ista))

print("\n" + "="*60)
print("Results: Generalization (ID vs OOD)")
print("="*60)
print(f"{'Method':<25} {'ID':<10} {'OOD':<10} {'Degradation':<15} {'vs ISTA':<10}")
print("-"*70)
print(f"{'ISTA':<25} {np.mean(ista_id_errors):<10.4f} {np.mean(ista_ood_errors):<10.4f} {'1.00x':<15} {'Baseline':<10}")
lista_id_mean = np.mean(id_errors)
lista_ood_mean = np.mean(ood_errors)
degradation = lista_ood_mean / (lista_id_mean + 1e-10)
vs_ista = np.mean(ista_id_errors) / (lista_id_mean + 1e-10)
print(f"{'LISTA-DimAgnostic':<25} {lista_id_mean:<10.4f} {lista_ood_mean:<10.4f} {degradation:<15.2f}x {vs_ista:<10.2f}x")

# ============================================================
# 实验 3: 噪声鲁棒性
# ============================================================
print("\n" + "="*60)
print("Experiment 3: Noise Robustness")
print("="*60)

noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
noise_results = {}

for noise in noise_levels:
    lista_err = []
    ista_err = []

    for i in range(30):
        A_test = np.random.randn(50, 200).astype(np.float32)
        A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
        B_test, X_test = gen_data(A_test, 1, sparsity, noise, seed=8000+i)

        # ISTA
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[0]))
        x_ista, _ = ista(A_test, B_test[0], lam, max_iter=10)
        ista_err.append(relative_error(X_test[0], x_ista))

        # LISTA
        model.eval()
        with torch.no_grad():
            b = torch.FloatTensor(B_test)
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            lista_err.append(relative_error(X_test[0], to_numpy(x_pred.squeeze())))

    noise_results[noise] = {
        'ista': float(np.mean(ista_err)),
        'lista': float(np.mean(lista_err)),
        'vs_ista': float(np.mean(ista_err) / (np.mean(lista_err) + 1e-10))
    }

print("\n" + "="*60)
print("Results: Noise Robustness")
print("="*60)
print(f"{'Noise σ':<10} {'ISTA':<10} {'LISTA-DimAgn':<15} {'vs ISTA':<10}")
print("-"*45)
for noise, res in noise_results.items():
    print(f"{noise:<10.3f} {res['ista']:<10.4f} {res['lista']:<15.4f} {res['vs_ista']:<10.2f}x")

# ============================================================
# 保存结果
# ============================================================
results = {
    'fair_comparison': {
        str(T): {
            'ista': float(np.mean(ista_errors[T])),
            'fista': float(np.mean(fista_errors[T])),
            'lista': float(np.mean(lista_errors[T])),
            'vs_ista': float(np.mean(ista_errors[T]) / (np.mean(lista_errors[T]) + 1e-10))
        } for T in T_values
    },
    'generalization': {
        'ista_id': float(np.mean(ista_id_errors)),
        'ista_ood': float(np.mean(ista_ood_errors)),
        'lista_id': float(lista_id_mean),
        'lista_ood': float(lista_ood_mean),
        'degradation': float(degradation),
        'vs_ista': float(vs_ista)
    },
    'noise_robustness': noise_results
}

with open('dimension_agnostic_complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to dimension_agnostic_complete_results.json")
