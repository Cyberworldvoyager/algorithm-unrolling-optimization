"""
run_complete_experiments.py — 补全维度无关 LISTA 的所有实验
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.metrics import relative_error
from lasso.classical import ista, fista

def set_seed(x):
    torch.manual_seed(x)
    np.random.seed(x)

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

def train_model(T=5, num_epochs=30):
    sparsity = 10
    train_samples = []
    rng = np.random.RandomState(42)
    for n in [100, 200, 300]:
        for i in range(8):
            A = rng.randn(50, n).astype(np.float32)
            A /= np.linalg.norm(A, axis=0, keepdims=True)
            B, X = gen_data(A, 8, sparsity, seed=1000+len(train_samples))
            for j in range(8):
                train_samples.append((B[j:j+1], X[j:j+1], A))

    model = DimensionAgnosticLISTA(T=T, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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

# 实验 1: 不同 T
print('Experiment 1: Different T')
sparsity = 10
T_values = [5, 10, 20]
A_test = np.random.randn(50, 200).astype(np.float32)
A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
B_test, X_test = gen_data(A_test, 30, sparsity, seed=9999)

results = {'fair_comparison': {}}

for T in T_values:
    ista_err, fista_err = [], []
    for i in range(30):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        x_fista, _ = fista(A_test, B_test[i], lam, max_iter=T)
        ista_err.append(relative_error(X_test[i], x_ista))
        fista_err.append(relative_error(X_test[i], x_fista))

    set_seed(42)
    model = train_model(T=T, num_epochs=30)
    model.eval()
    lista_err = []
    with torch.no_grad():
        for i in range(30):
            b = torch.FloatTensor(B_test[i:i+1])
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            lista_err.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))

    results['fair_comparison'][str(T)] = {
        'ista': float(np.mean(ista_err)),
        'fista': float(np.mean(fista_err)),
        'lista': float(np.mean(lista_err)),
        'vs_ista': float(np.mean(ista_err) / (np.mean(lista_err) + 1e-10))
    }
    print(f'  T={T}: ISTA={np.mean(ista_err):.4f}, LISTA={np.mean(lista_err):.4f}, vs_ISTA={np.mean(ista_err)/(np.mean(lista_err)+1e-10):.2f}x')

# 实验 2: 泛化性
print('\nExperiment 2: Generalization')
set_seed(42)
model = train_model(T=10, num_epochs=30)

A_id = np.random.randn(50, 200).astype(np.float32)
A_id /= np.linalg.norm(A_id, axis=0, keepdims=True)
B_id, X_id = gen_data(A_id, 30, sparsity, seed=9999)

ista_id, lista_id = [], []
model.eval()
with torch.no_grad():
    for i in range(30):
        lam = 0.01 * np.max(np.abs(A_id.T @ B_id[i]))
        x_ista, _ = ista(A_id, B_id[i], lam, max_iter=10)
        ista_id.append(relative_error(X_id[i], x_ista))
        b = torch.FloatTensor(B_id[i:i+1])
        A = torch.FloatTensor(A_id).unsqueeze(0)
        x_pred = model(b, A)
        lista_id.append(relative_error(X_id[i], to_numpy(x_pred.squeeze())))

ista_ood, lista_ood = [], []
for trial in range(5):
    A_ood = np.random.randn(50, 200).astype(np.float32)
    A_ood /= np.linalg.norm(A_ood, axis=0, keepdims=True)
    B_ood, X_ood = gen_data(A_ood, 15, sparsity, seed=5000+trial)
    with torch.no_grad():
        for i in range(15):
            lam = 0.01 * np.max(np.abs(A_ood.T @ B_ood[i]))
            x_ista, _ = ista(A_ood, B_ood[i], lam, max_iter=10)
            ista_ood.append(relative_error(X_ood[i], x_ista))
            b = torch.FloatTensor(B_ood[i:i+1])
            A = torch.FloatTensor(A_ood).unsqueeze(0)
            x_pred = model(b, A)
            lista_ood.append(relative_error(X_ood[i], to_numpy(x_pred.squeeze())))

results['generalization'] = {
    'ista_id': float(np.mean(ista_id)),
    'ista_ood': float(np.mean(ista_ood)),
    'lista_id': float(np.mean(lista_id)),
    'lista_ood': float(np.mean(lista_ood)),
    'degradation': float(np.mean(lista_ood) / (np.mean(lista_id) + 1e-10)),
    'vs_ista': float(np.mean(ista_id) / (np.mean(lista_id) + 1e-10))
}
print(f'  ID: ISTA={np.mean(ista_id):.4f}, LISTA={np.mean(lista_id):.4f}')
print(f'  OOD: ISTA={np.mean(ista_ood):.4f}, LISTA={np.mean(lista_ood):.4f}')
print(f'  vs ISTA: {np.mean(ista_id)/(np.mean(lista_id)+1e-10):.2f}x')

# 实验 3: 噪声鲁棒性
print('\nExperiment 3: Noise Robustness')
noise_levels = [0.001, 0.01, 0.05, 0.1]
results['noise'] = {}

for noise in noise_levels:
    ista_err, lista_err = [], []
    for i in range(20):
        A_t = np.random.randn(50, 200).astype(np.float32)
        A_t /= np.linalg.norm(A_t, axis=0, keepdims=True)
        B_t, X_t = gen_data(A_t, 1, sparsity, noise, seed=8000+i)
        lam = 0.01 * np.max(np.abs(A_t.T @ B_t[0]))
        x_ista, _ = ista(A_t, B_t[0], lam, max_iter=10)
        ista_err.append(relative_error(X_t[0], x_ista))
        with torch.no_grad():
            b = torch.FloatTensor(B_t)
            A = torch.FloatTensor(A_t).unsqueeze(0)
            x_pred = model(b, A)
            lista_err.append(relative_error(X_t[0], to_numpy(x_pred.squeeze())))

    results['noise'][str(noise)] = {
        'ista': float(np.mean(ista_err)),
        'lista': float(np.mean(lista_err)),
        'vs_ista': float(np.mean(ista_err) / (np.mean(lista_err) + 1e-10))
    }
    print(f'  sigma={noise:.3f}: ISTA={np.mean(ista_err):.4f}, LISTA={np.mean(lista_err):.4f}, vs_ISTA={np.mean(ista_err)/(np.mean(lista_err)+1e-10):.2f}x')

with open('dimension_agnostic_complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nResults saved!')
