"""
test_sequence_models.py — 测试序列模型 LISTA
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.metrics import relative_error
from lasso.classical import ista, fista
from lasso.lista_sequence import create_sequence_lista

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

def train_model(variant, n=200, T=10, epochs=20):
    """训练序列模型 LISTA"""
    sparsity = 10
    train_samples = []
    rng = np.random.RandomState(42)

    for i in range(20):
        A = rng.randn(50, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 10, sparsity, seed=1000+i)
        for j in range(10):
            train_samples.append((B[j:j+1], X[j:j+1], A))

    model = create_sequence_lista(variant, n, T, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
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
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return model

# ============================================================
# 实验: 序列模型对比
# ============================================================
print("="*60)
print("Experiment: Sequence Models Comparison")
print("="*60)

sparsity = 10
n = 200
T = 10

# 测试数据
A_test = np.random.randn(50, n).astype(np.float32)
A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
B_test, X_test = gen_data(A_test, 20, sparsity, seed=9999)

# ISTA/FISTA baseline
ista_errors, fista_errors = [], []
for i in range(20):
    lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
    x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
    x_fista, _ = fista(A_test, B_test[i], lam, max_iter=T)
    ista_errors.append(relative_error(X_test[i], x_ista))
    fista_errors.append(relative_error(X_test[i], x_fista))

print(f"\nISTA  (T={T}): {np.mean(ista_errors):.4f}")
print(f"FISTA (T={T}): {np.mean(fista_errors):.4f}")

# 训练并测试序列模型
variants = ['rnn', 'lstm', 'transformer']
results = {}

for variant in variants:
    print(f"\nTraining {variant.upper()}-LISTA...")
    set_seed(42)
    model = train_model(variant, n, T, epochs=20)

    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(20):
            b = torch.FloatTensor(B_test[i:i+1])
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            errors.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))

    params = sum(p.numel() for p in model.parameters())
    results[variant] = {
        'error': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'params': params,
        'vs_ista': float(np.mean(ista_errors) / (np.mean(errors) + 1e-10))
    }
    print(f"  {variant.upper()}: error={np.mean(errors):.4f}, params={params}, vs_ista={np.mean(ista_errors)/(np.mean(errors)+1e-10):.2f}x")

# 保存结果
output = {
    'ista': float(np.mean(ista_errors)),
    'fista': float(np.mean(fista_errors)),
    **results
}
with open('sequence_models_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved!")

# 打印总结
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<20} {'Error':<10} {'Params':<10} {'vs ISTA':<10}")
print("-"*50)
print(f"{'ISTA':<20} {np.mean(ista_errors):<10.4f} {'N/A':<10} {'1.00x':<10}")
print(f"{'FISTA':<20} {np.mean(fista_errors):<10.4f} {'N/A':<10} {np.mean(ista_errors)/np.mean(fista_errors):<10.2f}x")
for variant in variants:
    r = results[variant]
    print(f"{variant.upper()+'-LISTA':<20} {r['error']:<10.4f} {r['params']:<10} {r['vs_ista']:<10.2f}x")
