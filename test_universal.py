"""
test_universal.py — 测试通用 LISTA 架构

目标：让网络学会"适用于一类 LASSO 问题的通用优化算法"
"""

import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy
from common.metrics import relative_error
from lasso.classical import ista, fista
from lasso.lista_universal import create_universal_lista

set_seed(42)

m, n = 50, 200
sparsity = 10
T = 10
noise_std = 0.001

def gen_data(A, num, sparsity=10, noise=0.001, seed=0):
    rng = np.random.RandomState(seed)
    m, n = A.shape
    B, X = [], []
    for _ in range(num):
        x = np.zeros(n)
        support = rng.choice(n, sparsity, replace=False)
        x[support] = rng.randn(sparsity)
        b = A @ x + noise * rng.randn(m)
        B.append(b)
        X.append(x)
    return np.array(B), np.array(X)

# ============================================================
# 生成训练数据：多个 A 矩阵
# ============================================================
print("Generating training data...")
num_A_train = 50
A_train_list = []
rng = np.random.RandomState(42)
for i in range(num_A_train):
    A = rng.randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    A_train_list.append(A)

# 训练数据
B_train_all, X_train_all, A_train_all = [], [], []
for i, A in enumerate(A_train_list):
    B, X = gen_data(A, 100, sparsity, noise_std, seed=1000+i)
    B_train_all.append(B)
    X_train_all.append(X)
    A_train_all.append(np.tile(A, (100, 1, 1)))
B_train = np.concatenate(B_train_all)
X_train = np.concatenate(X_train_all)
A_train = np.concatenate(A_train_all)

# 验证数据
B_val, X_val, A_val = [], [], []
for i, A in enumerate(A_train_list[:10]):
    B, X = gen_data(A, 50, sparsity, noise_std, seed=5000+i)
    B_val.append(B)
    X_val.append(X)
    A_val.append(np.tile(A, (50, 1, 1)))
B_val = np.concatenate(B_val)
X_val = np.concatenate(X_val)
A_val = np.concatenate(A_val)

# ============================================================
# 测试数据
# ============================================================
# ID: 训练集中的 A
A_test_id = A_train_list[0]
B_test_id, X_test_id = gen_data(A_test_id, 200, sparsity, noise_std, seed=9999)

# OOD: 训练集外的 A
B_test_ood, X_test_ood = [], []
A_test_ood_list = []
for i in range(20):
    A = np.random.RandomState(5000+i).randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    B, X = gen_data(A, 50, sparsity, noise_std, seed=6000+i)
    B_test_ood.append(B)
    X_test_ood.append(X)
    A_test_ood_list.append(A)
B_test_ood = np.concatenate(B_test_ood)
X_test_ood = np.concatenate(X_test_ood)

# ============================================================
# ISTA/FISTA baseline
# ============================================================
print("\n=== Baseline: ISTA/FISTA ===")
ista_id, ista_ood = [], []
fista_id, fista_ood = [], []
for i in range(200):
    lam = 0.01 * np.max(np.abs(A_test_id.T @ B_test_id[i]))
    x_ista, _ = ista(A_test_id, B_test_id[i], lam, max_iter=T)
    x_fista, _ = fista(A_test_id, B_test_id[i], lam, max_iter=T)
    ista_id.append(relative_error(X_test_id[i], x_ista))
    fista_id.append(relative_error(X_test_id[i], x_fista))
for i in range(len(B_test_ood)):
    A_idx = i // 50
    A = A_test_ood_list[A_idx]
    lam = 0.01 * np.max(np.abs(A.T @ B_test_ood[i]))
    x_ista, _ = ista(A, B_test_ood[i], lam, max_iter=T)
    x_fista, _ = fista(A, B_test_ood[i], lam, max_iter=T)
    ista_ood.append(relative_error(X_test_ood[i], x_ista))
    fista_ood.append(relative_error(X_test_ood[i], x_fista))

print(f"ISTA  - ID: {np.mean(ista_id):.4f}, OOD: {np.mean(ista_ood):.4f}")
print(f"FISTA - ID: {np.mean(fista_id):.4f}, OOD: {np.mean(fista_ood):.4f}")

# ============================================================
# 测试通用 LISTA 架构
# ============================================================
architectures = {
    'adab': {'hidden_dim': 32},
    'shared': {},
    'momentum': {},
}

results = {}

for arch_name, kwargs in architectures.items():
    print(f"\n=== {arch_name.upper()} ===")
    set_seed(42)

    model = create_universal_lista(arch_name, m, n, T=T, **kwargs)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    # DataLoader
    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(B_train), torch.FloatTensor(X_train), torch.FloatTensor(A_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(B_val), torch.FloatTensor(X_val), torch.FloatTensor(A_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

    # 训练
    best_val = float('inf')
    patience_counter = 0
    best_state = None
    t0 = time.time()

    for epoch in range(100):
        model.train()
        for b_batch, x_batch, a_batch in train_loader:
            x_pred = model(b_batch, a_batch)
            loss = criterion(x_pred, x_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_batch, x_batch, a_batch in val_loader:
                val_loss += criterion(model(b_batch, a_batch), x_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: val_loss={val_loss:.8f}")

    if best_state:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    # 测试 ID
    model.eval()
    id_errors = []
    with torch.no_grad():
        for i in range(200):
            b = torch.FloatTensor(B_test_id[i:i+1])
            A = torch.FloatTensor(A_test_id).unsqueeze(0)
            x_pred = model(b, A)
            id_errors.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))

    # 测试 OOD
    ood_errors = []
    with torch.no_grad():
        for i in range(len(B_test_ood)):
            b = torch.FloatTensor(B_test_ood[i:i+1])
            A = torch.FloatTensor(A_test_ood_list[i // 50]).unsqueeze(0)
            x_pred = model(b, A)
            ood_errors.append(relative_error(X_test_ood[i], to_numpy(x_pred.squeeze())))

    results[arch_name] = {
        'id': float(np.mean(id_errors)),
        'ood': float(np.mean(ood_errors)),
        'degradation': float(np.mean(ood_errors) / np.mean(id_errors)),
        'params': params,
        'train_time': train_time,
    }

    print(f"  ID:  {np.mean(id_errors):.4f}")
    print(f"  OOD: {np.mean(ood_errors):.4f}")
    print(f"  Degradation: {np.mean(ood_errors)/np.mean(id_errors):.2f}x")
    print(f"  Train time: {train_time:.1f}s")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Method':<20} {'ID':>8} {'OOD':>8} {'Degrad':>8} {'vs ISTA ID':>12}")
print("-"*60)
print(f"{'ISTA':<20} {np.mean(ista_id):>8.4f} {np.mean(ista_ood):>8.4f} {'1.00x':>8} {'1.00x':>12}")
print(f"{'FISTA':<20} {np.mean(fista_id):>8.4f} {np.mean(fista_ood):>8.4f} {'1.00x':>8} {np.mean(ista_id)/np.mean(fista_id):>11.2f}x")
for arch_name, res in results.items():
    vs_ista = np.mean(ista_id) / res['id']
    print(f"{arch_name:<20} {res['id']:>8.4f} {res['ood']:>8.4f} {res['degradation']:>7.2f}x {vs_ista:>11.2f}x")

# 保存结果
import json
output = {
    'ista': {'id': float(np.mean(ista_id)), 'ood': float(np.mean(ista_ood))},
    'fista': {'id': float(np.mean(fista_id)), 'ood': float(np.mean(fista_ood))},
    **results
}
with open('universal_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\nResults saved to universal_results.json")
