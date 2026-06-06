import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy
from common.metrics import relative_error
from lasso.classical import ista
from lasso.lista import LISTACP

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

# 测试不同训练策略
strategies = {
    'single_A': {'num_A': 1, 'samples_per_A': 2000},
    'multi_A_5': {'num_A': 5, 'samples_per_A': 400},
    'multi_A_20': {'num_A': 20, 'samples_per_A': 100},
    'multi_A_50': {'num_A': 50, 'samples_per_A': 40},
}

# 测试数据
rng = np.random.RandomState(42)
A_test_id = rng.randn(m, n)
A_test_id /= np.linalg.norm(A_test_id, axis=0, keepdims=True)
B_test_id, X_test_id = gen_data(A_test_id, 200, sparsity, noise_std, seed=9999)

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

# ISTA baseline
ista_id, ista_ood = [], []
for i in range(200):
    lam = 0.01 * np.max(np.abs(A_test_id.T @ B_test_id[i]))
    x_ista, _ = ista(A_test_id, B_test_id[i], lam, max_iter=T)
    ista_id.append(relative_error(X_test_id[i], x_ista))
for i in range(len(B_test_ood)):
    A_idx = i // 50
    A = A_test_ood_list[A_idx]
    lam = 0.01 * np.max(np.abs(A.T @ B_test_ood[i]))
    x_ista, _ = ista(A, B_test_ood[i], lam, max_iter=T)
    ista_ood.append(relative_error(X_test_ood[i], x_ista))
print(f'ISTA ID: {np.mean(ista_id):.4f}, OOD: {np.mean(ista_ood):.4f}')
print()

results = {}

for strategy_name, config in strategies.items():
    print(f'=== {strategy_name.upper()} ===')
    num_A = config['num_A']
    samples_per_A = config['samples_per_A']

    # 生成训练 A
    rng_train = np.random.RandomState(42)
    A_train_list = []
    for i in range(num_A):
        A = rng_train.randn(m, n)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        A_train_list.append(A)

    # 确保第一个 A 是测试用的 A (ID 测试)
    A_train_list[0] = A_test_id

    # 生成训练数据
    B_train_all, X_train_all = [], []
    for i, A in enumerate(A_train_list):
        B, X = gen_data(A, samples_per_A, sparsity, noise_std, seed=1000+i)
        B_train_all.append(B)
        X_train_all.append(X)
    B_train = np.concatenate(B_train_all)
    X_train = np.concatenate(X_train_all)

    # 验证数据
    B_val, X_val = [], []
    for i, A in enumerate(A_train_list[:min(5, num_A)]):
        B, X = gen_data(A, 50, sparsity, noise_std, seed=5000+i)
        B_val.append(B)
        X_val.append(X)
    B_val = np.concatenate(B_val)
    X_val = np.concatenate(X_val)

    # 使用平均 A 初始化 (关键设计)
    A_mean = np.mean(A_train_list, axis=0)
    A_tensor = torch.FloatTensor(A_mean)

    set_seed(42)
    model = LISTACP(A_tensor, T=T, init_eta=0.1)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(B_train), torch.FloatTensor(X_train))
    val_ds = torch.utils.data.TensorDataset(torch.FloatTensor(B_val), torch.FloatTensor(X_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

    best_val = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(100):
        model.train()
        for b_batch, x_batch in train_loader:
            x_pred = model(b_batch)
            loss = criterion(x_pred, x_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b_batch, x_batch in val_loader:
                val_loss += criterion(model(b_batch), x_batch).item()
        val_loss /= len(val_loader)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 20:
                break

    if best_state:
        model.load_state_dict(best_state)

    # 测试 ID
    model.eval()
    id_errors = []
    with torch.no_grad():
        for i in range(200):
            b = torch.FloatTensor(B_test_id[i:i+1])
            x_pred = model(b)
            id_errors.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))

    # 测试 OOD
    ood_errors = []
    with torch.no_grad():
        for i in range(len(B_test_ood)):
            b = torch.FloatTensor(B_test_ood[i:i+1])
            x_pred = model(b)
            ood_errors.append(relative_error(X_test_ood[i], to_numpy(x_pred.squeeze())))

    results[strategy_name] = {
        'id': float(np.mean(id_errors)),
        'ood': float(np.mean(ood_errors)),
        'degradation': float(np.mean(ood_errors) / np.mean(id_errors)),
    }

    print(f'  Params: {params}')
    print(f'  ID:  {np.mean(id_errors):.4f}')
    print(f'  OOD: {np.mean(ood_errors):.4f}')
    print(f'  Degradation: {np.mean(ood_errors)/np.mean(id_errors):.2f}x')
    print()

# 保存结果
import json
output = {'ista': {'id': float(np.mean(ista_id)), 'ood': float(np.mean(ista_ood))}, **results}
with open('multi_a_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print('Done!')
