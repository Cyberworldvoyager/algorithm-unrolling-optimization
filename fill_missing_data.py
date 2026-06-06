"""
fill_missing_data.py — 补全报告中缺失的实验数据
"""

import sys, time, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy, count_parameters
from common.metrics import relative_error
from lasso.classical import ista, fista
from lasso.lista_universal import create_universal_lista

set_seed(42)

# ============================================================
# LASSO 实验设置
# ============================================================
m, n = 50, 200
sparsity = 10
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

# 生成训练用 A
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

B_val, X_val, A_val = [], [], []
for i, A in enumerate(A_train_list[:10]):
    B, X = gen_data(A, 50, sparsity, noise_std, seed=5000+i)
    B_val.append(B)
    X_val.append(X)
    A_val.append(np.tile(A, (50, 1, 1)))
B_val = np.concatenate(B_val)
X_val = np.concatenate(X_val)
A_val = np.concatenate(A_val)

# 测试数据
A_test_id = A_train_list[0]
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

# ISTA/FISTA baseline
ista_id, ista_ood = [], []
fista_id, fista_ood = [], []
for i in range(200):
    lam = 0.01 * np.max(np.abs(A_test_id.T @ B_test_id[i]))
    x_ista, _ = ista(A_test_id, B_test_id[i], lam, max_iter=10)
    x_fista, _ = fista(A_test_id, B_test_id[i], lam, max_iter=10)
    ista_id.append(relative_error(X_test_id[i], x_ista))
    fista_id.append(relative_error(X_test_id[i], x_fista))
for i in range(len(B_test_ood)):
    A = A_test_ood_list[i // 50]
    lam = 0.01 * np.max(np.abs(A.T @ B_test_ood[i]))
    x_ista, _ = ista(A, B_test_ood[i], lam, max_iter=10)
    x_fista, _ = fista(A, B_test_ood[i], lam, max_iter=10)
    ista_ood.append(relative_error(X_test_ood[i], x_ista))
    fista_ood.append(relative_error(X_test_ood[i], x_fista))

print("=== ISTA/FISTA Baseline ===")
print(f"ISTA  - ID: {np.mean(ista_id):.4f}, OOD: {np.mean(ista_ood):.4f}")
print(f"FISTA - ID: {np.mean(fista_id):.4f}, OOD: {np.mean(fista_ood):.4f}")
print()

results = {
    'ista': {'id': float(np.mean(ista_id)), 'ood': float(np.mean(ista_ood))},
    'fista': {'id': float(np.mean(fista_id)), 'ood': float(np.mean(fista_ood))},
}

# ============================================================
# 实验 1: LISTA-Momentum 不同层数
# ============================================================
print("=== Experiment 1: LISTA-Momentum at different T ===")

for T in [5, 10, 20]:
    print(f"\n--- T = {T} ---")
    set_seed(42)

    model = create_universal_lista('momentum', m, n, T=T)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(B_train), torch.FloatTensor(X_train), torch.FloatTensor(A_train))
    val_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(B_val), torch.FloatTensor(X_val), torch.FloatTensor(A_val))
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

    best_val = float('inf')
    patience_counter = 0
    best_state = None

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

    if best_state:
        model.load_state_dict(best_state)

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

    results[f'lista_momentum_T{T}'] = {
        'id': float(np.mean(id_errors)),
        'ood': float(np.mean(ood_errors)),
        'degradation': float(np.mean(ood_errors) / np.mean(id_errors)),
        'params': params,
    }

    print(f"  ID:  {np.mean(id_errors):.4f}")
    print(f"  OOD: {np.mean(ood_errors):.4f}")
    print(f"  Degradation: {np.mean(ood_errors)/np.mean(id_errors):.2f}x")

# ============================================================
# 实验 2: LISTA-Momentum 噪声鲁棒性
# ============================================================
print("\n=== Experiment 2: LISTA-Momentum Noise Robustness ===")

# 使用已训练的 T=10 模型
model_10 = create_universal_lista('momentum', m, n, T=10)
# 需要重新训练
set_seed(42)
optimizer = torch.optim.Adam(model_10.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.MSELoss()

train_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(B_train), torch.FloatTensor(X_train), torch.FloatTensor(A_train))
val_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(B_val), torch.FloatTensor(X_val), torch.FloatTensor(A_val))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256)

best_val = float('inf')
patience_counter = 0
best_state = None

for epoch in range(100):
    model_10.train()
    for b_batch, x_batch, a_batch in train_loader:
        x_pred = model_10(b_batch, a_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_10.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model_10.eval()
    val_loss = 0
    with torch.no_grad():
        for b_batch, x_batch, a_batch in val_loader:
            val_loss += criterion(model_10(b_batch, a_batch), x_batch).item()
    val_loss /= len(val_loader)

    if val_loss < best_val - 1e-8:
        best_val = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model_10.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

if best_state:
    model_10.load_state_dict(best_state)

# 测试不同噪声水平
noise_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
noise_results = {}

for noise in noise_levels:
    errors = []
    model_10.eval()
    with torch.no_grad():
        for i in range(100):
            B_test, X_test = gen_data(A_test_id, 1, sparsity, noise, seed=8000+i)
            b = torch.FloatTensor(B_test)
            A = torch.FloatTensor(A_test_id).unsqueeze(0)
            x_pred = model_10(b, A)
            errors.append(relative_error(X_test[0], to_numpy(x_pred.squeeze())))
    noise_results[noise] = float(np.mean(errors))
    print(f"  σ={noise:.3f}: error={np.mean(errors):.4f}")

results['noise_robustness'] = noise_results

# ============================================================
# 实验 3: 低秩矩阵恢复
# ============================================================
print("\n=== Experiment 3: Low-rank Matrix Recovery ===")

from low_rank.problem import generate_matrix_completion_data
from low_rank.classical import admm_matrix_completion
from low_rank.admm_net import ADMMNet, ADMMNetV2, SoftImputeNet

m_lr, n_lr = 50, 50
rank = 5
ratio = 0.5

# 生成数据
B_lr_train, X_lr_train, M_lr_train = [], [], []
for i in range(500):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=1000+i)
    B_lr_train.append(data['M_observed'])
    X_lr_train.append(data['mask'])
    M_lr_train.append(data['M'])

B_lr_val, X_lr_val, M_lr_val = [], [], []
for i in range(100):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=5000+i)
    B_lr_val.append(data['M_observed'])
    X_lr_val.append(data['mask'])
    M_lr_val.append(data['M'])

# ADMM baseline
print("\nADMM baseline...")
admm_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    X_admm, _ = admm_matrix_completion(data['M_observed'], data['mask'], rho=1.0, max_iter=200)
    admm_errors.append(relative_error(data['M'], X_admm))
print(f"  ADMM (200 iter): error={np.mean(admm_errors):.4f}")

# 训练 ADMMNet
print("\nTraining ADMMNet...")
set_seed(42)
model_admm = ADMMNet(m_lr, n_lr, T=10)
optimizer = torch.optim.Adam(model_admm.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(B_lr_train)),
    torch.FloatTensor(np.array(X_lr_train)),
    torch.FloatTensor(np.array(M_lr_train)))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

for epoch in range(50):
    model_admm.train()
    for b_batch, mask_batch, m_batch in train_loader:
        m_pred = model_admm(b_batch, mask_batch)
        loss = criterion(m_pred, m_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_admm.eval()
admm_net_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    with torch.no_grad():
        m_obs = torch.FloatTensor(data['M_observed']).unsqueeze(0)
        mask = torch.FloatTensor(data['mask']).unsqueeze(0)
        m_pred = model_admm(m_obs, mask)
        admm_net_errors.append(relative_error(data['M'], to_numpy(m_pred.squeeze())))
print(f"  ADMMNet (10 layers): error={np.mean(admm_net_errors):.4f}")

# 训练 ADMMNetV2
print("\nTraining ADMMNetV2...")
set_seed(42)
model_admm_v2 = ADMMNetV2(m_lr, n_lr, T=10)
optimizer = torch.optim.Adam(model_admm_v2.parameters(), lr=1e-3)

for epoch in range(50):
    model_admm_v2.train()
    for b_batch, mask_batch, m_batch in train_loader:
        m_pred = model_admm_v2(b_batch, mask_batch)
        loss = criterion(m_pred, m_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_admm_v2.eval()
admm_v2_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    with torch.no_grad():
        m_obs = torch.FloatTensor(data['M_observed']).unsqueeze(0)
        mask = torch.FloatTensor(data['mask']).unsqueeze(0)
        m_pred = model_admm_v2(m_obs, mask)
        admm_v2_errors.append(relative_error(data['M'], to_numpy(m_pred.squeeze())))
print(f"  ADMMNetV2 (10 layers): error={np.mean(admm_v2_errors):.4f}")

results['low_rank'] = {
    'admm': float(np.mean(admm_errors)),
    'admm_net': float(np.mean(admm_net_errors)),
    'admm_net_v2': float(np.mean(admm_v2_errors)),
}

# ============================================================
# 实验 4: 二次规划
# ============================================================
print("\n=== Experiment 4: Quadratic Programming ===")

from qp.problem import generate_qp_data, compute_optimal_solution, qp_objective
from qp.classical import pgd
from qp.pgd_net import PGDNet, PGDMomentumNet

n_qp = 50
constraint_type = 'box'

# PGD baseline
print("\nPGD baseline...")
pgd_gaps = []
for i in range(50):
    data = generate_qp_data(n_qp, constraint_type, seed=9000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    obj_opt = qp_objective(data['Q'], data['c'], x_opt)
    x_pgd, _ = pgd(data['Q'], data['c'], constraint_type, data['constraint_params'], max_iter=100)
    obj_pgd = qp_objective(data['Q'], data['c'], x_pgd)
    pgd_gaps.append(abs(obj_pgd - obj_opt) / abs(obj_opt))
print(f"  PGD (100 iter): gap={np.mean(pgd_gaps):.4f}")

# 训练 PGDNet
print("\nTraining PGDNet...")
set_seed(42)
model_pgd = PGDNet(n_qp, T=10, constraint_type=constraint_type)
optimizer = torch.optim.Adam(model_pgd.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# 生成训练数据
Q_train_list, c_train_list, x_opt_train_list = [], [], []
for i in range(500):
    data = generate_qp_data(n_qp, constraint_type, seed=1000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    Q_train_list.append(data['Q'])
    c_train_list.append(data['c'])
    x_opt_train_list.append(x_opt)

train_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(Q_train_list)),
    torch.FloatTensor(np.array(c_train_list)),
    torch.FloatTensor(np.array(x_opt_train_list)))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

for epoch in range(50):
    model_pgd.train()
    for q_batch, c_batch, x_batch in train_loader:
        x_pred = model_pgd(q_batch, c_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_pgd.eval()
pgd_net_gaps = []
for i in range(50):
    data = generate_qp_data(n_qp, constraint_type, seed=9000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    obj_opt = qp_objective(data['Q'], data['c'], x_opt)
    with torch.no_grad():
        q = torch.FloatTensor(data['Q']).unsqueeze(0)
        c = torch.FloatTensor(data['c']).unsqueeze(0)
        x_pred = model_pgd(q, c)
        obj_pred = qp_objective(data['Q'], data['c'], to_numpy(x_pred.squeeze()))
        pgd_net_gaps.append(abs(obj_pred - obj_opt) / abs(obj_opt))
print(f"  PGDNet (10 layers): gap={np.mean(pgd_net_gaps):.4f}")

# 训练 PGDMomentumNet
print("\nTraining PGDMomentumNet...")
set_seed(42)
model_pgd_mom = PGDMomentumNet(n_qp, T=10, constraint_type=constraint_type)
optimizer = torch.optim.Adam(model_pgd_mom.parameters(), lr=1e-3)

for epoch in range(50):
    model_pgd_mom.train()
    for q_batch, c_batch, x_batch in train_loader:
        x_pred = model_pgd_mom(q_batch, c_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model_pgd_mom.eval()
pgd_mom_gaps = []
for i in range(50):
    data = generate_qp_data(n_qp, constraint_type, seed=9000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    obj_opt = qp_objective(data['Q'], data['c'], x_opt)
    with torch.no_grad():
        q = torch.FloatTensor(data['Q']).unsqueeze(0)
        c = torch.FloatTensor(data['c']).unsqueeze(0)
        x_pred = model_pgd_mom(q, c)
        obj_pred = qp_objective(data['Q'], data['c'], to_numpy(x_pred.squeeze()))
        pgd_mom_gaps.append(abs(obj_pred - obj_opt) / abs(obj_opt))
print(f"  PGDMomentumNet (10 layers): gap={np.mean(pgd_mom_gaps):.4f}")

results['qp'] = {
    'pgd': float(np.mean(pgd_gaps)),
    'pgd_net': float(np.mean(pgd_net_gaps)),
    'pgd_momentum': float(np.mean(pgd_mom_gaps)),
}

# ============================================================
# 保存结果
# ============================================================
print("\n=== Saving Results ===")
with open('complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Results saved to complete_results.json")

# 打印总结
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\n--- LISTA-Momentum at different T ---")
for T in [5, 10, 20]:
    r = results[f'lista_momentum_T{T}']
    print(f"  T={T}: ID={r['id']:.4f}, OOD={r['ood']:.4f}, Deg={r['degradation']:.2f}x")

print("\n--- Noise Robustness ---")
for noise, err in results['noise_robustness'].items():
    print(f"  σ={noise:.3f}: error={err:.4f}")

print("\n--- Low-rank Matrix Recovery ---")
for method, err in results['low_rank'].items():
    print(f"  {method}: error={err:.4f}")

print("\n--- Quadratic Programming ---")
for method, gap in results['qp'].items():
    print(f"  {method}: gap={gap:.4f}")
