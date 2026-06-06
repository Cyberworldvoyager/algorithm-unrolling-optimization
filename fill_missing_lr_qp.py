"""
fill_missing_lr_qp.py — 补全低秩和QP实验数据
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy, count_parameters
from common.metrics import relative_error

set_seed(42)

# ============================================================
# 低秩矩阵恢复
# ============================================================
print("=== Low-rank Matrix Recovery ===")

from low_rank.problem import generate_matrix_completion_data
from low_rank.classical import admm_matrix_completion
from low_rank.admm_net import ADMMNet, ADMMNetV2

m_lr, n_lr = 50, 50
rank = 5
ratio = 0.5

# ADMM baseline
print("\nADMM baseline...")
admm_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    X_admm, _ = admm_matrix_completion(data['M_observed'], data['mask'], rho=1.0, max_iter=200)
    admm_errors.append(relative_error(data['M'], X_admm))
print(f"  ADMM (200 iter): error={np.mean(admm_errors):.4f} ± {np.std(admm_errors):.4f}")

# 生成训练数据
print("\nGenerating training data...")
M_train, M_obs_train, mask_train = [], [], []
for i in range(500):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=1000+i)
    M_train.append(data['M'])
    M_obs_train.append(data['M_observed'])
    mask_train.append(data['mask'])

M_val, M_obs_val, mask_val = [], [], []
for i in range(100):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=5000+i)
    M_val.append(data['M'])
    M_obs_val.append(data['M_observed'])
    mask_val.append(data['mask'])

train_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(M_obs_train)),
    torch.FloatTensor(np.array(mask_train)),
    torch.FloatTensor(np.array(M_train)))
val_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(M_obs_val)),
    torch.FloatTensor(np.array(mask_val)),
    torch.FloatTensor(np.array(M_val)))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

# 训练 ADMMNet
print("\nTraining ADMMNet...")
set_seed(42)
model_admm = ADMMNet(m_lr, n_lr, T=10)
params_admm = count_parameters(model_admm)
optimizer = torch.optim.Adam(model_admm.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

best_val = float('inf')
best_state = None
for epoch in range(50):
    model_admm.train()
    for b_batch, mask_batch, m_batch in train_loader:
        try:
            m_pred = model_admm(b_batch, mask_batch)
            loss = criterion(m_pred, m_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_admm.parameters(), 1.0)
            optimizer.step()
        except:
            continue
    scheduler.step()

    model_admm.eval()
    val_loss = 0
    with torch.no_grad():
        for b_batch, mask_batch, m_batch in val_loader:
            try:
                val_loss += criterion(model_admm(b_batch, mask_batch), m_batch).item()
            except:
                val_loss += 1.0
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model_admm.state_dict().items()}

if best_state:
    model_admm.load_state_dict(best_state)

model_admm.eval()
admm_net_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    try:
        with torch.no_grad():
            m_obs = torch.FloatTensor(data['M_observed']).unsqueeze(0)
            mask = torch.FloatTensor(data['mask']).unsqueeze(0)
            m_pred = model_admm(m_obs, mask)
            admm_net_errors.append(relative_error(data['M'], to_numpy(m_pred.squeeze())))
    except:
        admm_net_errors.append(1.0)
print(f"  ADMMNet (10 layers): error={np.mean(admm_net_errors):.4f} ± {np.std(admm_net_errors):.4f}")

# 训练 ADMMNetV2
print("\nTraining ADMMNetV2...")
set_seed(42)
model_admm_v2 = ADMMNetV2(m_lr, n_lr, T=10)
params_admm_v2 = count_parameters(model_admm_v2)
optimizer = torch.optim.Adam(model_admm_v2.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_val = float('inf')
best_state = None
for epoch in range(50):
    model_admm_v2.train()
    for b_batch, mask_batch, m_batch in train_loader:
        try:
            m_pred = model_admm_v2(b_batch, mask_batch)
            loss = criterion(m_pred, m_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model_admm_v2.parameters(), 1.0)
            optimizer.step()
        except:
            continue
    scheduler.step()

    model_admm_v2.eval()
    val_loss = 0
    with torch.no_grad():
        for b_batch, mask_batch, m_batch in val_loader:
            try:
                val_loss += criterion(model_admm_v2(b_batch, mask_batch), m_batch).item()
            except:
                val_loss += 1.0
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model_admm_v2.state_dict().items()}

if best_state:
    model_admm_v2.load_state_dict(best_state)

model_admm_v2.eval()
admm_v2_errors = []
for i in range(50):
    data = generate_matrix_completion_data(m_lr, n_lr, rank, ratio, seed=9000+i)
    try:
        with torch.no_grad():
            m_obs = torch.FloatTensor(data['M_observed']).unsqueeze(0)
            mask = torch.FloatTensor(data['mask']).unsqueeze(0)
            m_pred = model_admm_v2(m_obs, mask)
            admm_v2_errors.append(relative_error(data['M'], to_numpy(m_pred.squeeze())))
    except:
        admm_v2_errors.append(1.0)
print(f"  ADMMNetV2 (10 layers): error={np.mean(admm_v2_errors):.4f} ± {np.std(admm_v2_errors):.4f}")

# ============================================================
# 二次规划
# ============================================================
print("\n=== Quadratic Programming ===")

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
    pgd_gaps.append(abs(obj_pgd - obj_opt) / (abs(obj_opt) + 1e-10))
print(f"  PGD (100 iter): gap={np.mean(pgd_gaps):.6f} ± {np.std(pgd_gaps):.6f}")

# 生成训练数据
print("\nGenerating QP training data...")
Q_train, c_train, x_opt_train = [], [], []
for i in range(500):
    data = generate_qp_data(n_qp, constraint_type, seed=1000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    Q_train.append(data['Q'])
    c_train.append(data['c'])
    x_opt_train.append(x_opt)

Q_val, c_val, x_opt_val = [], [], []
for i in range(100):
    data = generate_qp_data(n_qp, constraint_type, seed=5000+i)
    x_opt = compute_optimal_solution(data['Q'], data['c'], constraint_type, data['constraint_params'])
    Q_val.append(data['Q'])
    c_val.append(data['c'])
    x_opt_val.append(x_opt)

train_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(Q_train)),
    torch.FloatTensor(np.array(c_train)),
    torch.FloatTensor(np.array(x_opt_train)))
val_ds = torch.utils.data.TensorDataset(
    torch.FloatTensor(np.array(Q_val)),
    torch.FloatTensor(np.array(c_val)),
    torch.FloatTensor(np.array(x_opt_val)))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64)

# 训练 PGDNet
print("\nTraining PGDNet...")
set_seed(42)
model_pgd = PGDNet(n_qp, T=10, constraint_type=constraint_type)
params_pgd = count_parameters(model_pgd)
optimizer = torch.optim.Adam(model_pgd.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
criterion = nn.MSELoss()

best_val = float('inf')
best_state = None
for epoch in range(50):
    model_pgd.train()
    for q_batch, c_batch, x_batch in train_loader:
        x_pred = model_pgd(q_batch, c_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_pgd.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model_pgd.eval()
    val_loss = 0
    with torch.no_grad():
        for q_batch, c_batch, x_batch in val_loader:
            val_loss += criterion(model_pgd(q_batch, c_batch), x_batch).item()
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model_pgd.state_dict().items()}

if best_state:
    model_pgd.load_state_dict(best_state)

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
        pgd_net_gaps.append(abs(obj_pred - obj_opt) / (abs(obj_opt) + 1e-10))
print(f"  PGDNet (10 layers): gap={np.mean(pgd_net_gaps):.6f} ± {np.std(pgd_net_gaps):.6f}")

# 训练 PGDMomentumNet
print("\nTraining PGDMomentumNet...")
set_seed(42)
model_pgd_mom = PGDMomentumNet(n_qp, T=10, constraint_type=constraint_type)
params_pgd_mom = count_parameters(model_pgd_mom)
optimizer = torch.optim.Adam(model_pgd_mom.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

best_val = float('inf')
best_state = None
for epoch in range(50):
    model_pgd_mom.train()
    for q_batch, c_batch, x_batch in train_loader:
        x_pred = model_pgd_mom(q_batch, c_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model_pgd_mom.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    model_pgd_mom.eval()
    val_loss = 0
    with torch.no_grad():
        for q_batch, c_batch, x_batch in val_loader:
            val_loss += criterion(model_pgd_mom(q_batch, c_batch), x_batch).item()
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        best_state = {k: v.clone() for k, v in model_pgd_mom.state_dict().items()}

if best_state:
    model_pgd_mom.load_state_dict(best_state)

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
        pgd_mom_gaps.append(abs(obj_pred - obj_opt) / (abs(obj_opt) + 1e-10))
print(f"  PGDMomentumNet (10 layers): gap={np.mean(pgd_mom_gaps):.6f} ± {np.std(pgd_mom_gaps):.6f}")

# ============================================================
# 保存结果
# ============================================================
results = {
    'low_rank': {
        'admm': {'error': float(np.mean(admm_errors)), 'std': float(np.std(admm_errors))},
        'admm_net': {'error': float(np.mean(admm_net_errors)), 'std': float(np.std(admm_net_errors)), 'params': params_admm},
        'admm_net_v2': {'error': float(np.mean(admm_v2_errors)), 'std': float(np.std(admm_v2_errors)), 'params': params_admm_v2},
    },
    'qp': {
        'pgd': {'gap': float(np.mean(pgd_gaps)), 'std': float(np.std(pgd_gaps))},
        'pgd_net': {'gap': float(np.mean(pgd_net_gaps)), 'std': float(np.std(pgd_net_gaps)), 'params': params_pgd},
        'pgd_momentum': {'gap': float(np.mean(pgd_mom_gaps)), 'std': float(np.std(pgd_mom_gaps)), 'params': params_pgd_mom},
    }
}

with open('lr_qp_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to lr_qp_results.json")

# 打印总结
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("\n--- Low-rank Matrix Recovery ---")
print(f"  ADMM (200 iter): error={np.mean(admm_errors):.4f}")
print(f"  ADMMNet (10 layers): error={np.mean(admm_net_errors):.4f}, params={params_admm}")
print(f"  ADMMNetV2 (10 layers): error={np.mean(admm_v2_errors):.4f}, params={params_admm_v2}")
print("\n--- Quadratic Programming ---")
print(f"  PGD (100 iter): gap={np.mean(pgd_gaps):.6f}")
print(f"  PGDNet (10 layers): gap={np.mean(pgd_net_gaps):.6f}, params={params_pgd}")
print(f"  PGDMomentumNet (10 layers): gap={np.mean(pgd_mom_gaps):.6f}, params={params_pgd_mom}")
