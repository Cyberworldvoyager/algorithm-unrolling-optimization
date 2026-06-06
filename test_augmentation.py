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

# 测试用 A
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

# 策略 1: 单 A 训练 + 噪声增强
print('=== SINGLE_A + NOISE AUGMENTATION ===')

# 训练数据：在 A 上加噪声
B_train_aug, X_train_aug = [], []
rng_aug = np.random.RandomState(42)
for i in range(2000):
    # 原始数据
    B, X = gen_data(A_test_id, 1, sparsity, noise_std, seed=10000+i)
    B_train_aug.append(B[0])
    X_train_aug.append(X[0])

    # 增强数据：对 A 加噪声
    noise_level = 0.1  # 10% 噪声
    A_noisy = A_test_id + noise_level * rng_aug.randn(m, n)
    A_noisy /= np.linalg.norm(A_noisy, axis=0, keepdims=True)
    B, X = gen_data(A_noisy, 1, sparsity, noise_std, seed=20000+i)
    B_train_aug.append(B[0])
    X_train_aug.append(X[0])

B_train_aug = np.array(B_train_aug)
X_train_aug = np.array(X_train_aug)

# 验证数据
B_val, X_val = gen_data(A_test_id, 200, sparsity, noise_std, seed=5000)

# 使用原始 A 初始化
A_tensor = torch.FloatTensor(A_test_id)
set_seed(42)
model = LISTACP(A_tensor, T=T, init_eta=0.1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.MSELoss()

train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(B_train_aug), torch.FloatTensor(X_train_aug))
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

model.eval()
id_errors = []
with torch.no_grad():
    for i in range(200):
        b = torch.FloatTensor(B_test_id[i:i+1])
        x_pred = model(b)
        id_errors.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))

ood_errors = []
with torch.no_grad():
    for i in range(len(B_test_ood)):
        b = torch.FloatTensor(B_test_ood[i:i+1])
        x_pred = model(b)
        ood_errors.append(relative_error(X_test_ood[i], to_numpy(x_pred.squeeze())))

print(f'  ID:  {np.mean(id_errors):.4f}')
print(f'  OOD: {np.mean(ood_errors):.4f}')
print(f'  Degradation: {np.mean(ood_errors)/np.mean(id_errors):.2f}x')
print()

# 策略 2: 单 A 训练 + 早停 (避免过拟合)
print('=== SINGLE_A + EARLY STOPPING ===')

# 训练数据：少量样本
B_train_small, X_train_small = gen_data(A_test_id, 200, sparsity, noise_std, seed=42)

A_tensor = torch.FloatTensor(A_test_id)
set_seed(42)
model2 = LISTACP(A_tensor, T=T, init_eta=0.1)

optimizer = torch.optim.Adam(model2.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(B_train_small), torch.FloatTensor(X_train_small))
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

best_val = float('inf')
patience_counter = 0
best_state = None

for epoch in range(50):
    model2.train()
    for b_batch, x_batch in train_loader:
        x_pred = model2(b_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer.step()

    # 用训练损失作为早停指标
    model2.eval()
    train_loss = 0
    with torch.no_grad():
        for b_batch, x_batch in train_loader:
            train_loss += criterion(model2(b_batch), x_batch).item()
    train_loss /= len(train_loader)

    if train_loss < best_val - 1e-7:
        best_val = train_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model2.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 10:
            break

if best_state:
    model2.load_state_dict(best_state)

model2.eval()
id_errors2 = []
with torch.no_grad():
    for i in range(200):
        b = torch.FloatTensor(B_test_id[i:i+1])
        x_pred = model2(b)
        id_errors2.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))

ood_errors2 = []
with torch.no_grad():
    for i in range(len(B_test_ood)):
        b = torch.FloatTensor(B_test_ood[i:i+1])
        x_pred = model2(b)
        ood_errors2.append(relative_error(X_test_ood[i], to_numpy(x_pred.squeeze())))

print(f'  ID:  {np.mean(id_errors2):.4f}')
print(f'  OOD: {np.mean(ood_errors2):.4f}')
print(f'  Degradation: {np.mean(ood_errors2)/np.mean(id_errors2):.2f}x')
print()

# 总结
print('=== SUMMARY ===')
print(f'ISTA:           ID={np.mean(ista_id):.4f}, OOD={np.mean(ista_ood):.4f}')
print(f'Single A:       ID=0.4031, OOD=1.2250, Deg=3.04x')
print(f'Multi A (50):   ID=0.9984, OOD=1.0024, Deg=1.00x')
print(f'A noise aug:    ID={np.mean(id_errors):.4f}, OOD={np.mean(ood_errors):.4f}, Deg={np.mean(ood_errors)/np.mean(id_errors):.2f}x')
print(f'Early stop:     ID={np.mean(id_errors2):.4f}, OOD={np.mean(ood_errors2):.4f}, Deg={np.mean(ood_errors2)/np.mean(id_errors2):.2f}x')
