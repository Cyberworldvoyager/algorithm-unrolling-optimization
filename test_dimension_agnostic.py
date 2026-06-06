"""
test_dimension_agnostic.py — 测试维度无关的 LISTA-Momentum
"""

import sys, time, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy, count_parameters
from common.metrics import relative_error
from lasso.classical import ista
from lasso.lista_dimension_agnostic import create_dimension_agnostic_lista

set_seed(42)

# ============================================================
# 工具函数
# ============================================================
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
# 实验 1: 训练维度无关 LISTA
# ============================================================
print("="*60)
print("Experiment 1: Train Dimension-Agnostic LISTA")
print("="*60)

# 训练配置
sparsity = 10
T = 10

# 生成训练数据 (多个不同维度)
dimension_configs = [
    (50, 100),
    (50, 150),
    (50, 200),
    (50, 250),
    (50, 300),
]

B_train_all, X_train_all, A_train_all = [], [], []
rng = np.random.RandomState(42)

for m, n in dimension_configs:
    for i in range(20):  # 每个维度 20 个 A
        A = rng.randn(m, n)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 50, sparsity, seed=1000+len(B_train_all))
        # 存储单个样本 (不合并)
        for j in range(50):
            B_train_all.append(B[j:j+1])
            X_train_all.append(X[j:j+1])
            A_train_all.append(A)

# 验证数据
B_val_all, X_val_all, A_val_all = [], [], []
for m, n in dimension_configs[:2]:
    for i in range(5):
        A = rng.randn(m, n)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 20, sparsity, seed=5000+len(B_val_all))
        for j in range(20):
            B_val_all.append(B[j:j+1])
            X_val_all.append(X[j:j+1])
            A_val_all.append(A)

print(f"Training data: {len(B_train_all)} samples, dimensions: {dimension_configs}")

# 创建模型
model = create_dimension_agnostic_lista('mlp', T=T, hidden_dim=32)
params = count_parameters(model)
print(f"Parameters: {params}")

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.MSELoss()

# 自定义 DataLoader (处理不同维度)
class DynamicBatchDataset(torch.utils.data.Dataset):
    def __init__(self, B_list, X_list, A_list):
        self.B_list = B_list
        self.X_list = X_list
        self.A_list = A_list

    def __len__(self):
        return len(self.B_list)

    def __getitem__(self, idx):
        return self.B_list[idx], self.X_list[idx], self.A_list[idx]

# 转换为 float32
B_train_all = [b.astype(np.float32) for b in B_train_all]
X_train_all = [x.astype(np.float32) for x in X_train_all]
A_train_all = [a.astype(np.float32) for a in A_train_all]

B_val_all = [b.astype(np.float32) for b in B_val_all]
X_val_all = [x.astype(np.float32) for x in X_val_all]
A_val_all = [a.astype(np.float32) for a in A_val_all]

train_dataset = DynamicBatchDataset(B_train_all, X_train_all, A_train_all)
val_dataset = DynamicBatchDataset(B_val_all, X_val_all, A_val_all)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)

print("\nTraining...")
best_val = float('inf')
patience_counter = 0
best_state = None

for epoch in range(100):
    model.train()
    train_loss = 0
    num_batches = 0

    for b_batch, x_batch, a_batch in train_loader:
        b_batch = b_batch.squeeze(0)
        x_batch = x_batch.squeeze(0)
        a_batch = a_batch.squeeze(0)

        x_pred = model(b_batch, a_batch)
        loss = criterion(x_pred, x_batch)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    scheduler.step()

    # 验证
    model.eval()
    val_loss = 0
    num_val = 0
    with torch.no_grad():
        for b_batch, x_batch, a_batch in val_loader:
            b_batch = b_batch.squeeze(0)
            x_batch = x_batch.squeeze(0)
            a_batch = a_batch.squeeze(0)

            x_pred = model(b_batch, a_batch)
            val_loss += criterion(x_pred, x_batch).item()
            num_val += 1

    val_loss /= num_val

    if val_loss < best_val - 1e-8:
        best_val = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= 20:
            break

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}")

if best_state:
    model.load_state_dict(best_state)

# ============================================================
# 实验 2: 测试不同维度
# ============================================================
print("\n" + "="*60)
print("Experiment 2: Test Different Dimensions")
print("="*60)

test_dimensions = [
    (50, 50),    # 小维度
    (50, 100),   # 训练维度
    (50, 150),   # 训练维度
    (50, 200),   # 训练维度
    (50, 300),   # 训练维度
    (50, 500),   # 大维度 (未见过)
    (50, 1000),  # 更大维度 (未见过)
    (100, 200),  # 不同 m
    (30, 200),   # 不同 m
]

dim_results = {}

for m_test, n_test in test_dimensions:
    print(f"\nTesting m={m_test}, n={n_test}...")

    # 生成测试数据
    A_test = np.random.randn(m_test, n_test)
    A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
    B_test, X_test = gen_data(A_test, 50, sparsity, seed=8000)

    # ISTA baseline
    ista_errors = []
    for i in range(50):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        ista_errors.append(relative_error(X_test[i], x_ista))

    # LISTA-Momentum
    lista_errors = []
    try:
        with torch.no_grad():
            for i in range(50):
                b = torch.FloatTensor(B_test[i:i+1])
                A = torch.FloatTensor(A_test).unsqueeze(0)
                x_pred = model(b, A)
                lista_errors.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))
    except Exception as e:
        print(f"  Error: {e}")
        lista_errors = [1.0] * 50

    dim_results[f'{m_test}x{n_test}'] = {
        'ista': float(np.mean(ista_errors)),
        'lista': float(np.mean(lista_errors)),
        'vs_ista': float(np.mean(ista_errors) / (np.mean(lista_errors) + 1e-10)),
    }

    print(f"  ISTA: {np.mean(ista_errors):.4f}")
    print(f"  LISTA: {np.mean(lista_errors):.4f}")
    print(f"  vs ISTA: {np.mean(ista_errors)/(np.mean(lista_errors) + 1e-10):.2f}x")

# ============================================================
# 保存结果
# ============================================================
results = {
    'model_params': params,
    'dimension_generalization': dim_results,
}

with open('dimension_agnostic_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to dimension_agnostic_results.json")

# 打印总结
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nModel parameters: {params}")
print(f"\nDimension Generalization:")
for dim, res in dim_results.items():
    status = "✓" if res['vs_ista'] > 1.0 else "✗"
    print(f"  {dim}: ISTA={res['ista']:.4f}, LISTA={res['lista']:.4f}, vs_ISTA={res['vs_ista']:.2f}x {status}")
