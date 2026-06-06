"""
test_dimension_generalization.py — 测试维度泛化性并分析参数矩阵 W
"""

import sys, time, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from common.utils import set_seed, to_numpy, count_parameters
from common.metrics import relative_error
from lasso.classical import ista, fista
from lasso.lista_universal import (
    create_universal_lista, analyze_W_matrix, visualize_W_matrix,
    analyze_all_layers
)

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
# 实验 1: 维度泛化性
# ============================================================
print("="*60)
print("Experiment 1: Dimension Generalization")
print("="*60)

# 训练配置
m_train, n_train = 50, 200
sparsity = 10
T = 10

# 生成训练数据
num_A = 50
A_train_list = []
rng = np.random.RandomState(42)
for i in range(num_A):
    A = rng.randn(m_train, n_train)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    A_train_list.append(A)

B_train_all, X_train_all, A_train_all = [], [], []
for i, A in enumerate(A_train_list):
    B, X = gen_data(A, 100, sparsity, seed=1000+i)
    B_train_all.append(B)
    X_train_all.append(X)
    A_train_all.append(np.tile(A, (100, 1, 1)))
B_train = np.concatenate(B_train_all)
X_train = np.concatenate(X_train_all)
A_train = np.concatenate(A_train_all)

B_val, X_val, A_val = [], [], []
for i, A in enumerate(A_train_list[:10]):
    B, X = gen_data(A, 50, sparsity, seed=5000+i)
    B_val.append(B)
    X_val.append(X)
    A_val.append(np.tile(A, (50, 1, 1)))
B_val = np.concatenate(B_val)
X_val = np.concatenate(X_val)
A_val = np.concatenate(A_val)

# 测试数据
A_test_id = A_train_list[0]
B_test_id, X_test_id = gen_data(A_test_id, 200, sparsity, seed=9999)

# ISTA baseline
ista_errors = []
for i in range(200):
    lam = 0.01 * np.max(np.abs(A_test_id.T @ B_test_id[i]))
    x_ista, _ = ista(A_test_id, B_test_id[i], lam, max_iter=T)
    ista_errors.append(relative_error(X_test_id[i], x_ista))
print(f"\nISTA baseline: {np.mean(ista_errors):.4f}")

# 训练 LISTA-Momentum
print("\nTraining LISTA-Momentum...")
set_seed(42)
model = create_universal_lista('momentum', m_train, n_train, T=T)
params = count_parameters(model)
print(f"Parameters: {params}")

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

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}: val_loss={val_loss:.8f}")

if best_state:
    model.load_state_dict(best_state)

# 测试 ID 性能
model.eval()
id_errors = []
with torch.no_grad():
    for i in range(200):
        b = torch.FloatTensor(B_test_id[i:i+1])
        A = torch.FloatTensor(A_test_id).unsqueeze(0)
        x_pred = model(b, A)
        id_errors.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))
print(f"\nLISTA-Momentum ID: {np.mean(id_errors):.4f}")

# ============================================================
# 实验 2: 维度泛化测试
# ============================================================
print("\n" + "="*60)
print("Experiment 2: Different Dimensions")
print("="*60)

# 测试不同维度
# 注意: 当前模型使用 nn.Linear(n, n)，维度固定
# 只能测试相同 n 的情况
dimension_tests_same_n = [
    (50, 200),  # 训练维度
    (100, 200),  # m 不同
    (30, 200),   # m 不同
]

dim_results = {}

for m_test, n_test in dimension_tests_same_n:
    print(f"\nTesting m={m_test}, n={n_test}...")

    # 生成测试数据
    A_test = np.random.randn(m_test, n_test)
    A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
    B_test, X_test = gen_data(A_test, 100, sparsity, seed=8000)

    # ISTA baseline
    ista_dim_errors = []
    for i in range(100):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        ista_dim_errors.append(relative_error(X_test[i], x_ista))

    # LISTA-Momentum
    lista_dim_errors = []
    try:
        with torch.no_grad():
            for i in range(100):
                b = torch.FloatTensor(B_test[i:i+1])
                A = torch.FloatTensor(A_test).unsqueeze(0)
                x_pred = model(b, A)
                lista_dim_errors.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))
    except Exception as e:
        print(f"  Error: {e}")
        lista_dim_errors = [1.0] * 100

    dim_results[f'{m_test}x{n_test}'] = {
        'ista': float(np.mean(ista_dim_errors)),
        'lista': float(np.mean(lista_dim_errors)),
        'vs_ista': float(np.mean(ista_dim_errors) / (np.mean(lista_dim_errors) + 1e-10)),
    }

    print(f"  ISTA: {np.mean(ista_dim_errors):.4f}")
    print(f"  LISTA: {np.mean(lista_dim_errors):.4f}")
    print(f"  vs ISTA: {np.mean(ista_dim_errors)/(np.mean(lista_dim_errors) + 1e-10):.2f}x")

# ============================================================
# 实验 3: 参数矩阵 W 分析
# ============================================================
print("\n" + "="*60)
print("Experiment 3: Parameter Matrix W Analysis")
print("="*60)

# 分析所有层的 W
analyses = analyze_all_layers(model)

print(f"\nNumber of layers: {len(analyses)}")

for i, analysis in enumerate(analyses):
    print(f"\nLayer {i+1}:")
    print(f"  W shape: {analysis['shape']}")
    print(f"  W mean: {analysis['mean']:.6f}")
    print(f"  W std: {analysis['std']:.6f}")
    print(f"  Diff from I: {analysis['diff_from_identity']:.6f}")
    print(f"  Cosine similarity to I: {analysis['cosine_similarity']:.6f}")
    print(f"  Spectral radius: {analysis['spectral_radius']:.6f}")
    print(f"  Condition number: {analysis['condition_number']:.6f}")
    print(f"  Effective rank: {analysis['effective_rank']:.1f}")

# 可视化 W 矩阵
print("\nVisualizing W matrices...")
for i in range(min(3, len(analyses))):
    visualize_W_matrix(analyses[i], save_path=f'W_matrix_layer_{i+1}.png')

# ============================================================
# 保存结果
# ============================================================
results = {
    'ista_baseline': float(np.mean(ista_errors)),
    'lista_id': float(np.mean(id_errors)),
    'dimension_generalization': dim_results,
    'W_analysis': [
        {
            'layer': i+1,
            'diff_from_identity': a['diff_from_identity'],
            'cosine_similarity': a['cosine_similarity'],
            'spectral_radius': a['spectral_radius'],
            'condition_number': a['condition_number'],
            'effective_rank': a['effective_rank'],
        }
        for i, a in enumerate(analyses)
    ]
}

with open('dimension_generalization_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to dimension_generalization_results.json")

# 打印总结
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nISTA baseline: {np.mean(ista_errors):.4f}")
print(f"LISTA-Momentum ID: {np.mean(id_errors):.4f}")
print(f"\nDimension Generalization:")
for dim, res in dim_results.items():
    print(f"  {dim}: ISTA={res['ista']:.4f}, LISTA={res['lista']:.4f}, vs_ISTA={res['vs_ista']:.2f}x")
