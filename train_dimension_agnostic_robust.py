"""
train_dimension_agnostic_robust.py — 鲁棒训练维度无关 LISTA

改进:
1. 更多训练数据
2. 更好的架构 (更宽的 MLP)
3. 学习率预热
4. 梯度裁剪
5. 更多训练轮次
6. 早停机制
"""

import sys, json, time
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
        B.append(b.astype(np.float32))
        X.append(x.astype(np.float32))
    return np.array(B), np.array(X)

# ============================================================
# 改进的维度无关架构
# ============================================================
class ImprovedDimensionAgnosticLayer(nn.Module):
    """改进的维度无关层。

    改进:
    1. 更宽的 MLP
    2. 残差连接
    3. 更好的初始化
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        # 梯度变换网络 (更宽)
        self.transform = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),  # 使用 GELU 替代 ReLU
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # 可学习的步长和动量
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.0))

        # 可学习的阈值
        self.threshold = nn.Parameter(torch.tensor(0.1))

        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.tensor(1.0))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.transform.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, b, x, x_prev, A):
        # 动量外推
        beta = torch.sigmoid(self.beta)
        y = x + beta * (x - x_prev)

        # 用当前 A 计算梯度
        Ay = torch.bmm(A, y.unsqueeze(-1)).squeeze(-1)
        residual = Ay - b
        grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)

        # 维度无关的梯度变换
        batch_size, n = grad.shape

        # 1. 归一化
        grad_mean = grad.mean(dim=-1, keepdim=True)
        grad_std = grad.std(dim=-1, keepdim=True) + 1e-6
        grad_normalized = (grad - grad_mean) / grad_std

        # 2. 元素独立变换
        grad_flat = grad_normalized.reshape(-1, 1)
        transformed_flat = self.transform(grad_flat)
        transformed_normalized = transformed_flat.reshape(batch_size, n)

        # 3. 恢复尺度 + 残差连接
        transformed_grad = (transformed_normalized + grad_normalized) * grad_std * self.scale

        # 更新
        z = y - self.eta * transformed_grad

        # 软阈值化
        return torch.sign(z) * torch.maximum(
            torch.abs(z) - self.threshold, torch.zeros_like(z))


class ImprovedDimensionAgnosticLISTA(nn.Module):
    """改进的维度无关 LISTA。"""

    def __init__(self, T: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([
            ImprovedDimensionAgnosticLayer(hidden_dim) for _ in range(T)
        ])

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

# ============================================================
# 训练配置
# ============================================================
sparsity = 10
T = 10
hidden_dim = 64
lr = 5e-4
num_epochs = 200
patience = 30

# 生成训练数据 (更多维度和样本)
dimension_configs = [
    (50, 50),
    (50, 100),
    (50, 150),
    (50, 200),
    (50, 250),
    (50, 300),
]

train_samples = []
rng = np.random.RandomState(42)

print("Generating training data...")
for m, n in dimension_configs:
    for i in range(20):  # 每个维度 20 个 A
        A = rng.randn(m, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 30, sparsity, seed=1000+len(train_samples))
        for j in range(30):
            train_samples.append((B[j:j+1], X[j:j+1], A))

# 验证数据
val_samples = []
for m, n in dimension_configs[:3]:
    for i in range(10):
        A = rng.randn(m, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 20, sparsity, seed=5000+len(val_samples))
        for j in range(20):
            val_samples.append((B[j:j+1], X[j:j+1], A))

print(f"Training samples: {len(train_samples)}")
print(f"Validation samples: {len(val_samples)}")

# ============================================================
# 创建模型
# ============================================================
model = ImprovedDimensionAgnosticLISTA(T=T, hidden_dim=hidden_dim)
params = count_parameters(model)
print(f"Model parameters: {params}")

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
criterion = nn.MSELoss()

# ============================================================
# 训练
# ============================================================
print("\nTraining...")
best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(num_epochs):
    # 训练
    model.train()
    np.random.shuffle(train_samples)
    train_loss = 0
    num_batches = 0

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

        train_loss += loss.item()
        num_batches += 1

    train_loss /= num_batches
    scheduler.step()

    # 验证
    model.eval()
    val_loss = 0
    num_val = 0
    with torch.no_grad():
        for b, x, A in val_samples:
            b_t = torch.FloatTensor(b)
            x_t = torch.FloatTensor(x)
            A_t = torch.FloatTensor(A).unsqueeze(0)

            x_pred = model(b_t, A_t)
            val_loss += criterion(x_pred, x_t).item()
            num_val += 1

    val_loss /= num_val

    # 早停
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}: train_loss={train_loss:.8f}, val_loss={val_loss:.8f}")

if best_state:
    model.load_state_dict(best_state)

# ============================================================
# 测试
# ============================================================
print("\n" + "="*60)
print("Testing different dimensions:")
print("="*60)

test_dims = [(50, 50), (50, 100), (50, 150), (50, 200), (50, 300), (50, 500)]

results = {}
for m, n in test_dims:
    A_test = np.random.randn(m, n).astype(np.float32)
    A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
    B_test, X_test = gen_data(A_test, 50, sparsity, seed=9000)

    # ISTA
    ista_err = []
    for i in range(50):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        ista_err.append(relative_error(X_test[i], x_ista))

    # LISTA
    lista_err = []
    model.eval()
    with torch.no_grad():
        for i in range(50):
            b = torch.FloatTensor(B_test[i:i+1])
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            lista_err.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))

    vs_ista = np.mean(ista_err) / (np.mean(lista_err) + 1e-10)
    results[f'{m}x{n}'] = {
        'ista': float(np.mean(ista_err)),
        'lista': float(np.mean(lista_err)),
        'vs_ista': float(vs_ista),
    }

    status = "OK" if vs_ista > 1.0 else "FAIL"
    print(f"  {m}x{n}: ISTA={np.mean(ista_err):.4f}, LISTA={np.mean(lista_err):.4f}, vs_ISTA={vs_ista:.2f}x [{status}]")

# 保存结果
with open('dimension_agnostic_robust_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to dimension_agnostic_robust_results.json")
