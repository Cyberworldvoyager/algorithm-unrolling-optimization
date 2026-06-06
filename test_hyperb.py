import sys, time
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.utils import set_seed, to_numpy
from common.metrics import relative_error
from lasso.classical import ista

set_seed(42)

m, n = 50, 200
sparsity = 10
T = 10
noise_std = 0.001

num_A_train = 50
A_list = []
rng = np.random.RandomState(42)
for i in range(num_A_train):
    A = rng.randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    A_list.append(A)

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

# 训练数据
B_train_all, X_train_all, A_train_all = [], [], []
for i, A in enumerate(A_list):
    B, X = gen_data(A, 200, sparsity, noise_std, seed=1000+i)
    B_train_all.append(B)
    X_train_all.append(X)
    A_train_all.append(np.tile(A, (200, 1, 1)))
B_train = np.concatenate(B_train_all)
X_train = np.concatenate(X_train_all)
A_train = np.concatenate(A_train_all)

B_val, X_val, A_val = [], [], []
for i, A in enumerate(A_list[:10]):
    B, X = gen_data(A, 50, sparsity, noise_std, seed=5000+i)
    B_val.append(B)
    X_val.append(X)
    A_val.append(np.tile(A, (50, 1, 1)))
B_val = np.concatenate(B_val)
X_val = np.concatenate(X_val)
A_val = np.concatenate(A_val)

# 测试
A_test_id = A_list[0]
B_test_id, X_test_id = gen_data(A_test_id, 200, sparsity, noise_std, seed=9999)

B_test_ood, X_test_ood, A_test_ood = [], [], []
for i in range(20):
    A = np.random.RandomState(5000+i).randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    B, X = gen_data(A, 50, sparsity, noise_std, seed=6000+i)
    B_test_ood.append(B)
    X_test_ood.append(X)
    A_test_ood.append(np.tile(A, (50, 1, 1)))
B_test_ood = np.concatenate(B_test_ood)
X_test_ood = np.concatenate(X_test_ood)
A_test_ood = np.concatenate(A_test_ood)

# ISTA baseline
ista_id, ista_ood = [], []
for i in range(200):
    lam = 0.01 * np.max(np.abs(A_test_id.T @ B_test_id[i]))
    x_ista, _ = ista(A_test_id, B_test_id[i], lam, max_iter=T)
    ista_id.append(relative_error(X_test_id[i], x_ista))
for i in range(len(B_test_ood)):
    A = A_test_ood[i]
    lam = 0.01 * np.max(np.abs(A.T @ B_test_ood[i]))
    x_ista, _ = ista(A, B_test_ood[i], lam, max_iter=T)
    ista_ood.append(relative_error(X_test_ood[i], x_ista))
print(f'ISTA ID: {np.mean(ista_id):.4f}, OOD: {np.mean(ista_ood):.4f}')
print()


# HyperB Layer
class HyperBLayer(nn.Module):
    def __init__(self, m, n, hidden_dim=128):
        super().__init__()
        self.m, self.n = m, n
        self.B_generator = nn.Sequential(
            nn.Linear(m * n, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n * m),
        )
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, b, x, A_flat):
        batch_size = b.shape[0]
        B = self.B_generator(A_flat).view(batch_size, self.n, self.m)
        A_mat = A_flat.view(batch_size, self.m, self.n)
        Ax = torch.bmm(A_mat, x.unsqueeze(-1)).squeeze(-1)
        residual = b - Ax
        B_residual = torch.bmm(B, residual.unsqueeze(-1)).squeeze(-1)
        z = x + self.eta * B_residual
        return torch.sign(z) * torch.maximum(torch.abs(z) - self.threshold, torch.zeros_like(z))


class HyperBLista(nn.Module):
    def __init__(self, m, n, T=10, hidden_dim=128):
        super().__init__()
        self.m, self.n, self.T = m, n, T
        self.layers = nn.ModuleList([HyperBLayer(m, n, hidden_dim) for _ in range(T)])

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        A_flat = A.view(batch_size, -1)
        x = x0 if x0 is not None else torch.zeros(batch_size, self.n, device=b.device)
        for layer in self.layers:
            x = layer(b, x, A_flat)
        return x


print('=== HYPERB LISTA ===')
set_seed(42)
model = HyperBLista(m, n, T=T, hidden_dim=128)
params = sum(p.numel() for p in model.parameters())
print(f'Parameters: {params}')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
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

for epoch in range(150):
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
        if patience_counter >= 25:
            break

    if (epoch + 1) % 20 == 0:
        print(f'  Epoch {epoch+1}: val_loss={val_loss:.8f}')

if best_state:
    model.load_state_dict(best_state)

model.eval()
id_errors = []
with torch.no_grad():
    for i in range(200):
        b = torch.FloatTensor(B_test_id[i:i+1])
        A = torch.FloatTensor(A_test_id).unsqueeze(0)
        x_pred = model(b, A)
        id_errors.append(relative_error(X_test_id[i], to_numpy(x_pred.squeeze())))

ood_errors = []
with torch.no_grad():
    for i in range(len(B_test_ood)):
        b = torch.FloatTensor(B_test_ood[i:i+1])
        A = torch.FloatTensor(A_test_ood[i]).unsqueeze(0)
        x_pred = model(b, A)
        ood_errors.append(relative_error(X_test_ood[i], to_numpy(x_pred.squeeze())))

print(f'\nHyperB LISTA Results:')
print(f'  ID:  {np.mean(id_errors):.4f} (ISTA: {np.mean(ista_id):.4f})')
print(f'  OOD: {np.mean(ood_errors):.4f} (ISTA: {np.mean(ista_ood):.4f})')
print(f'  Degradation: {np.mean(ood_errors)/np.mean(id_errors):.2f}x')
print(f'  vs ISTA ID:  {np.mean(ista_id)/np.mean(id_errors):.2f}x')
print(f'  vs ISTA OOD: {np.mean(ista_ood)/np.mean(ood_errors):.2f}x')
