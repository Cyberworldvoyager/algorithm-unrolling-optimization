"""
test_lstm_lista.py — 测试维度无关 LSTM-LISTA
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.metrics import relative_error
from lasso.classical import ista

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

# 简化的 LSTM-LISTA (不使用元素独立处理)
class LSTMLISTALayer(nn.Module):
    def __init__(self, n, hidden_dim=32):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim

        # 输入: [x, grad] (2n 维)
        self.lstm = nn.LSTMCell(input_size=2 * n, hidden_size=hidden_dim)
        self.hidden_to_update = nn.Linear(hidden_dim, n)
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, grad, h, c):
        lstm_input = torch.cat([x, grad], dim=-1)
        h_new, c_new = self.lstm(lstm_input, (h, c))
        update = self.hidden_to_update(h_new)
        x_new = x + self.eta * update
        return x_new, h_new, c_new


class LSTMLISTA(nn.Module):
    def __init__(self, n, T=10, hidden_dim=32):
        super().__init__()
        self.n = n
        self.T = T
        self.hidden_dim = hidden_dim
        self.layer = LSTMLISTALayer(n, hidden_dim)

    def forward(self, b, A, x0=None):
        batch_size = b.shape[0]
        if A.dim() == 2:
            A = A.unsqueeze(0).expand(batch_size, -1, -1)
        n = A.shape[2]
        x = x0 if x0 is not None else torch.zeros(batch_size, n, device=b.device)
        h = torch.zeros(batch_size, self.hidden_dim, device=b.device)
        c = torch.zeros(batch_size, self.hidden_dim, device=b.device)

        for _ in range(self.T):
            Ax = torch.bmm(A, x.unsqueeze(-1)).squeeze(-1)
            residual = Ax - b
            grad = torch.bmm(A.transpose(1, 2), residual.unsqueeze(-1)).squeeze(-1)
            x, h, c = self.layer(x, grad, h, c)

        return x

# 训练
def train_model(n=200, T=10, epochs=30):
    sparsity = 10
    train_samples = []
    rng = np.random.RandomState(42)
    for i in range(15):
        A = rng.randn(50, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 10, sparsity, seed=1000+i)
        for j in range(10):
            train_samples.append((B[j:j+1], X[j:j+1], A))

    model = LSTMLISTA(n, T, hidden_dim=32)
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

        if (epoch + 1) % 10 == 0:
            print(f'  Epoch {epoch+1}')

    return model

# 测试
sparsity = 10
T = 10

print('Training LSTM-LISTA (n=200)...')
set_seed(42)
model = train_model(n=200, T=T, epochs=30)

# 测试
A_test = np.random.randn(50, 200).astype(np.float32)
A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
B_test, X_test = gen_data(A_test, 20, sparsity, seed=9999)

ista_err, lista_err = [], []
model.eval()
with torch.no_grad():
    for i in range(20):
        lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
        x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
        ista_err.append(relative_error(X_test[i], x_ista))

        b = torch.FloatTensor(B_test[i:i+1])
        A = torch.FloatTensor(A_test).unsqueeze(0)
        x_pred = model(b, A)
        lista_err.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))

print(f'\nResults:')
print(f'  ISTA: {np.mean(ista_err):.4f}')
print(f'  LSTM-LISTA: {np.mean(lista_err):.4f}')
print(f'  vs ISTA: {np.mean(ista_err)/(np.mean(lista_err)+1e-10):.2f}x')
