"""
test_sequence_v2.py — 测试改进的序列模型
"""

import sys, json
sys.path.insert(0, '.')
import torch
import torch.nn as nn
import numpy as np
from common.metrics import relative_error
from lasso.classical import ista, fista
from lasso.lista_sequence_v2 import create_improved_sequence_lista

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

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def train_model(variant, n=200, T=10, max_epochs=200, patience=30):
    sparsity = 10
    train_samples = []
    rng = np.random.RandomState(42)
    for i in range(20):
        A = rng.randn(50, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 10, sparsity, seed=1000+i)
        for j in range(10):
            train_samples.append((B[j:j+1], X[j:j+1], A))

    val_samples = []
    for i in range(5):
        A = rng.randn(50, n).astype(np.float32)
        A /= np.linalg.norm(A, axis=0, keepdims=True)
        B, X = gen_data(A, 5, sparsity, seed=5000+i)
        for j in range(5):
            val_samples.append((B[j:j+1], X[j:j+1], A))

    model = create_improved_sequence_lista(variant, n, T, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    print(f'  Training {variant.upper()}-LISTA (n={n}, T={T}, max_epochs={max_epochs})')

    for epoch in range(max_epochs):
        model.train()
        np.random.shuffle(train_samples)
        train_loss = 0
        count = 0
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
            count += 1
        train_loss /= count

        model.eval()
        val_loss = 0
        count = 0
        with torch.no_grad():
            for b, x, A in val_samples:
                b_t = torch.FloatTensor(b)
                x_t = torch.FloatTensor(x)
                A_t = torch.FloatTensor(A).unsqueeze(0)
                x_pred = model(b_t, A_t)
                val_loss += criterion(x_pred, x_t).item()
                count += 1
        val_loss /= count

        scheduler.step()

        if early_stopping(val_loss, model):
            print(f'  Early stopping at epoch {epoch+1}')
            break

        if (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}')

    model.load_state_dict(early_stopping.best_model)
    print(f'  Best val: {early_stopping.best_loss:.6f}')
    return model

# 测试
sparsity = 10
n = 200
T = 10

print('='*60)
print('Experiment: Improved Sequence Models')
print('='*60)

A_test = np.random.randn(50, n).astype(np.float32)
A_test /= np.linalg.norm(A_test, axis=0, keepdims=True)
B_test, X_test = gen_data(A_test, 20, sparsity, seed=9999)

ista_err, fista_err = [], []
for i in range(20):
    lam = 0.01 * np.max(np.abs(A_test.T @ B_test[i]))
    x_ista, _ = ista(A_test, B_test[i], lam, max_iter=T)
    x_fista, _ = fista(A_test, B_test[i], lam, max_iter=T)
    ista_err.append(relative_error(X_test[i], x_ista))
    fista_err.append(relative_error(X_test[i], x_fista))

print(f'\nISTA  (T={T}): {np.mean(ista_err):.4f}')
print(f'FISTA (T={T}): {np.mean(fista_err):.4f}')

results = {'ista': float(np.mean(ista_err)), 'fista': float(np.mean(fista_err))}

for variant in ['rnn', 'lstm']:
    print(f'\n--- {variant.upper()} ---')
    set_seed(42)
    model = train_model(variant, n, T, max_epochs=200, patience=30)

    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(20):
            b = torch.FloatTensor(B_test[i:i+1])
            A = torch.FloatTensor(A_test).unsqueeze(0)
            x_pred = model(b, A)
            errors.append(relative_error(X_test[i], to_numpy(x_pred.squeeze())))

    params = sum(p.numel() for p in model.parameters())
    results[variant] = {
        'error': float(np.mean(errors)),
        'params': params,
        'vs_ista': float(np.mean(ista_err) / (np.mean(errors) + 1e-10))
    }
    print(f'  Test: error={np.mean(errors):.4f}, params={params}, vs_ista={np.mean(ista_err)/(np.mean(errors)+1e-10):.2f}x')

with open('sequence_models_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('\n' + '='*60)
print('SUMMARY')
print('='*60)
print(f'{"Method":<20} {"Error":<10} {"Params":<10} {"vs ISTA":<10}')
print('-'*50)
print(f'{"ISTA":<20} {np.mean(ista_err):<10.4f} {"N/A":<10} {"1.00x":<10}')
print(f'{"FISTA":<20} {np.mean(fista_err):<10.4f} {"N/A":<10} {np.mean(ista_err)/np.mean(fista_err):<10.2f}x')
for variant in ['rnn', 'lstm']:
    r = results[variant]
    print(f'{variant.upper()+"-LISTA":<20} {r["error"]:<10.4f} {r["params"]:<10} {r["vs_ista"]:<10.2f}x')
