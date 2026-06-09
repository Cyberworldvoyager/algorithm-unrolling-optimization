"""
run_all_experiments.py — 全部模型的完整对比实验 (单一入口)

模型 (全部从 lasso 包导入，无内联重复实现):
  经典基线:   ISTA, FISTA
  展开网络:   LISTA, LISTA-CP, LISTA-Momentum, 维度无关 LISTA (DA-LISTA)
  序列模型:   RNN-LISTA, LSTM-LISTA, Transformer-LISTA

实验:
  1. 全模型对比 (T=10, m=100, n=200, k=5)，多随机种子，报告 均值±标准差
  2. 不同展开层数 T ∈ {5, 10, 20}
  3. 泛化性 (ID vs OOD：训练矩阵 A 之外的新 A)
  4. 跨维度泛化 (训练 n∈{100,150,200}，测试 n∈{50,100,200,300,400})——验证 DA-LISTA
  5. W 矩阵谱分析 (数值结果写入 JSON)
  以上 1-3 在 无噪声 / 有噪声 两种条件下各跑一遍。

设备: 自动选择 CUDA (若可用)，否则 CPU。
训练: mini-batch + Adam + 梯度裁剪 (max_norm=1.0)。
问题规模选 well-posed 区 (m/n=0.5, k=5) 且展开网络按 ISTA 等价初始化 (η=1/L)，
故 LISTA/CP/Momentum 初始即达经典 ISTA 水平。
"""

import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common.metrics import relative_error
from lasso import (ista, fista, LISTA, LISTACP, LISTAMomentum,
                   DimensionAgnosticLISTA, RNNLISTA, LSTMLISTA, TransformerLISTA)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练超参数
NUM_SAMPLES = 1000      # 训练样本数
NUM_EPOCHS = 50         # 训练轮数 (早停: patience=10)
BATCH_SIZE = 64         # 批大小
LR = 1e-3               # Adam 学习率
GRAD_CLIP = 1.0         # 梯度裁剪 max_norm
N_TEST = 100            # 测试样本数
N_SEEDS = 3             # 多种子重复次数
SEEDS = [42, 123, 2024]
M_OBS = 100             # 观测维度 m (well-posed: m/n=0.5)
N_DIM = 200             # 信号维度 n
SPARSITY = 5            # 稀疏度 k (可恢复区: k << m)
DEFAULT_T = 10          # 默认展开层数


def set_seed(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def to_numpy(t):
    return t.detach().cpu().numpy()


# ============================================================
# 数据生成
# ============================================================

def make_matrix(m, n, rng):
    A = rng.randn(m, n).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    return A


def step_size(A_np):
    """ISTA 步长 η = 1/L，L = ‖AᵀA‖₂ (谱范数)。"""
    L = np.linalg.norm(A_np.T @ A_np, ord=2)
    return 1.0 / float(L)


def gen_data(A, num, sparsity=SPARSITY, noise=0.0, seed=0):
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


def gen_train_set(noise, fixed_n, seed, multi_dim=False, A_fixed=None):
    """生成训练样本 (b, x, A)。

    - multi_dim=True : 每个样本 n∈{100,150,200} 且 A 各自随机生成
                       (DA / 序列模型的设定，体现跨 A、跨维度泛化)。
    - multi_dim=False: 所有样本共用同一个固定字典 A=A_fixed
                       (LISTA / LISTA-CP / LISTA-Momentum 的标准设定，
                        Gregor&LeCun / Chen et al.：在固定字典上学习快速近似)。
                       该 A 必须与模型初始化、评估所用的 A 一致，否则
                       监督对 (b,x) 与前向用的 A 不自洽，训练无意义。
    """
    rng = np.random.RandomState(seed)
    if not multi_dim and A_fixed is None:
        raise ValueError("固定维度训练必须提供 A_fixed (与初始化/评估一致的字典)")
    samples = []
    for _ in range(NUM_SAMPLES):
        if multi_dim:
            n = rng.choice([100, 150, 200])
            A = make_matrix(M_OBS, n, rng)
        else:
            n = fixed_n
            A = A_fixed
        x = np.zeros(n)
        support = rng.choice(n, SPARSITY, replace=False)
        x[support] = rng.randn(SPARSITY)
        b = A @ x + noise * rng.randn(M_OBS)
        samples.append((b.astype(np.float32), x.astype(np.float32), A))
    return samples


# ============================================================
# 训练 / 评估
# ============================================================

def train_fixed_dim(model, samples, needs_A, val_frac=0.15, patience=15):
    """固定维度模型的 batch 训练。

    所有样本共用同一个固定字典 A (= samples[i][2]，gen_train_set 保证一致)。
    划出 val_frac 作验证集，按验证误差早停 (patience 轮无提升即停)，
    并回滚到验证集最优权重，避免欠/过拟合。
    """
    import copy
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    B_all = torch.from_numpy(np.stack([s[0] for s in samples])).to(DEVICE)
    X_all = torch.from_numpy(np.stack([s[1] for s in samples])).to(DEVICE)
    A = torch.from_numpy(samples[0][2]).to(DEVICE)   # 固定字典 (所有样本同一 A)

    n_total = B_all.shape[0]
    n_val = int(n_total * val_frac)
    g = torch.Generator(device=DEVICE); g.manual_seed(0)
    perm0 = torch.randperm(n_total, device=DEVICE, generator=g)
    val_idx, tr_idx = perm0[:n_val], perm0[n_val:]
    Btr, Xtr = B_all[tr_idx], X_all[tr_idx]
    Bval, Xval = B_all[val_idx], X_all[val_idx]
    n_tr = Btr.shape[0]

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        for i in range(0, n_tr, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            pred = model(Btr[idx], A) if needs_A else model(Btr[idx])
            loss = criterion(pred, Xtr[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
        # 验证
        model.eval()
        with torch.no_grad():
            vpred = model(Bval, A) if needs_A else model(Bval)
            vloss = float(criterion(vpred, Xval))
        if vloss < best_val - 1e-6:
            best_val, wait = vloss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_multi_dim(model, samples, val_frac=0.15, patience=15):
    """变维度模型 (DA/序列) 的训练。按维度分组成 batch。

    每个样本携带各自的 A (体现跨 A / 跨维度泛化)。同样划验证集 + 早停。
    """
    import copy
    model.to(DEVICE).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    # 按 n 分组, 每组再切训练/验证
    groups = {}
    for b, x, A in samples:
        groups.setdefault(x.shape[0], []).append((b, x, A))
    tr_tens, val_tens = {}, {}
    rng = np.random.RandomState(0)
    for n, grp in groups.items():
        idx = rng.permutation(len(grp))
        n_val = int(len(grp) * val_frac)
        v_ids, t_ids = idx[:n_val], idx[n_val:]
        def pack(ids):
            return (
                torch.from_numpy(np.stack([grp[j][0] for j in ids])).to(DEVICE),
                torch.from_numpy(np.stack([grp[j][1] for j in ids])).to(DEVICE),
                torch.from_numpy(np.stack([grp[j][2] for j in ids])).to(DEVICE),
            )
        tr_tens[n], val_tens[n] = pack(t_ids), pack(v_ids)

    best_val, best_state, wait = float('inf'), None, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        for n, (B, X, A) in tr_tens.items():
            cnt = B.shape[0]
            perm = torch.randperm(cnt, device=DEVICE)
            for i in range(0, cnt, BATCH_SIZE):
                idx = perm[i:i + BATCH_SIZE]
                pred = model(B[idx], A[idx])
                loss = criterion(pred, X[idx])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
        # 验证 (所有维度合并的平均 MSE)
        model.eval()
        with torch.no_grad():
            vlosses = [float(criterion(model(B, A), X)) for B, X, A in val_tens.values()]
            vloss = float(np.mean(vlosses))
        if vloss < best_val - 1e-6:
            best_val, wait = vloss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, A_np, B_np, X_np, needs_A):
    model.eval()
    A = torch.from_numpy(A_np).to(DEVICE)
    B = torch.from_numpy(B_np).to(DEVICE)
    pred = model(B, A) if needs_A else model(B)
    pred = to_numpy(pred)
    return float(np.mean([relative_error(X_np[i], pred[i]) for i in range(len(X_np))]))


def evaluate_classical(A_np, B_np, X_np, T):
    ista_errs, fista_errs = [], []
    for i in range(len(B_np)):
        lam = 0.01 * np.max(np.abs(A_np.T @ B_np[i]))
        x_i, _ = ista(A_np, B_np[i], lam, max_iter=T)
        x_f, _ = fista(A_np, B_np[i], lam, max_iter=T)
        ista_errs.append(relative_error(X_np[i], x_i))
        fista_errs.append(relative_error(X_np[i], x_f))
    return float(np.mean(ista_errs)), float(np.mean(fista_errs))


# ============================================================
# 模型工厂 (统一构建 + 训练接口)
# 返回: (model, needs_A, is_multi_dim)
# ============================================================

def build_and_train(name, T, noise, seed, A_train_np):
    """构建并训练一个学习型模型。

    LISTA / LISTA-CP / LISTA-Momentum 均按 ISTA 等价方式初始化 (η=1/L)，
    使其初始即达到经典 ISTA 水平，训练只在此基础上改进。
    """
    set_seed(seed)
    n = A_train_np.shape[1]
    A_tensor = torch.from_numpy(A_train_np)
    eta0 = step_size(A_train_np)
    th0 = 1e-3                                  # 初始软阈值 (≈η·λ 量级)

    if name == 'lista':
        model = LISTA(M_OBS, n, T, init_threshold=th0,
                      A_init=A_tensor, init_eta=eta0)
        needs_A, multi = False, False
    elif name == 'lista_cp':
        model = LISTACP(A_tensor, T, init_eta=eta0, init_threshold=th0)
        needs_A, multi = False, False
    elif name == 'lista_momentum':
        model = LISTAMomentum(M_OBS, n, T, init_eta=eta0, init_threshold=th0)
        needs_A, multi = True, False
    elif name == 'da_lista':
        model, needs_A, multi = DimensionAgnosticLISTA(T), True, True
    elif name == 'rnn_lista':
        model, needs_A, multi = RNNLISTA(T), True, True
    elif name == 'lstm_lista':
        model, needs_A, multi = LSTMLISTA(T), True, True
    elif name == 'transformer_lista':
        model, needs_A, multi = TransformerLISTA(T), True, True
    else:
        raise ValueError(name)

    if multi:
        # DA/序列模型: 每个样本独立 A, 体现跨 A / 跨维度泛化
        samples = gen_train_set(noise, fixed_n=n, seed=seed, multi_dim=True)
        model = train_multi_dim(model, samples)
    else:
        # 固定字典模型: 所有样本共用同一个 A (= A_train_np, 与初始化/评估一致)
        samples = gen_train_set(noise, fixed_n=n, seed=seed,
                                multi_dim=False, A_fixed=A_train_np)
        model = train_fixed_dim(model, samples, needs_A)
    return model, needs_A


LEARN_MODELS = ['lista', 'lista_cp', 'lista_momentum', 'da_lista',
                'rnn_lista', 'lstm_lista', 'transformer_lista']
ALL_MODELS = ['ista', 'fista'] + LEARN_MODELS


# ============================================================
# 实验 1-3 (在给定噪声条件下)
# ============================================================

def run_condition(noise_label, noise_std):
    print(f"\n{'='*70}\n  {noise_label} (sigma={noise_std}) on {DEVICE}\n{'='*70}")
    out = {}

    # ---- 实验 1: 全模型对比 (默认 T, n=N_DIM), 多种子 ----
    print(f"\n--- 实验 1: 全模型对比 (T={DEFAULT_T}, n={N_DIM}), 多种子 ---")
    T = DEFAULT_T
    per_seed = {name: [] for name in ALL_MODELS}
    trained_ref = {}   # 保存第一个种子的模型供泛化性实验复用
    A_ref = None

    for si, seed in enumerate(SEEDS):
        rng = np.random.RandomState(seed)
        A_test = make_matrix(M_OBS, N_DIM, rng)
        B_test, X_test = gen_data(A_test, N_TEST, SPARSITY, noise_std, seed=9999 + seed)
        if si == 0:
            A_ref, B_ref, X_ref = A_test, B_test, X_test

        ista_e, fista_e = evaluate_classical(A_test, B_test, X_test, T)
        per_seed['ista'].append(ista_e)
        per_seed['fista'].append(fista_e)

        for name in LEARN_MODELS:
            model, needs_A = build_and_train(name, T, noise_std, seed, A_test)
            err = evaluate(model, A_test, B_test, X_test, needs_A)
            per_seed[name].append(err)
            if si == 0:
                trained_ref[name] = (model, needs_A)
        print(f"  seed={seed}: " + " ".join(
            f"{k}={per_seed[k][-1]:.4f}" for k in ALL_MODELS))

    out['full_comparison'] = {
        name: {'mean': float(np.mean(per_seed[name])),
               'std': float(np.std(per_seed[name])),
               'values': [float(v) for v in per_seed[name]]}
        for name in ALL_MODELS
    }

    # ---- 实验 2: 不同 T (单种子, 快速) ----
    print("\n--- 实验 2: 不同展开层数 T ---")
    out['different_T'] = {}
    seed = SEEDS[0]
    rng = np.random.RandomState(seed)
    A_t = make_matrix(M_OBS, N_DIM, rng)
    B_t, X_t = gen_data(A_t, N_TEST, SPARSITY, noise_std, seed=9999 + seed)
    for T_val in [5, 10, 20]:
        ista_e, fista_e = evaluate_classical(A_t, B_t, X_t, T_val)
        row = {'ista': ista_e, 'fista': fista_e}
        for name in LEARN_MODELS:
            model, needs_A = build_and_train(name, T_val, noise_std, seed, A_t)
            row[name] = evaluate(model, A_t, B_t, X_t, needs_A)
        out['different_T'][str(T_val)] = row
        print(f"  T={T_val}: " + " ".join(f"{k}={row[k]:.4f}" for k in ALL_MODELS))

    # ---- 实验 3: 泛化性 (ID vs OOD), 复用实验1种子0的模型 ----
    print("\n--- 实验 3: 泛化性 (ID vs OOD) ---")
    gen = {}
    # ID = 训练所用 A_ref 的测试集
    id_err = {}
    id_err['ista'], id_err['fista'] = evaluate_classical(A_ref, B_ref, X_ref, T)
    for name in LEARN_MODELS:
        model, needs_A = trained_ref[name]
        id_err[name] = evaluate(model, A_ref, B_ref, X_ref, needs_A)

    ood_acc = {name: [] for name in ALL_MODELS}
    for trial in range(3):
        rng = np.random.RandomState(5000 + trial)
        A_ood = make_matrix(M_OBS, N_DIM, rng)
        B_ood, X_ood = gen_data(A_ood, 50, SPARSITY, noise_std, seed=6000 + trial)
        ie, fe = evaluate_classical(A_ood, B_ood, X_ood, T)
        ood_acc['ista'].append(ie)
        ood_acc['fista'].append(fe)
        for name in LEARN_MODELS:
            model, needs_A = trained_ref[name]
            ood_acc[name].append(evaluate(model, A_ood, B_ood, X_ood, needs_A))

    for name in ALL_MODELS:
        ood_mean = float(np.mean(ood_acc[name]))
        gen[name] = {'id': id_err[name], 'ood': ood_mean,
                     'degradation': ood_mean / (id_err[name] + 1e-10)}
        print(f"  {name:18s}: ID={id_err[name]:.4f} OOD={ood_mean:.4f} "
              f"degrad={gen[name]['degradation']:.2f}x")
    out['generalization'] = gen
    return out, (trained_ref, A_ref)


# ============================================================
# 实验 4: 跨维度泛化 (验证 DA-LISTA 的维度无关性)
# ============================================================

def run_cross_dimension():
    print(f"\n{'='*70}\n  实验 4: 跨维度泛化 (DA/序列模型, 训练 n∈{{100,150,200}})\n{'='*70}")
    seed = SEEDS[0]
    T = DEFAULT_T
    noise_std = 0.0
    results = {}
    test_dims = [50, 100, 200, 300, 400]

    # 多维度模型 (维度无关): 训练一次, 测试多维度
    for name in ['da_lista', 'rnn_lista', 'lstm_lista']:
        set_seed(seed)
        if name == 'da_lista':
            model = DimensionAgnosticLISTA(T)
        elif name == 'rnn_lista':
            model = RNNLISTA(T)
        else:
            model = LSTMLISTA(T)
        samples = gen_train_set(noise_std, fixed_n=200, seed=seed, multi_dim=True)
        model = train_multi_dim(model, samples)

        per_dim = {}
        for n in test_dims:
            rng = np.random.RandomState(7000 + n)
            A = make_matrix(M_OBS, n, rng)
            B, X = gen_data(A, 50, SPARSITY, noise_std, seed=8000 + n)
            per_dim[str(n)] = evaluate(model, A, B, X, needs_A=True)
        results[name] = per_dim
        print(f"  {name:14s}: " + " ".join(f"n={n}:{per_dim[str(n)]:.4f}" for n in test_dims))

    # ISTA 基线 (每个维度独立求解, 无需训练)
    ista_dim = {}
    for n in test_dims:
        rng = np.random.RandomState(7000 + n)
        A = make_matrix(M_OBS, n, rng)
        B, X = gen_data(A, 50, SPARSITY, noise_std, seed=8000 + n)
        errs = []
        for i in range(len(B)):
            lam = 0.01 * np.max(np.abs(A.T @ B[i]))
            x_i, _ = ista(A, B[i], lam, max_iter=T)
            errs.append(relative_error(X[i], x_i))
        ista_dim[str(n)] = float(np.mean(errs))
    results['ista'] = ista_dim
    print(f"  {'ista':14s}: " + " ".join(f"n={n}:{ista_dim[str(n)]:.4f}" for n in test_dims))
    results['_test_dims'] = test_dims
    results['_note'] = "训练维度 n∈{100,150,200}; n=50,300,400 为训练范围外 (OOD 维度)"
    return results


# ============================================================
# 实验 5: W 矩阵谱分析 (数值写入 JSON)
# ============================================================

def analyze_W(trained_ref):
    """对 LISTA-Momentum 各层有效变换矩阵 W=I+dW 做谱分析。"""
    model = trained_ref['lista_momentum'][0]
    rows = {}
    for layer_idx in [0, 4, 9]:
        if layer_idx >= len(model.layers):
            continue
        W = model.get_W_matrix(layer_idx).detach().cpu().numpy()
        I = np.eye(W.shape[0])
        eig = np.linalg.eigvals(W)
        S = np.linalg.svd(W, compute_uv=False)
        rows[str(layer_idx + 1)] = {
            'cosine_with_I': float(np.sum(W * I) / (np.linalg.norm(W) * np.linalg.norm(I))),
            'spectral_radius': float(np.max(np.abs(eig))),
            'condition_number': float(S[0] / (S[-1] + 1e-10)),
            'effective_rank': int(np.sum(S > 0.1 * S[0])),
            'diff_from_I': float(np.linalg.norm(W - I)),
        }
    return rows


# ============================================================
# 可视化
# ============================================================

def plot_comparison(results, save='report/training_comparison.png'):
    methods = ALL_MODELS
    labels = ['ISTA', 'FISTA', 'LISTA', 'CP', 'Mom', 'DA', 'RNN', 'LSTM', 'Trans']
    nl = [results['noiseless']['full_comparison'][m]['mean'] for m in methods]
    ny = [results['noisy']['full_comparison'][m]['mean'] for m in methods]
    nl_e = [results['noiseless']['full_comparison'][m]['std'] for m in methods]
    ny_e = [results['noisy']['full_comparison'][m]['std'] for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(methods)); w = 0.35
    ax = axes[0]
    ax.bar(x - w/2, nl, w, yerr=nl_e, label='Noiseless', color='steelblue', capsize=3)
    ax.bar(x + w/2, ny, w, yerr=ny_e, label='Noisy', color='coral', capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Relative Error'); ax.set_yscale('log')
    ax.set_title('All Models: Noiseless vs Noisy (mean ± std over seeds)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    T_vals = [5, 10, 20]
    for key, lab, style in [('lstm_lista', 'LSTM', 'b-o'), ('rnn_lista', 'RNN', 'g-^'),
                            ('da_lista', 'DA', 'm-s'), ('ista', 'ISTA', 'k--')]:
        vals = [results['noiseless']['different_T'][str(t)][key] for t in T_vals]
        ax.plot(T_vals, vals, style, label=lab, linewidth=2)
    ax.set_xlabel('Number of Layers T'); ax.set_ylabel('Relative Error')
    ax.set_yscale('log'); ax.set_title('Effect of Unrolling Depth (noiseless)')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  对比图已保存到 {save}")


def plot_cross_dim(cross, save='report/cross_dimension.png'):
    dims = cross['_test_dims']
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, style in [('lstm_lista', 'b-o'), ('rnn_lista', 'g-^'),
                        ('da_lista', 'm-s'), ('ista', 'k--')]:
        vals = [cross[name][str(n)] for n in dims]
        ax.plot(dims, vals, style, label=name, linewidth=2)
    ax.axvspan(100, 200, alpha=0.12, color='green', label='train range')
    ax.set_xlabel('Signal dimension n'); ax.set_ylabel('Relative Error')
    ax.set_yscale('log'); ax.set_title('Cross-dimension generalization (trained on n∈{100,150,200})')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  跨维度图已保存到 {save}")


def plot_W_matrix(trained_ref, save='report/W_matrix_analysis.png'):
    model = trained_ref['lista_momentum'][0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for idx, li in enumerate([0, 4, 9]):
        if li >= len(model.layers):
            break
        W = model.get_W_matrix(li).detach().cpu().numpy()
        ax = axes[0, idx]
        im = ax.imshow(W, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        ax.set_title(f'Layer {li+1}: W = I + dW'); plt.colorbar(im, ax=ax)
        ax = axes[1, idx]
        eig = np.linalg.eigvals(W)
        ax.scatter(eig.real, eig.imag, alpha=0.5, s=10)
        th = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(th), np.sin(th), 'r--', linewidth=1, label='Unit circle')
        ax.set_aspect('equal'); ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
        ax.set_title(f'Layer {li+1} Eigenvalues'); ax.legend()
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  W 矩阵图已保存到 {save}")


# ============================================================
# 主流程
# ============================================================

def main():
    print(f"Device: {DEVICE} | seeds={SEEDS} | epochs={NUM_EPOCHS} | batch={BATCH_SIZE}")
    results = {}
    ref_holder = {}
    for label, std in [("noiseless", 0.0), ("noisy", 0.01)]:
        results[label], ref = run_condition(label, std)
        ref_holder[label] = ref

    # 实验 4: 跨维度
    results['cross_dimension'] = run_cross_dimension()

    # 实验 5: W 矩阵谱分析 (用 noiseless 的 Momentum 模型)
    trained_ref = ref_holder['noiseless'][0]
    results['W_analysis'] = analyze_W(trained_ref)
    print("\n--- 实验 5: W 矩阵谱分析 ---")
    for k, v in results['W_analysis'].items():
        print(f"  Layer {k}: cos(I)={v['cosine_with_I']:.3f} "
              f"rho={v['spectral_radius']:.3f} rank={v['effective_rank']}")

    # 元信息 (便于报告引用真实配置)
    results['_meta'] = {
        'device': str(DEVICE), 'seeds': SEEDS, 'num_samples': NUM_SAMPLES,
        'num_epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
        'grad_clip': GRAD_CLIP, 'n_test': N_TEST, 'm_obs': M_OBS,
        'n_dim': N_DIM, 'default_T': DEFAULT_T,
        'sparsity': SPARSITY, 'optimizer': 'Adam',
        'init': 'ISTA-equivalent (W1=eta*A^T, W2=I-eta*A^T A, eta=1/L)',
    }

    # 可视化
    print("\n--- 生成图表 ---")
    plot_comparison(results)
    plot_cross_dim(results['cross_dimension'])
    plot_W_matrix(trained_ref)

    with open('all_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n结果已保存到 all_experiment_results.json")
    return results


if __name__ == '__main__':
    main()
