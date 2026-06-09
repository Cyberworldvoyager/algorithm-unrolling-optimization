"""
run_socp_ablation.py — Stage 1 (Learned-PDHG-CP) 消融实验

在 `medium` suite 上做四类消融:
  (A) K 扫描:     K ∈ {5, 10, 20, 30, 50}
  (B) 学什么:     {全冻, 只学步长, 只学动量, 全学} 四档
  (C) 初始化敏感性:
      η_init ∈ {0.45, 0.9} / ||A||  与
      β_init ∈ {halpern, constant=0.5, constant=0.9} 的笛卡尔积 (5 组)
  (D) 训练样本量: N_train ∈ {32, 128, 512}

每个配置: 训练 1 次 (seed=0)，测试集 50 个实例，统计 rel_x_err 中位数/均值。
输出:
  report/socp_ablation_results.json
  report/socp_ablation_K_sweep.png
  report/socp_ablation_what_to_learn.png
  report/socp_ablation_init.png
  report/socp_ablation_train_size.png
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from socp.problem import generate_socp_data, kkt_residuals
from socp.classical import pdhg_socp, pdqp_socp, estimate_op_norm
from socp.pdhg_cp import LearnedPDHGCP, supervised_mse


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')
os.makedirs(OUT_DIR, exist_ok=True)

SUITE_CFG = {'block_sizes': [5]*6 + [1]*4, 'm': 20}  # medium
N_TRAIN_DEFAULT = 256
N_TEST = 50
NUM_EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-3
DEVICE = torch.device('cpu')


# ============================================================
# Utilities (shared with run_socp_learned.py)
# ============================================================

def build_dataset(cfg, num, seed_offset):
    block = cfg['block_sizes']; m = cfg['m']
    As, bs, cs, xs, ops = [], [], [], [], []
    for i in range(num):
        d = generate_socp_data(block_sizes=block, m=m, seed=seed_offset + i)
        As.append(d['A']); bs.append(d['b']); cs.append(d['c']); xs.append(d['x_star'])
        ops.append(estimate_op_norm(d['A'], n_iter=30))
    return {
        'A': torch.tensor(np.stack(As).astype(np.float32)),
        'b': torch.tensor(np.stack(bs).astype(np.float32)),
        'c': torch.tensor(np.stack(cs).astype(np.float32)),
        'x_star': torch.tensor(np.stack(xs).astype(np.float32)),
        'block_sizes': block, 'm': m, 'N': sum(block),
        'op_norm_mean': float(np.mean(ops)),
    }


def freeze_groups(model: LearnedPDHGCP, learn: dict):
    """learn = {'eta':bool,'tau':bool,'beta':bool,'theta':bool}"""
    model.eta_raw.requires_grad_(learn.get('eta', True))
    model.tau_raw.requires_grad_(learn.get('tau', True))
    model.beta_raw.requires_grad_(learn.get('beta', True))
    model.theta_raw.requires_grad_(learn.get('theta', True))


def train_model(train_ds, K, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                init_eta=None, init_tau=None, init_beta_schedule='halpern',
                init_beta=0.5, learn=None, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    L = train_ds['op_norm_mean']
    ie = init_eta if init_eta is not None else 0.9 / L
    it = init_tau if init_tau is not None else 0.9 / L
    model = LearnedPDHGCP(
        K=K, init_eta=ie, init_tau=it,
        init_beta_schedule=init_beta_schedule, init_beta=init_beta,
        init_theta=0.999, block_sizes=train_ds['block_sizes'],
    ).to(DEVICE)
    if learn is not None:
        freeze_groups(model, learn)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) == 0:
        # all frozen: no training
        return model, [supervised_mse_dataset(model, train_ds)]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    n = train_ds['A'].shape[0]
    hist = []
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            A_b = train_ds['A'][idx]; b_b = train_ds['b'][idx]
            c_b = train_ds['c'][idx]; x_b = train_ds['x_star'][idx]
            x_hat, _ = model(A_b, b_b, c_b)
            loss = supervised_mse(x_hat, x_b)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item() * idx.numel()
        hist.append(total / n)
    return model, hist


def supervised_mse_dataset(model, ds):
    with torch.no_grad():
        x_hat, _ = model(ds['A'], ds['b'], ds['c'])
        return float(((x_hat - ds['x_star'])**2).sum(-1).mean())


def eval_test(model, test_ds, method='learned'):
    block = test_ds['block_sizes']
    rel_errs, primal = [], []
    n = test_ds['A'].shape[0]
    if method == 'learned':
        with torch.no_grad():
            x_hat, y_hat = model(test_ds['A'], test_ds['b'], test_ds['c'])
        for i in range(n):
            xs = test_ds['x_star'][i].numpy().astype(np.float64)
            xh = x_hat[i].numpy().astype(np.float64)
            yh = y_hat[i].numpy().astype(np.float64)
            data = {'A': test_ds['A'][i].numpy().astype(np.float64),
                    'b': test_ds['b'][i].numpy().astype(np.float64),
                    'c': test_ds['c'][i].numpy().astype(np.float64),
                    'block_sizes': block, 'N': test_ds['N'], 'm': test_ds['m'],
                    'x_star': xs}
            res = kkt_residuals(data, xh, yh)
            rel_errs.append(res['rel_x_err']); primal.append(res['primal_feas'])
    else:
        raise ValueError(method)
    return {
        'rel_x_err_mean': float(np.mean(rel_errs)),
        'rel_x_err_median': float(np.median(rel_errs)),
        'primal_feas_median': float(np.median(primal)),
    }


def eval_classical(test_ds, method, K):
    block = test_ds['block_sizes']
    rel_errs, primal = [], []
    n = test_ds['A'].shape[0]
    for i in range(n):
        data = {'A': test_ds['A'][i].numpy().astype(np.float64),
                'b': test_ds['b'][i].numpy().astype(np.float64),
                'c': test_ds['c'][i].numpy().astype(np.float64),
                'block_sizes': block, 'N': test_ds['N'], 'm': test_ds['m'],
                'x_star': test_ds['x_star'][i].numpy().astype(np.float64)}
        if method == 'pdhg':
            x, y, _ = pdhg_socp(data, K_iters=K, return_history=False)
        elif method == 'pdqp':
            x, y, _ = pdqp_socp(data, K_iters=K, return_history=False)
        else:
            raise ValueError(method)
        res = kkt_residuals(data, x, y)
        rel_errs.append(res['rel_x_err']); primal.append(res['primal_feas'])
    return {
        'rel_x_err_mean': float(np.mean(rel_errs)),
        'rel_x_err_median': float(np.median(rel_errs)),
        'primal_feas_median': float(np.median(primal)),
    }


# ============================================================
# (A) K sweep
# ============================================================

def ablation_K_sweep(train_ds, test_ds, K_list):
    print("\n=== (A) K sweep ===")
    out = {}
    for K in K_list:
        t0 = time.time()
        model, hist = train_model(train_ds, K)
        train_time = time.time() - t0
        ev_l = eval_test(model, test_ds)
        ev_p = eval_classical(test_ds, 'pdhg', K)
        ev_q = eval_classical(test_ds, 'pdqp', K)
        out[K] = {'learned': ev_l, 'pdhg': ev_p, 'pdqp': ev_q,
                  'train_time': train_time}
        print(f"  K={K:3d}  PDHG={ev_p['rel_x_err_median']:.2e}  "
              f"PDQP={ev_q['rel_x_err_median']:.2e}  "
              f"Learned={ev_l['rel_x_err_median']:.2e}  "
              f"[{train_time:.1f}s]")
    return out


# ============================================================
# (B) What to learn
# ============================================================

LEARN_VARIANTS = {
    'frozen (all)':       {'eta': False, 'tau': False, 'beta': False, 'theta': False},
    'steps only (η,τ)':   {'eta': True,  'tau': True,  'beta': False, 'theta': False},
    'momentum only (β,θ)':{'eta': False, 'tau': False, 'beta': True,  'theta': True},
    'all (η,τ,β,θ)':      {'eta': True,  'tau': True,  'beta': True,  'theta': True},
}


def ablation_what_to_learn(train_ds, test_ds, K=20):
    print(f"\n=== (B) What to learn (K={K}) ===")
    out = {}
    for name, learn in LEARN_VARIANTS.items():
        n_param = sum(int(v) for v in learn.values()) * K
        t0 = time.time()
        model, hist = train_model(train_ds, K, learn=learn)
        ev = eval_test(model, test_ds)
        out[name] = {'eval': ev, 'n_params': n_param,
                     'train_time': time.time()-t0, 'final_loss': hist[-1]}
        print(f"  {name:20s}  params={n_param:3d}  "
              f"median={ev['rel_x_err_median']:.2e}  mean={ev['rel_x_err_mean']:.2e}")
    return out


# ============================================================
# (C) Initialization sensitivity
# ============================================================

def ablation_init(train_ds, test_ds, K=20):
    print(f"\n=== (C) Initialization sensitivity (K={K}) ===")
    L = train_ds['op_norm_mean']
    configs = [
        ('η=0.9/||A||, β=halpern',   0.9/L, 'halpern',  None),
        ('η=0.45/||A||, β=halpern',  0.45/L, 'halpern', None),
        ('η=0.9/||A||, β=const0.5',  0.9/L, 'constant', 0.5),
        ('η=0.9/||A||, β=const0.9',  0.9/L, 'constant', 0.9),
        ('η=0.45/||A||, β=const0.9', 0.45/L, 'constant', 0.9),
    ]
    out = {}
    for name, ie, bsch, bval in configs:
        t0 = time.time()
        model, hist = train_model(train_ds, K, init_eta=ie, init_tau=ie,
                                  init_beta_schedule=bsch, init_beta=bval or 0.5)
        ev = eval_test(model, test_ds)
        out[name] = {'eval': ev, 'init_eta': ie, 'init_beta_schedule': bsch,
                     'init_beta': bval, 'train_time': time.time()-t0,
                     'final_loss': hist[-1]}
        print(f"  {name:32s}  median={ev['rel_x_err_median']:.2e}  "
              f"mean={ev['rel_x_err_mean']:.2e}")
    return out


# ============================================================
# (D) Training size
# ============================================================

def ablation_train_size(cfg, test_ds, K=20, sizes=(32, 128, 512)):
    print(f"\n=== (D) Train size (K={K}) ===")
    out = {}
    for n_tr in sizes:
        train_ds = build_dataset(cfg, n_tr, seed_offset=0)
        t0 = time.time()
        model, hist = train_model(train_ds, K)
        ev = eval_test(model, test_ds)
        out[n_tr] = {'eval': ev, 'train_time': time.time()-t0, 'final_loss': hist[-1]}
        print(f"  N_train={n_tr:4d}  median={ev['rel_x_err_median']:.2e}  "
              f"mean={ev['rel_x_err_mean']:.2e}  [{time.time()-t0:.1f}s]")
    return out


# ============================================================
# Plotting
# ============================================================

def plot_K_sweep(results):
    Ks = sorted(results.keys())
    pdhg = [results[k]['pdhg']['rel_x_err_median'] for k in Ks]
    pdqp = [results[k]['pdqp']['rel_x_err_median'] for k in Ks]
    learned = [results[k]['learned']['rel_x_err_median'] for k in Ks]
    plt.figure(figsize=(7, 5))
    plt.semilogy(Ks, pdhg, 'o-', label='PDHG (K)', color='tab:blue')
    plt.semilogy(Ks, pdqp, 's-', label='PDQP-style (K)', color='tab:red')
    plt.semilogy(Ks, learned, 'D-', label='Learned-PDHG-CP (K)', color='tab:green', lw=2)
    plt.xlabel('K (unrolled iterations)')
    plt.ylabel(r'median $\|x-x^\star\|/\|x^\star\|$')
    plt.title('K sweep — medium suite')
    plt.grid(alpha=0.3); plt.legend()
    out = os.path.join(OUT_DIR, 'socp_ablation_K_sweep.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_what_to_learn(results):
    names = list(results.keys())
    meds = [results[n]['eval']['rel_x_err_median'] for n in names]
    n_params = [results[n]['n_params'] for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(names)), meds, color=['gray', 'tab:blue', 'tab:orange', 'tab:green'])
    for bar, m, p in zip(bars, meds, n_params):
        ax.text(bar.get_x() + bar.get_width()/2, m * 1.05,
                f'{m:.2e}\n({p} params)', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_yscale('log')
    ax.set_ylabel(r'median $\|x-x^\star\|/\|x^\star\|$')
    ax.set_title('What to learn (K=20, medium)')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, 'socp_ablation_what_to_learn.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_init(results):
    names = list(results.keys())
    meds = [results[n]['eval']['rel_x_err_median'] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(names)), meds, color='tab:purple')
    for bar, m in zip(bars, meds):
        ax.text(bar.get_x() + bar.get_width()/2, m * 1.05,
                f'{m:.2e}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_yscale('log')
    ax.set_ylabel(r'median $\|x-x^\star\|/\|x^\star\|$')
    ax.set_title('Initialization sensitivity (K=20, medium)')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, 'socp_ablation_init.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_train_size(results):
    Ns = sorted(results.keys())
    meds = [results[n]['eval']['rel_x_err_median'] for n in Ns]
    plt.figure(figsize=(7, 5))
    plt.semilogy(Ns, meds, 'D-', color='tab:green', lw=2, markersize=8)
    for n, m in zip(Ns, meds):
        plt.text(n, m * 1.1, f'{m:.2e}', ha='center', fontsize=9)
    plt.xscale('log')
    plt.xlabel('# training instances')
    plt.ylabel(r'median $\|x-x^\star\|/\|x^\star\|$')
    plt.title('Train size (K=20, medium)')
    plt.grid(alpha=0.3)
    out = os.path.join(OUT_DIR, 'socp_ablation_train_size.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    print(f"Suite: medium  block={SUITE_CFG['block_sizes']}  m={SUITE_CFG['m']}")
    train_ds = build_dataset(SUITE_CFG, N_TRAIN_DEFAULT, seed_offset=0)
    test_ds  = build_dataset(SUITE_CFG, N_TEST, seed_offset=10000)
    print(f"  ||A|| ≈ {train_ds['op_norm_mean']:.3f}  "
          f"N={train_ds['N']}  m={train_ds['m']}")

    results = {}

    A = ablation_K_sweep(train_ds, test_ds, K_list=[5, 10, 20, 30, 50])
    plot_K_sweep(A);            results['A_K_sweep'] = A

    B = ablation_what_to_learn(train_ds, test_ds, K=20)
    plot_what_to_learn(B);      results['B_what_to_learn'] = B

    C = ablation_init(train_ds, test_ds, K=20)
    plot_init(C);               results['C_init'] = C

    D = ablation_train_size(SUITE_CFG, test_ds, K=20, sizes=(32, 128, 512))
    plot_train_size(D);         results['D_train_size'] = D

    out_json = os.path.join(OUT_DIR, 'socp_ablation_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_json}")

    # Markdown summary
    print("\n=========== Summary ===========")
    print("\n(A) K sweep, median rel_x_err:")
    print("| K | PDHG | PDQP-style | Learned | speedup vs PDHG |")
    print("|---|------|------------|---------|------------------|")
    for K in sorted(A.keys()):
        r = A[K]
        sp = r['pdhg']['rel_x_err_median'] / max(r['learned']['rel_x_err_median'], 1e-16)
        print(f"| {K} | {r['pdhg']['rel_x_err_median']:.2e} | "
              f"{r['pdqp']['rel_x_err_median']:.2e} | "
              f"**{r['learned']['rel_x_err_median']:.2e}** | {sp:.2f}× |")

    print("\n(B) What to learn (K=20):")
    print("| variant | # params | median | mean |")
    print("|---------|----------|--------|------|")
    for n, r in B.items():
        print(f"| {n} | {r['n_params']} | {r['eval']['rel_x_err_median']:.2e} | {r['eval']['rel_x_err_mean']:.2e} |")

    print("\n(C) Initialization (K=20):")
    print("| config | median | mean |")
    print("|--------|--------|------|")
    for n, r in C.items():
        print(f"| {n} | {r['eval']['rel_x_err_median']:.2e} | {r['eval']['rel_x_err_mean']:.2e} |")

    print("\n(D) Training size (K=20):")
    print("| N_train | median | mean |")
    print("|---------|--------|------|")
    for n, r in D.items():
        print(f"| {n} | {r['eval']['rel_x_err_median']:.2e} | {r['eval']['rel_x_err_mean']:.2e} |")


if __name__ == '__main__':
    main()
