"""
run_socp_stage2.py — Stage 2 (Learned-PDHG-CP-Couple) 训练与对比

实验:
  (E1) 跨 suite × K 主对比:
       PDHG / PDQP-style / Stage1 / Stage2(independent, λ=0)
  (E2) 变体消融 (medium, K=20):
       independent / shared / symmetric / shared_symmetric
       报告 #params, median rel_x_err, 训练时间, ||Δ_p||_F
  (E3) 正则 λ_couple 扫描 (medium, K=20, independent):
       λ ∈ {0, 1e-5, 1e-3, 1e-1}
       报告 (median rel_x_err, ||Δ_p||_F_total)
  + 学到的 Δ_p^{(0)} 与 A^T 的可视化 (medium, K=20)
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
from socp.pdhg_cp_couple import LearnedPDHGCPCouple, count_params, \
    supervised_mse_with_coupling

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')
torch.manual_seed(0); np.random.seed(0)

PROBLEM_SUITES = {
    'small':  {'block_sizes': [3, 3, 3, 3, 1, 1, 1],     'm': 10},
    'medium': {'block_sizes': [5]*6 + [1]*4,             'm': 20},
    'mixed':  {'block_sizes': [10, 6, 6, 3, 3, 1, 1, 1], 'm': 15},
}
N_TRAIN = 256
N_TEST = 50
NUM_EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-3


# ============================================================
# Dataset & shared helpers
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


def train_stage1(train_ds, K, num_epochs=NUM_EPOCHS, batch=BATCH_SIZE, lr=LR, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    L = train_ds['op_norm_mean']
    model = LearnedPDHGCP(
        K=K, init_eta=0.9/L, init_tau=0.9/L,
        init_beta_schedule='halpern', init_theta=0.999,
        block_sizes=train_ds['block_sizes'],
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_ds['A'].shape[0]
    hist = []
    for ep in range(num_epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_hat, _ = model(train_ds['A'][idx], train_ds['b'][idx], train_ds['c'][idx])
            loss = supervised_mse(x_hat, train_ds['x_star'][idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * idx.numel()
        hist.append(tot/n)
    return model, hist


def train_stage2(train_ds, K, coupling_mode='independent', lam_couple=0.0,
                 num_epochs=NUM_EPOCHS, batch=BATCH_SIZE, lr=LR, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    L = train_ds['op_norm_mean']
    A_tpl = train_ds['A'][0]  # 用第一条数据的 A 作 shape 参考
    model = LearnedPDHGCPCouple(
        K=K, A_template=A_tpl, block_sizes=train_ds['block_sizes'],
        init_eta=0.9/L, init_tau=0.9/L,
        init_beta_schedule='halpern', init_theta=0.999,
        coupling_mode=coupling_mode,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_ds['A'].shape[0]
    hist, hist_mse, hist_reg = [], [], []
    for ep in range(num_epochs):
        perm = torch.randperm(n); tot = tot_mse = tot_reg = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_hat, _ = model(train_ds['A'][idx], train_ds['b'][idx], train_ds['c'][idx])
            loss, mse, reg = supervised_mse_with_coupling(
                x_hat, train_ds['x_star'][idx], model, lam_couple=lam_couple)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * idx.numel()
            tot_mse += mse.item() * idx.numel()
            tot_reg += float(reg) * idx.numel()
        hist.append(tot/n); hist_mse.append(tot_mse/n); hist_reg.append(tot_reg/n)
    return model, {'loss': hist, 'mse': hist_mse, 'reg': hist_reg}


# ============================================================
# Evaluation
# ============================================================

def eval_model(model, test_ds):
    block = test_ds['block_sizes']
    rel_errs, primal = [], []
    n = test_ds['A'].shape[0]
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
        res = kkt_residuals(data, x, y)
        rel_errs.append(res['rel_x_err']); primal.append(res['primal_feas'])
    return {
        'rel_x_err_mean': float(np.mean(rel_errs)),
        'rel_x_err_median': float(np.median(rel_errs)),
        'primal_feas_median': float(np.median(primal)),
    }


# ============================================================
# (E1) suite × K main comparison
# ============================================================

K_OPTIONS = [10, 20, 30]

def exp_E1():
    print("\n========================= (E1) suite × K main comparison =========================")
    out = {}
    for sname, cfg in PROBLEM_SUITES.items():
        print(f"\n----- Suite: {sname}  block={cfg['block_sizes']}  m={cfg['m']} -----")
        train_ds = build_dataset(cfg, N_TRAIN, seed_offset=0)
        test_ds = build_dataset(cfg, N_TEST, seed_offset=10000)
        print(f"  ||A|| ≈ {train_ds['op_norm_mean']:.3f}; N={train_ds['N']}; m={train_ds['m']}")
        suite_out = {}
        for K in K_OPTIONS:
            ev_pdhg = eval_classical(test_ds, 'pdhg', K)
            ev_pdqp = eval_classical(test_ds, 'pdqp', K)
            t0 = time.time(); m1, _ = train_stage1(train_ds, K); t_s1 = time.time()-t0
            ev_s1 = eval_model(m1, test_ds)
            t0 = time.time(); m2, h2 = train_stage2(train_ds, K, 'independent', 0.0); t_s2 = time.time()-t0
            ev_s2 = eval_model(m2, test_ds)
            print(f"  K={K:2d}  PDHG={ev_pdhg['rel_x_err_median']:.2e}  "
                  f"PDQP={ev_pdqp['rel_x_err_median']:.2e}  "
                  f"S1={ev_s1['rel_x_err_median']:.2e} ({t_s1:.1f}s)  "
                  f"S2={ev_s2['rel_x_err_median']:.2e} ({t_s2:.1f}s)  "
                  f"||Δp||={float(torch.linalg.norm(m2.delta_p)):.2e}")
            suite_out[str(K)] = {
                'pdhg': ev_pdhg, 'pdqp': ev_pdqp,
                'stage1': {'eval': ev_s1, 'train_time': t_s1,
                           'params': sum(p.numel() for p in m1.parameters())},
                'stage2': {'eval': ev_s2, 'train_time': t_s2,
                           'params': count_params(m2),
                           'delta_norm_total': float(torch.linalg.norm(m2.delta_p)),
                           'delta_norms_per_layer': m2.delta_norms()},
            }
        out[sname] = suite_out
    return out


# ============================================================
# (E2) coupling mode ablation (medium, K=20)
# ============================================================

COUPLING_MODES = ['independent', 'shared', 'symmetric', 'shared_symmetric']

def exp_E2():
    print("\n========================= (E2) coupling-mode ablation (medium, K=20) =========================")
    cfg = PROBLEM_SUITES['medium']
    train_ds = build_dataset(cfg, N_TRAIN, seed_offset=0)
    test_ds = build_dataset(cfg, N_TEST, seed_offset=10000)
    K = 20
    out = {}
    # Stage 1 baseline at K=20
    m1, _ = train_stage1(train_ds, K)
    ev_s1 = eval_model(m1, test_ds)
    out['stage1'] = {'eval': ev_s1, 'params':
                     {'total': sum(p.numel() for p in m1.parameters())}}
    print(f"  Stage1 (ref)        params={out['stage1']['params']['total']:5d}  "
          f"median={ev_s1['rel_x_err_median']:.2e}")
    for mode in COUPLING_MODES:
        t0 = time.time()
        m2, _ = train_stage2(train_ds, K, coupling_mode=mode, lam_couple=0.0)
        elapsed = time.time() - t0
        ev = eval_model(m2, test_ds)
        cp = count_params(m2)
        dnorm = float(torch.linalg.norm(m2.delta_p))
        out[mode] = {'eval': ev, 'params': cp, 'train_time': elapsed,
                     'delta_norm_total': dnorm}
        print(f"  {mode:18s}  params={cp['total']:5d}  "
              f"median={ev['rel_x_err_median']:.2e}  mean={ev['rel_x_err_mean']:.2e}  "
              f"||Δp||={dnorm:.2e}  [{elapsed:.1f}s]")
    return out


# ============================================================
# (E3) coupling regularization sweep
# ============================================================

LAMBDAS = [0.0, 1e-5, 1e-3, 1e-1]

def exp_E3():
    print("\n========================= (E3) coupling regularization λ sweep =========================")
    cfg = PROBLEM_SUITES['medium']
    train_ds = build_dataset(cfg, N_TRAIN, seed_offset=0)
    test_ds = build_dataset(cfg, N_TEST, seed_offset=10000)
    K = 20
    out = {}
    for lam in LAMBDAS:
        t0 = time.time()
        m2, _ = train_stage2(train_ds, K, coupling_mode='independent', lam_couple=lam)
        elapsed = time.time() - t0
        ev = eval_model(m2, test_ds)
        dnorm = float(torch.linalg.norm(m2.delta_p))
        out[str(lam)] = {'eval': ev, 'delta_norm_total': dnorm,
                         'train_time': elapsed}
        print(f"  λ={lam:.0e}  median={ev['rel_x_err_median']:.2e}  "
              f"mean={ev['rel_x_err_mean']:.2e}  ||Δp||={dnorm:.2e}  [{elapsed:.1f}s]")
    return out, train_ds


# ============================================================
# Visualization
# ============================================================

def plot_E1(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4), sharey=True)
    if len(results) == 1: axes = [axes]
    for ax, (sname, sout) in zip(axes, results.items()):
        Ks = sorted(int(k) for k in sout.keys())
        pdhg = [sout[str(k)]['pdhg']['rel_x_err_median'] for k in Ks]
        pdqp = [sout[str(k)]['pdqp']['rel_x_err_median'] for k in Ks]
        s1 = [sout[str(k)]['stage1']['eval']['rel_x_err_median'] for k in Ks]
        s2 = [sout[str(k)]['stage2']['eval']['rel_x_err_median'] for k in Ks]
        ax.semilogy(Ks, pdhg, 'o-', label='PDHG', color='tab:blue')
        ax.semilogy(Ks, pdqp, 's-', label='PDQP-style', color='tab:red')
        ax.semilogy(Ks, s1, '^-', label='Stage 1', color='tab:orange', lw=2)
        ax.semilogy(Ks, s2, 'D-', label='Stage 2 (indep)', color='tab:green', lw=2)
        ax.set_title(sname); ax.set_xlabel('K'); ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'median $\|x-x^\star\|/\|x^\star\|$')
    axes[-1].legend(); plt.tight_layout()
    out = os.path.join(OUT_DIR, 'socp_stage2_E1_suite_K.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_E2(results):
    modes = [k for k in results.keys() if k != 'stage1']
    meds = [results[m]['eval']['rel_x_err_median'] for m in modes]
    n_params = [results[m]['params']['total'] for m in modes]
    s1_med = results['stage1']['eval']['rel_x_err_median']
    s1_p = results['stage1']['params']['total']

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = list(range(len(modes) + 1))
    all_meds = [s1_med] + meds; all_p = [s1_p] + n_params
    labels = ['stage1\n(ref)'] + modes
    colors = ['gray'] + ['tab:blue', 'tab:cyan', 'tab:purple', 'tab:olive']
    bars = ax.bar(xs, all_meds, color=colors)
    for b, m, p in zip(bars, all_meds, all_p):
        ax.text(b.get_x() + b.get_width()/2, m * 1.05,
                f'{m:.2e}\n({p})', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_yscale('log'); ax.set_ylabel(r'median rel. err')
    ax.set_title('Coupling-mode ablation (medium, K=20)')
    ax.grid(axis='y', alpha=0.3)
    out = os.path.join(OUT_DIR, 'socp_stage2_E2_modes.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_E3(results):
    lams = sorted(float(k) for k in results.keys())
    meds = [results[str(l)]['eval']['rel_x_err_median'] for l in lams]
    dn = [results[str(l)]['delta_norm_total'] for l in lams]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(lams)), meds, 'D-', color='tab:green', lw=2, label='median err')
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f'{l:.0e}' for l in lams])
    ax.set_yscale('log'); ax.set_ylabel('median rel. err', color='tab:green')
    ax.tick_params(axis='y', labelcolor='tab:green')
    ax2 = ax.twinx()
    ax2.plot(range(len(lams)), dn, 'o--', color='tab:purple', label='||Δp||_F')
    ax2.set_ylabel(r'$\|\Delta_p\|_F$', color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax2.set_yscale('log')
    ax.set_xlabel(r'coupling reg. $\lambda$')
    ax.set_title('Coupling regularization sweep (medium, K=20)')
    ax.grid(alpha=0.3)
    out = os.path.join(OUT_DIR, 'socp_stage2_E3_lambda.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


def plot_delta_visualization(train_ds, K=20):
    """训练一个 indep Stage 2，可视化 Δ_p^{(0)} 与 A^T 的对比。"""
    m2, _ = train_stage2(train_ds, K, coupling_mode='independent', lam_couple=0.0)
    A_tpl = train_ds['A'][0].numpy()
    At = A_tpl.T
    dp0 = m2.delta_p[0].detach().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vmax = max(np.max(np.abs(At)), np.max(np.abs(dp0)))
    axes[0].imshow(At, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[0].set_title(r'$A^T$ (template)')
    axes[1].imshow(dp0, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[1].set_title(r'$\Delta_p^{(0)}$ (learned)')
    axes[2].imshow(At + dp0, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    axes[2].set_title(r'$B_p^{(0)} = A^T + \Delta_p^{(0)}$')
    for ax in axes: ax.set_xlabel('m'); ax.set_ylabel('N')
    plt.suptitle(f'Learned coupling matrix (medium, K=20, indep, λ=0)')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'socp_stage2_delta_visualization.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")
    # delta norms per layer
    fig, ax = plt.subplots(figsize=(8, 4))
    dn = m2.delta_norms()
    ax.plot(dn['delta_p'], 'o-', label=r'$\|\Delta_p^{(k)}\|_F$', color='tab:purple')
    ax.plot(dn['delta_d'], 's-', label=r'$\|\Delta_d^{(k)}\|_F$', color='tab:olive')
    ax.set_xlabel('layer k'); ax.set_ylabel('Frobenius norm')
    ax.set_title('Per-layer coupling magnitudes')
    ax.grid(alpha=0.3); ax.legend()
    out = os.path.join(OUT_DIR, 'socp_stage2_delta_norms.png')
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print(f"  saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    results = {}
    results['E1_suite_K'] = exp_E1();  plot_E1(results['E1_suite_K'])
    results['E2_modes'] = exp_E2();    plot_E2(results['E2_modes'])
    E3_out, medium_train = exp_E3();   results['E3_lambda'] = E3_out
    plot_E3(results['E3_lambda'])
    plot_delta_visualization(medium_train, K=20)

    out_json = os.path.join(OUT_DIR, 'socp_stage2_results.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {out_json}")

    # ----- Markdown summaries -----
    print("\n=========== (E1) Summary: median rel_x_err on test set ===========")
    print("| Suite | K | PDHG | PDQP | Stage1 | **Stage2 (indep, λ=0)** | S2 vs S1 |")
    print("|-------|---|------|------|--------|--------------------------|----------|")
    for sname, sout in results['E1_suite_K'].items():
        for K in sorted(int(k) for k in sout.keys()):
            r = sout[str(K)]
            s1_med = r['stage1']['eval']['rel_x_err_median']
            s2_med = r['stage2']['eval']['rel_x_err_median']
            speedup = s1_med / max(s2_med, 1e-16)
            print(f"| {sname} | {K} | {r['pdhg']['rel_x_err_median']:.2e} | "
                  f"{r['pdqp']['rel_x_err_median']:.2e} | "
                  f"{s1_med:.2e} | **{s2_med:.2e}** | {speedup:.2f}× |")

    print("\n=========== (E2) Coupling-mode ablation (medium, K=20) ===========")
    print("| Variant | #params | median | mean | ||Δp||_F |")
    print("|---------|---------|--------|------|----------|")
    r = results['E2_modes']
    print(f"| stage1 (ref) | {r['stage1']['params']['total']} | "
          f"{r['stage1']['eval']['rel_x_err_median']:.2e} | "
          f"{r['stage1']['eval']['rel_x_err_mean']:.2e} | - |")
    for mode in COUPLING_MODES:
        rr = r[mode]
        print(f"| {mode} | {rr['params']['total']} | "
              f"{rr['eval']['rel_x_err_median']:.2e} | "
              f"{rr['eval']['rel_x_err_mean']:.2e} | {rr['delta_norm_total']:.2e} |")

    print("\n=========== (E3) Coupling regularization λ sweep ===========")
    print("| λ | median | mean | ||Δp||_F |")
    print("|---|--------|------|----------|")
    for lam in LAMBDAS:
        rr = results['E3_lambda'][str(lam)]
        print(f"| {lam:.0e} | {rr['eval']['rel_x_err_median']:.2e} | "
              f"{rr['eval']['rel_x_err_mean']:.2e} | {rr['delta_norm_total']:.2e} |")


if __name__ == '__main__':
    main()
