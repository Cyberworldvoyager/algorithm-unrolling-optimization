"""
run_socp_learned.py — Stage 1: Learned-PDHG-CP 训练与对比

实验流程:
  1. 对每个 suite 生成训练集 (固定锥结构，A/b/c/x* 多 seed)
  2. 训练 Learned-PDHG-CP (K = 10, 30 两档)
  3. 测试集上对比:
        - 裸 PDHG (相同 K)
        - 裸 PDHG (K=2000，作上限参照)
        - PDQP-style (相同 K)
        - Learned-PDHG-CP (相同 K)
  4. 输出:
        - 对比表 (JSON + Markdown)
        - 学到的 (η_k, τ_k, β_k, θ_k) 调度图
        - 各方法的收敛曲线 (同 K, 在测试集上)
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from socp.problem import generate_socp_data, kkt_residuals, solve_with_scs
from socp.classical import pdhg_socp, pdqp_socp, estimate_op_norm
from socp.pdhg_cp import LearnedPDHGCP, supervised_mse


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cpu')   # 实例很小，CPU 即可
torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Configurations
# ============================================================

PROBLEM_SUITES = {
    'small':  {'block_sizes': [3, 3, 3, 3, 1, 1, 1],     'm': 10},
    'medium': {'block_sizes': [5]*6 + [1]*4,             'm': 20},
    'mixed':  {'block_sizes': [10, 6, 6, 3, 3, 1, 1, 1], 'm': 15},
}

N_TRAIN = 256
N_TEST = 50
K_OPTIONS = [10, 30]      # 展开层数
NUM_EPOCHS = 80
BATCH_SIZE = 32
LR = 5e-3


# ============================================================
# Data
# ============================================================

def build_dataset(suite_cfg: dict, num: int, seed_offset: int):
    """生成一个数据集，存为 stacked tensor。"""
    block = suite_cfg['block_sizes']
    m = suite_cfg['m']
    As, bs, cs, xs = [], [], [], []
    op_norms = []
    for i in range(num):
        d = generate_socp_data(block_sizes=block, m=m, seed=seed_offset + i)
        As.append(d['A']); bs.append(d['b']); cs.append(d['c']); xs.append(d['x_star'])
        op_norms.append(estimate_op_norm(d['A'], n_iter=30))
    As = np.stack(As).astype(np.float32)
    bs = np.stack(bs).astype(np.float32)
    cs = np.stack(cs).astype(np.float32)
    xs = np.stack(xs).astype(np.float32)
    return {
        'A': torch.tensor(As), 'b': torch.tensor(bs),
        'c': torch.tensor(cs), 'x_star': torch.tensor(xs),
        'block_sizes': block, 'm': m, 'N': sum(block),
        'op_norm_mean': float(np.mean(op_norms)),
    }


# ============================================================
# Training
# ============================================================

def train_learned_pdhg(train_ds: dict, K: int, num_epochs=NUM_EPOCHS,
                       batch_size=BATCH_SIZE, lr=LR, verbose=True):
    L = train_ds['op_norm_mean']
    init_step = 0.9 / max(L, 1e-8)
    model = LearnedPDHGCP(
        K=K,
        init_eta=init_step,
        init_tau=init_step,
        init_beta_schedule='halpern',
        init_theta=0.999,
        block_sizes=train_ds['block_sizes'],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_ds['A'].shape[0]

    history = []
    for epoch in range(num_epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            A_b = train_ds['A'][idx].to(DEVICE)
            b_b = train_ds['b'][idx].to(DEVICE)
            c_b = train_ds['c'][idx].to(DEVICE)
            x_b = train_ds['x_star'][idx].to(DEVICE)

            x_hat, _ = model(A_b, b_b, c_b)
            loss = supervised_mse(x_hat, x_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * idx.numel()
        avg = total / n
        history.append(avg)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}/{num_epochs}  loss={avg:.4e}")

    return model, history


# ============================================================
# Evaluation
# ============================================================

def eval_method_on_test(test_ds: dict, method: str, K: int, model=None) -> dict:
    """在测试集上跑某方法的最终 K 步，返回逐实例的 KKT 指标。"""
    n_test = test_ds['A'].shape[0]
    block = test_ds['block_sizes']
    rel_errs, primal, dual_resid, obj_gap = [], [], [], []

    for i in range(n_test):
        A_np = test_ds['A'][i].numpy().astype(np.float64)
        b_np = test_ds['b'][i].numpy().astype(np.float64)
        c_np = test_ds['c'][i].numpy().astype(np.float64)
        x_star = test_ds['x_star'][i].numpy().astype(np.float64)
        data = {
            'A': A_np, 'b': b_np, 'c': c_np,
            'block_sizes': block, 'N': test_ds['N'], 'm': test_ds['m'],
            'x_star': x_star,
        }

        if method == 'pdhg':
            x, y, _ = pdhg_socp(data, K_iters=K, return_history=False)
        elif method == 'pdqp':
            x, y, _ = pdqp_socp(data, K_iters=K, return_history=False)
        elif method == 'learned':
            with torch.no_grad():
                A_t = test_ds['A'][i:i+1]
                b_t = test_ds['b'][i:i+1]
                c_t = test_ds['c'][i:i+1]
                x_hat, y_hat = model(A_t, b_t, c_t)
            x = x_hat[0].numpy().astype(np.float64)
            y = y_hat[0].numpy().astype(np.float64)
        else:
            raise ValueError(method)

        res = kkt_residuals(data, x, y)
        rel_errs.append(res['rel_x_err'])
        primal.append(res['primal_feas'])
        dual_resid.append(res['dual_resid'])
        obj_gap.append(abs(res['obj_gap']))

    return {
        'rel_x_err_mean': float(np.mean(rel_errs)),
        'rel_x_err_median': float(np.median(rel_errs)),
        'primal_feas_mean': float(np.mean(primal)),
        'primal_feas_median': float(np.median(primal)),
        'dual_resid_mean': float(np.mean(dual_resid)),
        'dual_resid_median': float(np.median(dual_resid)),
        'obj_gap_median': float(np.median(obj_gap)),
        'rel_x_err_all': rel_errs,
    }


def eval_pdhg_long(test_ds: dict, K_long: int = 2000) -> dict:
    """裸 PDHG K_long 步作为参照上限。"""
    return eval_method_on_test(test_ds, 'pdhg', K_long)


# ============================================================
# Plotting
# ============================================================

def plot_schedules(model: LearnedPDHGCP, suite_name: str, K: int):
    sch = model.schedules()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    keys = ['eta', 'tau', 'beta', 'theta']
    titles = [r'$\eta_k$ (primal step)', r'$\tau_k$ (dual step)',
              r'$\beta_k$ (Halpern)',    r'$\theta_k$ (overrelax.)']
    for ax, k, t in zip(axes.flatten(), keys, titles):
        vals = sch[k]
        ax.plot(range(len(vals)), vals, 'o-', color='tab:purple')
        ax.set_title(t); ax.set_xlabel('layer k'); ax.grid(alpha=0.3)
    plt.suptitle(f'Learned schedule — {suite_name} (K={K})')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'socp_learned_schedule_{suite_name}_K{K}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {out}")


def plot_test_convergence(test_ds: dict, model: LearnedPDHGCP, suite_name: str, K: int,
                          K_long: int = 2000, n_show: int = 5):
    """前 n_show 个测试实例：三种方法的逐步 rel_x_err 曲线。"""
    block = test_ds['block_sizes']
    n_show = min(n_show, test_ds['A'].shape[0])

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), sharey=True)
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        ax = axes[i]
        A_np = test_ds['A'][i].numpy().astype(np.float64)
        b_np = test_ds['b'][i].numpy().astype(np.float64)
        c_np = test_ds['c'][i].numpy().astype(np.float64)
        x_star = test_ds['x_star'][i].numpy().astype(np.float64)
        data = {'A': A_np, 'b': b_np, 'c': c_np,
                'block_sizes': block, 'N': test_ds['N'], 'm': test_ds['m'],
                'x_star': x_star}

        # 裸 PDHG/PDQP 长曲线
        _, _, h_pdhg = pdhg_socp(data, K_iters=K_long, record_every=max(1, K_long//200))
        _, _, h_pdqp = pdqp_socp(data, K_iters=K_long, record_every=max(1, K_long//200))
        ax.semilogy(h_pdhg['iter'], np.maximum(h_pdhg['rel_x_err'], 1e-16),
                    label='PDHG', color='tab:blue')
        ax.semilogy(h_pdqp['iter'], np.maximum(h_pdqp['rel_x_err'], 1e-16),
                    label='PDQP-style', color='tab:red')

        # Learned: 每层取出 trajectory
        A_t = test_ds['A'][i:i+1]
        b_t = test_ds['b'][i:i+1]
        c_t = test_ds['c'][i:i+1]
        with torch.no_grad():
            _, _, traj = model(A_t, b_t, c_t, return_trajectory=True)
        re_layer = []
        xs_ref = test_ds['x_star'][i].numpy()
        norm_ref = np.linalg.norm(xs_ref) + 1e-12
        for (x_k, _) in traj:
            re_layer.append(float(np.linalg.norm(x_k[0].numpy() - xs_ref) / norm_ref))
        layer_iters = list(range(len(re_layer)))
        ax.semilogy(layer_iters, np.maximum(re_layer, 1e-16),
                    'o-', label=f'Learned-PDHG-CP (K={K})', color='tab:green', markersize=4)

        ax.set_title(f'seed={i}'); ax.set_xlabel('iter / layer'); ax.grid(alpha=0.3)
        if i == 0:
            ax.set_ylabel(r'$\|x - x^\star\| / \|x^\star\|$')
            ax.legend()
    plt.suptitle(f'Test convergence — {suite_name} (Learned K={K})')
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'socp_learned_curve_{suite_name}_K{K}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  saved: {out}")


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}
    for name, cfg in PROBLEM_SUITES.items():
        print(f"\n========== Suite: {name}  block={cfg['block_sizes']} m={cfg['m']} ==========")
        train_ds = build_dataset(cfg, N_TRAIN, seed_offset=0)
        test_ds = build_dataset(cfg, N_TEST, seed_offset=10000)
        print(f"  ||A|| ≈ {train_ds['op_norm_mean']:.3f}; "
              f"N={train_ds['N']}, m={train_ds['m']}, train={N_TRAIN}, test={N_TEST}")

        suite_res = {'config': cfg, 'op_norm_mean': train_ds['op_norm_mean'], 'by_K': {}}

        # 长跑参照
        print("  >>> PDHG (K=2000) reference ...")
        ref = eval_pdhg_long(test_ds, K_long=2000)
        suite_res['pdhg_K2000'] = {k: v for k, v in ref.items() if k != 'rel_x_err_all'}
        print(f"      rel_x_err median={ref['rel_x_err_median']:.2e}")

        for K in K_OPTIONS:
            print(f"\n  ----- K = {K} -----")
            t0 = time.time()
            model, hist = train_learned_pdhg(train_ds, K)
            train_time = time.time() - t0
            print(f"    [train done in {train_time:.1f}s]")

            ev_pdhg    = eval_method_on_test(test_ds, 'pdhg', K)
            ev_pdqp    = eval_method_on_test(test_ds, 'pdqp', K)
            ev_learned = eval_method_on_test(test_ds, 'learned', K, model=model)

            print(f"    PDHG-K{K}      : rel_x_err median={ev_pdhg['rel_x_err_median']:.2e}, mean={ev_pdhg['rel_x_err_mean']:.2e}")
            print(f"    PDQP-K{K}      : rel_x_err median={ev_pdqp['rel_x_err_median']:.2e}, mean={ev_pdqp['rel_x_err_mean']:.2e}")
            print(f"    Learned-K{K}   : rel_x_err median={ev_learned['rel_x_err_median']:.2e}, mean={ev_learned['rel_x_err_mean']:.2e}")

            plot_schedules(model, name, K)
            plot_test_convergence(test_ds, model, name, K)

            suite_res['by_K'][str(K)] = {
                'pdhg': {k: v for k, v in ev_pdhg.items() if k != 'rel_x_err_all'},
                'pdqp': {k: v for k, v in ev_pdqp.items() if k != 'rel_x_err_all'},
                'learned': {k: v for k, v in ev_learned.items() if k != 'rel_x_err_all'},
                'train_loss_hist': hist,
                'train_time': train_time,
                'learned_schedule': model.schedules(),
            }

        all_results[name] = suite_res

    out_json = os.path.join(OUT_DIR, 'socp_learned_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults: {out_json}")

    # Markdown 总结
    print("\n=== Summary: median rel_x_err on test set ===")
    print("| Suite | K | PDHG(K) | PDQP-style(K) | Learned-PDHG-CP(K) | PDHG(K=2000) |")
    print("|-------|---|---------|---------------|---------------------|--------------|")
    for name, res in all_results.items():
        ref_med = res['pdhg_K2000']['rel_x_err_median']
        for K in K_OPTIONS:
            r = res['by_K'][str(K)]
            print(f"| {name} | {K} | {r['pdhg']['rel_x_err_median']:.2e} | "
                  f"{r['pdqp']['rel_x_err_median']:.2e} | "
                  f"**{r['learned']['rel_x_err_median']:.2e}** | "
                  f"{ref_med:.2e} |")


if __name__ == '__main__':
    main()
