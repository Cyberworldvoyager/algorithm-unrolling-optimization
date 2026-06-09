"""
run_socp_baseline.py — SOCP Stage A 基线实验

流程:
  1) 生成多组随机 SOCP 实例 (多种锥结构与维度)
  2) 用 SCS (CVXPY) 求精确解作为参考
  3) 跑裸 PDHG 与 PDQP 风格基线
  4) 输出对比表 + 收敛曲线图 + JSON 结果
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from socp.problem import generate_socp_data, kkt_residuals, solve_with_scs
from socp.classical import pdhg_socp, pdqp_socp, estimate_op_norm


OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- problem suites ----------------
PROBLEM_SUITES = {
    'small':  {'block_sizes': [3, 3, 3, 3, 1, 1, 1],     'm': 10},
    'medium': {'block_sizes': [5]*6 + [1]*4,             'm': 20},
    'large':  {'block_sizes': [8]*8 + [1]*8,             'm': 40},
    'mixed':  {'block_sizes': [10, 6, 6, 3, 3, 1, 1, 1], 'm': 15},
}

K_ITERS = 2000
SEEDS = list(range(10))


def relerr(x_ref, x):
    return float(np.linalg.norm(x - x_ref) / (np.linalg.norm(x_ref) + 1e-12))


def run_one(data, label, fn, **kwargs):
    t0 = time.time()
    x, y, hist = fn(data, K_iters=K_ITERS, **kwargs)
    elapsed = time.time() - t0
    final = {k: hist[k][-1] for k in hist if k != 'iter'}
    final['time_sec'] = elapsed
    final['label'] = label
    return x, y, hist, final


def run_suite(name: str, cfg: dict):
    print(f"\n========== Suite: {name}  (block={cfg['block_sizes']}, m={cfg['m']}) ==========")
    rows = []
    curves = {'PDHG': [], 'PDQP-style': []}
    for seed in SEEDS:
        data = generate_socp_data(seed=seed, **cfg)
        # ground truth (validate generator)
        sol = solve_with_scs(data)
        err_gen_vs_scs = relerr(sol['x'], data['x_star'])
        # PDHG
        x_p, y_p, hist_p, fin_p = run_one(data, 'PDHG', pdhg_socp)
        # PDQP-style
        x_q, y_q, hist_q, fin_q = run_one(data, 'PDQP-style', pdqp_socp)

        row = {
            'seed': seed,
            'N': data['N'], 'm': data['m'],
            'scs_obj': sol['obj'],
            'gen_vs_scs_rel_x': err_gen_vs_scs,
            'pdhg': fin_p,
            'pdqp': fin_q,
        }
        rows.append(row)
        curves['PDHG'].append(hist_p)
        curves['PDQP-style'].append(hist_q)
        print(f" seed={seed:2d}  scs_obj={sol['obj']:+.4f}  "
              f"PDHG[rel_x={fin_p['rel_x_err']:.2e}, pf={fin_p['primal_feas']:.2e}, t={fin_p['time_sec']:.2f}s]  "
              f"PDQP[rel_x={fin_q['rel_x_err']:.2e}, pf={fin_q['primal_feas']:.2e}, t={fin_q['time_sec']:.2f}s]")

    return rows, curves


def summarize(rows):
    def agg(key):
        vs = [r['pdhg'][key] for r in rows]
        vq = [r['pdqp'][key] for r in rows]
        return (float(np.mean(vs)), float(np.median(vs)),
                float(np.mean(vq)), float(np.median(vq)))
    return {
        'rel_x_err': agg('rel_x_err'),
        'primal_feas': agg('primal_feas'),
        'dual_resid': agg('dual_resid'),
        'obj_gap': agg('obj_gap'),
        'time_sec': agg('time_sec'),
    }


def plot_curves(curves: dict, suite_name: str):
    """绘制 (seed=0) 的收敛曲线 + 中位数曲线。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_keys = [('rel_x_err', 'Relative ||x - x*|| / ||x*||'),
                   ('primal_feas', 'Primal residual ||Ax-b||'),
                   ('dual_resid', 'Dual residual')]
    colors = {'PDHG': 'tab:blue', 'PDQP-style': 'tab:red'}

    for ax, (mk, title) in zip(axes, metric_keys):
        for label, hist_list in curves.items():
            iters = np.array(hist_list[0]['iter'])
            stacked = np.stack([np.array(h[mk]) for h in hist_list], axis=0)
            stacked = np.maximum(stacked, 1e-16)
            med = np.median(stacked, axis=0)
            lo = np.percentile(stacked, 25, axis=0)
            hi = np.percentile(stacked, 75, axis=0)
            ax.plot(iters, med, label=label, color=colors[label], lw=2)
            ax.fill_between(iters, lo, hi, alpha=0.2, color=colors[label])
        ax.set_yscale('log')
        ax.set_xlabel('iter k')
        ax.set_ylabel(title)
        ax.set_title(f'{suite_name}: {mk}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'socp_baseline_{suite_name}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  saved: {out}")


def main():
    all_results = {}
    for name, cfg in PROBLEM_SUITES.items():
        rows, curves = run_suite(name, cfg)
        summary = summarize(rows)
        all_results[name] = {'config': cfg, 'summary': summary, 'rows': rows}
        plot_curves(curves, name)

    # 输出 JSON
    out_json = os.path.join(OUT_DIR, 'socp_baseline_results.json')
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults written to: {out_json}")

    # 打印 Markdown 表
    print("\n=== Summary Table (mean / median over 10 seeds, K=%d) ===" % K_ITERS)
    print("| Suite | Method | rel_x_err | primal_feas | dual_resid | obj_gap | time(s) |")
    print("|-------|--------|-----------|-------------|------------|---------|---------|")
    for name, res in all_results.items():
        s = res['summary']
        print(f"| {name} | PDHG       | "
              f"{s['rel_x_err'][0]:.2e} ({s['rel_x_err'][1]:.1e}) | "
              f"{s['primal_feas'][0]:.2e} ({s['primal_feas'][1]:.1e}) | "
              f"{s['dual_resid'][0]:.2e} ({s['dual_resid'][1]:.1e}) | "
              f"{s['obj_gap'][0]:.2e} ({s['obj_gap'][1]:.1e}) | "
              f"{s['time_sec'][0]:.2f} |")
        print(f"| {name} | PDQP-style | "
              f"{s['rel_x_err'][2]:.2e} ({s['rel_x_err'][3]:.1e}) | "
              f"{s['primal_feas'][2]:.2e} ({s['primal_feas'][3]:.1e}) | "
              f"{s['dual_resid'][2]:.2e} ({s['dual_resid'][3]:.1e}) | "
              f"{s['obj_gap'][2]:.2e} ({s['obj_gap'][3]:.1e}) | "
              f"{s['time_sec'][2]:.2f} |")


if __name__ == '__main__':
    main()
