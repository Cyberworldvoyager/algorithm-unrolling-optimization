"""
run_socp_stage2_fixedA.py — 验证 Stage 2 的"varying A 退化"假说

只改一处: 数据集使用 *单一固定 A*，只 x_star 随 seed 变。
评测同分布的测试集 (也是同一 A)。
预期: Stage 2 (indep) ≥ Stage 1，因为 LISTA-CP 那套 "B = A^T + Δ" 在固定 A 下才有意义。
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from socp.problem import (generate_socp_data, kkt_residuals,
                          _sample_soc_interior, _sample_soc_boundary,
                          _soc_dual_at_boundary)
from socp.classical import pdhg_socp, estimate_op_norm
from socp.pdhg_cp import LearnedPDHGCP, supervised_mse
from socp.pdhg_cp_couple import LearnedPDHGCPCouple, supervised_mse_with_coupling, count_params

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report')


def generate_socp_data_fixed_A(A: np.ndarray, block_sizes, m, seed: int, p_active=0.4):
    """复用 generate_socp_data 的 (x*, s*, y*) 采样逻辑，但 A 由外部固定。"""
    rng = np.random.RandomState(seed)
    N = sum(block_sizes)
    x_star = np.zeros(N); s_star = np.zeros(N)
    offset = 0
    for k in block_sizes:
        end = offset + k
        r = rng.rand()
        if r < (1 - p_active) / 2:
            x_star[offset:end] = _sample_soc_interior(k, rng)
        elif r < 1 - p_active:
            s_star[offset:end] = _sample_soc_interior(k, rng)
        else:
            if k == 1:
                (x_star if rng.rand() < 0.5 else s_star)[offset:end] = \
                    _sample_soc_interior(k, rng)
            else:
                z = _sample_soc_boundary(k, rng)
                x_star[offset:end] = z
                s_star[offset:end] = _soc_dual_at_boundary(z, rng)
        offset = end
    y_star = rng.randn(m)
    b = A @ x_star
    c = A.T @ y_star + s_star
    return {'A': A, 'b': b, 'c': c, 'block_sizes': list(block_sizes),
            'N': N, 'm': m, 'x_star': x_star, 'y_star': y_star, 's_star': s_star}


def build_dataset_fixed_A(cfg, num, seed_offset, A_seed=0):
    """生成同 A 的数据集。"""
    block = cfg['block_sizes']; m = cfg['m']
    # 用一个 dummy 调用拿到 A
    init = generate_socp_data(block_sizes=block, m=m, seed=A_seed)
    A = init['A']
    L = estimate_op_norm(A, n_iter=50)

    As, bs, cs, xs = [], [], [], []
    for i in range(num):
        d = generate_socp_data_fixed_A(A, block, m, seed=seed_offset + 1 + i)
        As.append(A); bs.append(d['b']); cs.append(d['c']); xs.append(d['x_star'])
    return {
        'A': torch.tensor(np.stack(As).astype(np.float32)),
        'b': torch.tensor(np.stack(bs).astype(np.float32)),
        'c': torch.tensor(np.stack(cs).astype(np.float32)),
        'x_star': torch.tensor(np.stack(xs).astype(np.float32)),
        'block_sizes': block, 'm': m, 'N': sum(block),
        'op_norm_mean': float(L),
    }


def train_stage1(ds, K, num_epochs=80, batch=32, lr=5e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    L = ds['op_norm_mean']
    m = LearnedPDHGCP(K=K, init_eta=0.9/L, init_tau=0.9/L,
                      init_beta_schedule='halpern', init_theta=0.999,
                      block_sizes=ds['block_sizes'])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = ds['A'].shape[0]
    for ep in range(num_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_hat, _ = m(ds['A'][idx], ds['b'][idx], ds['c'][idx])
            loss = supervised_mse(x_hat, ds['x_star'][idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m


def train_stage2(ds, K, mode='independent', lam=0.0, num_epochs=80, batch=32, lr=5e-3, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    L = ds['op_norm_mean']
    m = LearnedPDHGCPCouple(K=K, A_template=ds['A'][0],
                            block_sizes=ds['block_sizes'],
                            init_eta=0.9/L, init_tau=0.9/L,
                            init_beta_schedule='halpern', init_theta=0.999,
                            coupling_mode=mode)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = ds['A'].shape[0]
    for ep in range(num_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_hat, _ = m(ds['A'][idx], ds['b'][idx], ds['c'][idx])
            loss, mse, reg = supervised_mse_with_coupling(
                x_hat, ds['x_star'][idx], m, lam_couple=lam)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
    return m


def eval_model(model, ds):
    rel = []
    with torch.no_grad():
        x_hat, y_hat = model(ds['A'], ds['b'], ds['c'])
    for i in range(ds['A'].shape[0]):
        xs = ds['x_star'][i].numpy().astype(np.float64)
        err = float(np.linalg.norm(x_hat[i].numpy().astype(np.float64) - xs) /
                    (np.linalg.norm(xs) + 1e-12))
        rel.append(err)
    return {'mean': float(np.mean(rel)), 'median': float(np.median(rel))}


def main():
    cfg = {'block_sizes': [5]*6 + [1]*4, 'm': 20}  # medium

    print("===== Fixed-A regime (same A across train/test) =====")
    train_ds = build_dataset_fixed_A(cfg, 256, seed_offset=0, A_seed=42)
    test_ds = build_dataset_fixed_A(cfg, 50, seed_offset=10000, A_seed=42)
    print(f"  ||A|| ≈ {train_ds['op_norm_mean']:.3f}; N={train_ds['N']}; m={train_ds['m']}")

    out_fixed = {}
    for K in [10, 20, 30]:
        m1 = train_stage1(train_ds, K)
        ev1 = eval_model(m1, test_ds)
        m2 = train_stage2(train_ds, K, mode='independent', lam=0.0)
        ev2 = eval_model(m2, test_ds)
        dp_norm = float(torch.linalg.norm(m2.delta_p))
        print(f"  K={K:2d}  S1={ev1['median']:.2e}  S2={ev2['median']:.2e}  "
              f"S2/S1={ev2['median']/ev1['median']:.2f}×  ||Δp||={dp_norm:.2e}")
        out_fixed[K] = {'stage1': ev1, 'stage2': ev2, 'delta_p_norm': dp_norm}

    print("\n===== Varying-A regime (each instance different A, recap from E1) =====")
    print("  K=10  S1=2.20e-01  S2=4.40e-01  (S2/S1=2.00× WORSE)")
    print("  K=20  S1=1.25e-01  S2=3.32e-01  (S2/S1=2.66× WORSE)")
    print("  K=30  S1=9.69e-02  S2=2.64e-01  (S2/S1=2.72× WORSE)")

    out = {'fixed_A': out_fixed}
    with open(os.path.join(OUT_DIR, 'socp_stage2_fixedA_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: report/socp_stage2_fixedA_results.json")


if __name__ == '__main__':
    main()
