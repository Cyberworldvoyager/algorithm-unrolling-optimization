"""
run_socp_stage2_warmstart.py — Stage 2 诊断 #2: warm-start

假设: 27k+ 参数同时训会"稀释"优化预算，使得标量调度 (η,τ,β,θ) 训不到位。
策略: 先训 Stage 1 拿到好的标量调度，再把它复制到 Stage 2，只放开 Δ_p, Δ_d。
预期: 如果稀释假说成立，warm-start 应当让 Stage 2 ≥ Stage 1。
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from socp.problem import generate_socp_data
from socp.classical import estimate_op_norm
from socp.pdhg_cp import LearnedPDHGCP, supervised_mse
from socp.pdhg_cp_couple import LearnedPDHGCPCouple, supervised_mse_with_coupling


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


def train_loop(model, ds, optimizer, num_epochs, batch=32, lam_couple=0.0,
               is_stage2=False):
    n = ds['A'].shape[0]
    hist = []
    for ep in range(num_epochs):
        perm = torch.randperm(n); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            x_hat, _ = model(ds['A'][idx], ds['b'][idx], ds['c'][idx])
            if is_stage2:
                loss, mse, reg = supervised_mse_with_coupling(
                    x_hat, ds['x_star'][idx], model, lam_couple=lam_couple)
            else:
                loss = supervised_mse(x_hat, ds['x_star'][idx])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            tot += float(loss) * idx.numel()
        hist.append(tot/n)
    return hist


def main():
    cfg = {'block_sizes': [5]*6 + [1]*4, 'm': 20}
    K = 20
    train_ds = build_dataset(cfg, 256, 0)
    test_ds = build_dataset(cfg, 50, 10000)
    L = train_ds['op_norm_mean']
    print(f"medium, K={K}; ||A|| ≈ {L:.3f}")

    out = {}

    # ---- (1) Stage 1 baseline ----
    print("\n[1] Stage 1 train (80 epoch, lr=5e-3) ...")
    torch.manual_seed(0); np.random.seed(0)
    m1 = LearnedPDHGCP(K=K, init_eta=0.9/L, init_tau=0.9/L,
                       init_beta_schedule='halpern', init_theta=0.999,
                       block_sizes=cfg['block_sizes'])
    opt1 = torch.optim.Adam(m1.parameters(), lr=5e-3)
    train_loop(m1, train_ds, opt1, num_epochs=80)
    ev1 = eval_model(m1, test_ds)
    print(f"    Stage 1                       median={ev1['median']:.2e}")
    out['stage1'] = ev1

    # ---- (2) Stage 2 vanilla (E1 复现) ----
    print("\n[2] Stage 2 vanilla (random init, joint train 80 epoch, lr=5e-3) ...")
    torch.manual_seed(0); np.random.seed(0)
    m2a = LearnedPDHGCPCouple(K=K, A_template=train_ds['A'][0],
                              block_sizes=cfg['block_sizes'],
                              init_eta=0.9/L, init_tau=0.9/L,
                              init_beta_schedule='halpern', init_theta=0.999,
                              coupling_mode='independent')
    opt2a = torch.optim.Adam(m2a.parameters(), lr=5e-3)
    train_loop(m2a, train_ds, opt2a, num_epochs=80, is_stage2=True)
    ev2a = eval_model(m2a, test_ds)
    print(f"    Stage 2 vanilla               median={ev2a['median']:.2e}  "
          f"||Δp||={float(torch.linalg.norm(m2a.delta_p)):.2e}")
    out['stage2_vanilla'] = ev2a

    # ---- (3) Stage 2 warm-start: copy stage1 scalars, lr_couple smaller ----
    print("\n[3] Stage 2 warm-start (copy S1 scalars, Δ at 0, joint train 80 epoch) ...")
    torch.manual_seed(0); np.random.seed(0)
    m2b = LearnedPDHGCPCouple(K=K, A_template=train_ds['A'][0],
                              block_sizes=cfg['block_sizes'],
                              init_eta=0.9/L, init_tau=0.9/L,
                              init_beta_schedule='halpern', init_theta=0.999,
                              coupling_mode='independent')
    # 复制 Stage 1 已训练好的标量
    with torch.no_grad():
        m2b.eta_raw.copy_(m1.eta_raw.detach())
        m2b.tau_raw.copy_(m1.tau_raw.detach())
        m2b.beta_raw.copy_(m1.beta_raw.detach())
        m2b.theta_raw.copy_(m1.theta_raw.detach())
    # 验证: Δ 仍为 0 => m2b 输出 = m1 输出
    with torch.no_grad():
        x_a, _ = m2b(test_ds['A'][:1], test_ds['b'][:1], test_ds['c'][:1])
        x_ref, _ = m1(test_ds['A'][:1], test_ds['b'][:1], test_ds['c'][:1])
        print(f"    sanity (Δ=0): warm-start vs Stage1 max diff = {(x_a-x_ref).abs().max().item():.2e}")
    # 用更小 lr 训 Δ，更小 lr 调整标量
    opt2b = torch.optim.Adam([
        {'params': [m2b.eta_raw, m2b.tau_raw, m2b.beta_raw, m2b.theta_raw], 'lr': 1e-3},
        {'params': [m2b.delta_p, m2b.delta_d], 'lr': 1e-3},
    ])
    train_loop(m2b, train_ds, opt2b, num_epochs=80, is_stage2=True)
    ev2b = eval_model(m2b, test_ds)
    print(f"    Stage 2 warm-start            median={ev2b['median']:.2e}  "
          f"||Δp||={float(torch.linalg.norm(m2b.delta_p)):.2e}")
    out['stage2_warmstart'] = ev2b

    # ---- (4) Stage 2 warm + 强 coupling reg ----
    print("\n[4] Stage 2 warm-start + λ=1e-1 ...")
    torch.manual_seed(0); np.random.seed(0)
    m2c = LearnedPDHGCPCouple(K=K, A_template=train_ds['A'][0],
                              block_sizes=cfg['block_sizes'],
                              init_eta=0.9/L, init_tau=0.9/L,
                              init_beta_schedule='halpern', init_theta=0.999,
                              coupling_mode='independent')
    with torch.no_grad():
        m2c.eta_raw.copy_(m1.eta_raw.detach())
        m2c.tau_raw.copy_(m1.tau_raw.detach())
        m2c.beta_raw.copy_(m1.beta_raw.detach())
        m2c.theta_raw.copy_(m1.theta_raw.detach())
    opt2c = torch.optim.Adam([
        {'params': [m2c.eta_raw, m2c.tau_raw, m2c.beta_raw, m2c.theta_raw], 'lr': 1e-3},
        {'params': [m2c.delta_p, m2c.delta_d], 'lr': 1e-3},
    ])
    train_loop(m2c, train_ds, opt2c, num_epochs=80, lam_couple=1e-1, is_stage2=True)
    ev2c = eval_model(m2c, test_ds)
    print(f"    Stage 2 warm + λ=1e-1         median={ev2c['median']:.2e}  "
          f"||Δp||={float(torch.linalg.norm(m2c.delta_p)):.2e}")
    out['stage2_warm_reg'] = ev2c

    # ---- (5) Stage 2 shared_symmetric warm (轻量耦合) ----
    print("\n[5] Stage 2 shared_symmetric warm-start ...")
    torch.manual_seed(0); np.random.seed(0)
    m2d = LearnedPDHGCPCouple(K=K, A_template=train_ds['A'][0],
                              block_sizes=cfg['block_sizes'],
                              init_eta=0.9/L, init_tau=0.9/L,
                              init_beta_schedule='halpern', init_theta=0.999,
                              coupling_mode='shared_symmetric')
    with torch.no_grad():
        m2d.eta_raw.copy_(m1.eta_raw.detach())
        m2d.tau_raw.copy_(m1.tau_raw.detach())
        m2d.beta_raw.copy_(m1.beta_raw.detach())
        m2d.theta_raw.copy_(m1.theta_raw.detach())
    opt2d = torch.optim.Adam([
        {'params': [m2d.eta_raw, m2d.tau_raw, m2d.beta_raw, m2d.theta_raw], 'lr': 1e-3},
        {'params': [m2d.delta_p], 'lr': 1e-3},
    ])
    train_loop(m2d, train_ds, opt2d, num_epochs=80, is_stage2=True)
    ev2d = eval_model(m2d, test_ds)
    print(f"    Stage 2 shared_sym warm       median={ev2d['median']:.2e}  "
          f"||Δp||={float(torch.linalg.norm(m2d.delta_p)):.2e}")
    out['stage2_warm_shared_sym'] = ev2d

    print("\n=========== Summary ===========")
    print(f"{'method':30s}  {'median':>10s}  {'vs S1':>8s}")
    for name, e in out.items():
        sp = ev1['median'] / max(e['median'], 1e-16)
        print(f"{name:30s}  {e['median']:>10.2e}  {sp:>7.2f}×")

    with open(os.path.join(os.path.dirname(__file__),
                           'report', 'socp_stage2_warmstart_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=float)


if __name__ == '__main__':
    main()
