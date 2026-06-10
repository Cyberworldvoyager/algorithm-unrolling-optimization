"""
visualize_2d.py — 用已训练的模型权重在 2D LASSO 问题上可视化优化轨迹

策略:
  - 从 models/ 加载已训练的维度无关模型 (DA-LISTA, RNN/LSTM/Transformer-LISTA)
    这些模型的参数与 n 无关，可在任意维度问题上推理
  - LISTA / LISTA-CP / LISTA-Momentum 的权重绑定到 n=200 的 A，跳过
  - 用 n=2 的 LASSO 问题作为可视化场景，画 T=10 步的优化轨迹

输出:
  report/optimization_trajectory_3d.png  (3D 曲面)
  report/optimization_trajectory_2d.png  (2D 等高线)

用法: python visualize_2d.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lasso import (ista, fista, LISTAMomentum, DimensionAgnosticLISTA,
                   RNNLISTA, LSTMLISTA, TransformerLISTA)


# ============================================================
# 2D LASSO 问题
# ============================================================

def make_2d_problem(seed=42):
    rng = np.random.RandomState(seed)
    m, n = 6, 2
    A = rng.randn(m, n).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.array([1.5, 0.0], dtype=np.float32)
    b = (A @ x_true).astype(np.float32)
    lam = 0.1
    return A, b, x_true, lam


def lasso_obj(A, b, lam, x):
    """LASSO 目标函数 (单点或网格)。"""
    if x.ndim == 1:
        return 0.5 * np.linalg.norm(A @ x - b) ** 2 + lam * np.sum(np.abs(x))
    # x: (..., 2)
    AX = x @ A.T
    residual = AX - b
    return 0.5 * np.sum(residual ** 2, axis=-1) + lam * np.sum(np.abs(x), axis=-1)


def lasso_obj_grid(A, b, lam, X1, X2):
    X = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    return lasso_obj(A, b, lam, X).reshape(X1.shape)


# ============================================================
# 轨迹收集
# ============================================================

def collect_ista(A, b, lam, T=10):
    L = np.linalg.norm(A.T @ A, ord=2); eta = 1.0 / L
    AtA, Atb = A.T @ A, A.T @ b
    x = np.zeros(2); traj = [x.copy()]
    for _ in range(T):
        z = x - eta * (AtA @ x - Atb)
        x = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        traj.append(x.copy())
    return np.array(traj)


def collect_fista(A, b, lam, T=10):
    L = np.linalg.norm(A.T @ A, ord=2); eta = 1.0 / L
    AtA, Atb = A.T @ A, A.T @ b
    x = np.zeros(2); y = x.copy(); t = 1.0; traj = [x.copy()]
    for _ in range(T):
        z = y - eta * (AtA @ y - Atb)
        x_new = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
        traj.append(x.copy())
    return np.array(traj)


def collect_da_lista(model, A, b, T=10):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n); x_prev = x.clone()
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_t, x, x_prev, A_t)
            x_prev, x = x, x_new
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_momentum(model, A, b, T=10):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n); x_prev = x.clone()
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_t, x, x_prev, A_t)
            x_prev, x = x, x_new
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_rnn(model, A, b, T=10):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n)
    h = torch.zeros(1, n, model.hidden_dim)
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((g - g.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h = model.rnn(g_norm, h.reshape(-1, model.hidden_dim))
            update = model.head(h).reshape(1, n) * g_std
            x = x + model.eta * update
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_lstm(model, A, b, T=10):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n)
    h = torch.zeros(1, n, model.hidden_dim)
    c = torch.zeros(1, n, model.hidden_dim)
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((g - g.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h_new, c_new = model.lstm(g_norm, (h.reshape(-1, model.hidden_dim),
                                                c.reshape(-1, model.hidden_dim)))
            h, c = h_new.reshape(1, n, -1), c_new.reshape(1, n, -1)
            update = model.head(h.reshape(-1, model.hidden_dim)).reshape(1, n) * g_std
            x = x + model.eta * update
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_transformer(model, A, b, T=10):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n)
    hs = []
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((g - g.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h_t = model.grad_to_hidden(g_norm.reshape(-1, 1)).reshape(1, n, -1)
            hs.append(h_t)
            # 暂存当前 x (最后一步会被 attention 更新覆盖)
            traj.append(x.squeeze().numpy().copy())
        # 最终 attention 更新
        seq = torch.cat(hs, dim=0).unsqueeze(0)
        B, T_len, N, H = seq.shape
        attn_in = seq.reshape(B * N, T_len, H)
        attn_out, _ = model.self_attn(attn_in, attn_in, attn_in)
        attn_last = attn_out[:, -1, :].reshape(B, N, H)
        update = model.head(attn_last.reshape(-1, H)).reshape(B, N) * g_std
        x = x + model.eta * update
        traj[-1] = x.squeeze().numpy().copy()
    return np.array(traj)


# ============================================================
# 可视化
# ============================================================

def plot_3d(A, b, lam, x_true, trajs, labels, colors, save):
    margin = 2.5; res = 80
    x1 = np.linspace(-margin, margin, res)
    x2 = np.linspace(-margin, margin, res)
    X1, X2 = np.meshgrid(x1, x2)
    Z = lasso_obj_grid(A, b, lam, X1, X2)
    f_opt = lasso_obj(A, b, lam, x_true)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.2, linewidth=0, antialiased=True)
    cz = Z.min() - 0.3
    ax.contourf(X1, X2, Z, levels=20, zdir='z', offset=cz, cmap='coolwarm', alpha=0.3)

    for traj, lab, col in zip(trajs, labels, colors):
        zs = np.array([lasso_obj(A, b, lam, p) for p in traj])
        ax.plot(traj[:, 0], traj[:, 1], zs, color=col, linewidth=2.5, label=lab,
                marker='o', markersize=4, zorder=5)
        ax.scatter(*traj[0], zs[0], color=col, s=80, marker='s',
                   edgecolors='black', linewidth=1, zorder=6)
        ax.scatter(*traj[-1], zs[-1], color=col, s=120, marker='*',
                   edgecolors='black', linewidth=1, zorder=6)

    ax.scatter(*x_true, f_opt, color='gold', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Optimal', zorder=7)
    ax.set_xlabel('$x_1$', fontsize=14); ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('Objective $f(x)$', fontsize=14)
    ax.set_title('Optimization Trajectories on 2D LASSO Landscape\n'
                 f'($n=2, m=6, \\lambda={lam}$, trained weights from n=200)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.view_init(elev=35, azim=-50); ax.set_zlim(cz, Z.max())
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  3D 轨迹图 → {save}")


def plot_2d(A, b, lam, x_true, trajs, labels, colors, save):
    margin = 2.5; res = 100
    x1 = np.linspace(-margin, margin, res)
    x2 = np.linspace(-margin, margin, res)
    X1, X2 = np.meshgrid(x1, x2)
    Z = lasso_obj_grid(A, b, lam, X1, X2)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X1, X2, Z, levels=30, cmap='coolwarm', alpha=0.6)
    ax.contour(X1, X2, Z, levels=30, colors='white', linewidths=0.3, alpha=0.4)
    plt.colorbar(cs, ax=ax, label='Objective $f(x)$')

    for traj, lab, col in zip(trajs, labels, colors):
        ax.plot(traj[:, 0], traj[:, 1], '-o', color=col, linewidth=2.5,
                markersize=5, label=lab, zorder=5)
        ax.plot(*traj[0], 's', color=col, markersize=8,
                markeredgecolor='black', markeredgewidth=1, zorder=6)
        ax.plot(*traj[-1], '*', color=col, markersize=12,
                markeredgecolor='black', markeredgewidth=1, zorder=6)

    ax.plot(*x_true, '*', color='gold', markersize=18,
            markeredgecolor='black', markeredgewidth=2, label='Optimal', zorder=7)
    ax.set_xlabel('$x_1$', fontsize=14); ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Optimization Trajectories on 2D LASSO\n'
                 f'($n=2, m=6, \\lambda={lam}$, T=10 steps, loaded trained weights)', fontsize=13)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  2D 等高线图 → {save}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=== 2D LASSO 优化轨迹可视化 (加载已训练权重) ===\n")

    A, b, x_true, lam = make_2d_problem(seed=42)
    T = 10
    print(f"问题: n=2, m=6, k=1, λ={lam}, x_true={x_true}\n")

    models_dir = 'models'
    if not os.path.isdir(models_dir):
        print(f"错误: 未找到 {models_dir}/ 目录，请先运行 run_all_experiments.py")
        return

    # 经典方法 (无需模型权重)
    trajs = [collect_ista(A, b, lam, T), collect_fista(A, b, lam, T)]
    labels = ['ISTA', 'FISTA']
    colors_list = ['#1f77b4', '#ff7f0e']

    # 维度无关模型 (可从 models/ 加载)
    loadable = [
        ('da_lista',       DimensionAgnosticLISTA, {'T': T},               collect_da_lista,  '#2ca02c'),
        ('rnn_lista',      RNNLISTA,               {'T': T},               collect_rnn,       '#d62728'),
        ('lstm_lista',     LSTMLISTA,              {'T': T},               collect_lstm,      '#9467bd'),
        ('transformer_lista', TransformerLISTA,    {'T': T},               collect_transformer,'#8c564b'),
    ]

    for name, cls, kwargs, collector, color in loadable:
        path = f'{models_dir}/{name}_noiseless.pt'
        if not os.path.exists(path):
            print(f"  跳过 {name}: 未找到权重文件 {path}")
            continue
        model = cls(**kwargs)
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        traj = collector(model, A, b, T)
        err = np.linalg.norm(x_true - traj[-1]) / (np.linalg.norm(x_true) + 1e-10)
        print(f"  {name:18s}: 终点=[{traj[-1][0]:.3f}, {traj[-1][1]:.3f}], 相对误差={err:.4f}")
        trajs.append(traj)
        labels.append(name.replace('_lista', '-LISTA').replace('_', '-').title())
        colors_list.append(color)

    # LISTA-Momentum 也跳过 (dW 是 n×n 绑定到 n=200)
    print("\n  跳过 lista / lista_cp / lista_momentum: 权重绑定到 n=200")

    # 绘图
    print("\n生成可视化...")
    os.makedirs('report', exist_ok=True)
    plot_3d(A, b, lam, x_true, trajs, labels, colors_list,
            'report/optimization_trajectory_3d.png')
    plot_2d(A, b, lam, x_true, trajs, labels, colors_list,
            'report/optimization_trajectory_2d.png')

    # 打印总结
    print("\n各模型终点误差:")
    for lab, traj in zip(labels, trajs):
        err = np.linalg.norm(x_true - traj[-1]) / (np.linalg.norm(x_true) + 1e-10)
        print(f"  {lab:18s}: {err:.4f}")
    print("\n完成!")


if __name__ == '__main__':
    main()
