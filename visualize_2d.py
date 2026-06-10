"""
visualize_2d.py — 用 PCA 将 n=200 的优化轨迹投影到 2D 并可视化

策略:
  1. 加载已训练的全部 9 个模型 (含 LISTA/CP/Momentum)
  2. 在一个测试样本 (A, b) 上运行各模型 T=10 步，记录每步的 x 估计
  3. 将所有轨迹 (含 ISTA/FISTA) 用 PCA 投影到 2D
  4. 在 PCA 子空间上画目标函数等高线 + 各模型轨迹
  5. 测量并输出各模型推理时间

输出:
  report/optimization_trajectory_3d.png
  report/optimization_trajectory_2d.png
  终端打印推理时间对比

用法: python visualize_2d.py
"""

import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lasso import (ista, fista, LISTA, LISTACP, LISTAMomentum,
                   DimensionAgnosticLISTA, RNNLISTA, LSTMLISTA, TransformerLISTA)


# ============================================================
# 问题设置: 使用与主实验一致的 n=200 问题
# ============================================================

M, N, K, T = 100, 200, 5, 10


def make_problem(seed=42):
    rng = np.random.RandomState(seed)
    A = rng.randn(M, N).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.zeros(N, dtype=np.float32)
    support = rng.choice(N, K, replace=False)
    x_true[support] = rng.randn(K).astype(np.float32)
    b = (A @ x_true).astype(np.float32)
    lam = 0.01 * np.max(np.abs(A.T @ b))
    return A, b, x_true, lam


def lasso_obj(A, b, lam, x):
    """LASSO 目标函数 (单点或批量)。"""
    if x.ndim == 1:
        return 0.5 * np.linalg.norm(A @ x - b) ** 2 + lam * np.sum(np.abs(x))
    AX = x @ A.T
    residual = AX - b
    return 0.5 * np.sum(residual ** 2, axis=-1) + lam * np.sum(np.abs(x), axis=-1)


# ============================================================
# 轨迹收集: 每个模型返回 T+1 个 n 维向量
# ============================================================

def collect_ista(A, b, lam, T):
    L = np.linalg.norm(A.T @ A, ord=2); eta = 1.0 / L
    AtA, Atb = A.T @ A, A.T @ b
    x = np.zeros(N); traj = [x.copy()]
    for _ in range(T):
        z = x - eta * (AtA @ x - Atb)
        x = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        traj.append(x.copy())
    return np.array(traj)


def collect_fista(A, b, lam, T):
    L = np.linalg.norm(A.T @ A, ord=2); eta = 1.0 / L
    AtA, Atb = A.T @ A, A.T @ b
    x = np.zeros(N); y = x.copy(); t = 1.0; traj = [x.copy()]
    for _ in range(T):
        z = y - eta * (AtA @ y - Atb)
        x_new = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x, t = x_new, t_new
        traj.append(x.copy())
    return np.array(traj)


def collect_lista(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N)
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x = layer(b_t, x)
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_cp(model, A, b, T):
    return collect_lista(model, A, b, T)


def collect_momentum(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N); x_prev = x.clone()
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_t, x, x_prev, A_t)
            x_prev, x = x, x_new
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_da(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N); x_prev = x.clone()
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_t, x, x_prev, A_t)
            x_prev, x = x, x_new
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_rnn(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N)
    h = torch.zeros(1, N, model.hidden_dim)
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((g - g.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h = model.rnn(g_norm, h.reshape(-1, model.hidden_dim))
            update = model.head(h).reshape(1, N) * g_std
            x = x + model.eta * update
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_lstm(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N)
    h = torch.zeros(1, N, model.hidden_dim)
    c = torch.zeros(1, N, model.hidden_dim)
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = ((g - g.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
            h_new, c_new = model.lstm(g_norm, (h.reshape(-1, model.hidden_dim),
                                                c.reshape(-1, model.hidden_dim)))
            h, c = h_new.reshape(1, N, -1), c_new.reshape(1, N, -1)
            update = model.head(h.reshape(-1, model.hidden_dim)).reshape(1, N) * g_std
            x = x + model.eta * update
            traj.append(x.squeeze().numpy().copy())
    return np.array(traj)


def collect_transformer(model, A, b, T):
    model.eval()
    A_t = torch.from_numpy(A).unsqueeze(0)
    b_t = torch.from_numpy(b).unsqueeze(0)
    x = torch.zeros(1, N)
    hs = []; stds = []
    traj = [x.squeeze().numpy().copy()]
    with torch.no_grad():
        for _ in range(T):
            Ax = torch.bmm(A_t, x.unsqueeze(-1)).squeeze(-1)
            g = torch.bmm(A_t.transpose(1, 2), (Ax - b_t).unsqueeze(-1)).squeeze(-1)
            g_std = g.std(dim=-1, keepdim=True) + 1e-6
            g_norm = (g - g.mean(dim=-1, keepdim=True)) / g_std
            h_t = model.grad_to_hidden(g_norm.reshape(-1, 1)).reshape(1, N, -1)
            hs.append(h_t); stds.append(g_std)
            traj.append(x.squeeze().numpy().copy())
        # attention 聚合
        seq = torch.cat(hs, dim=0).unsqueeze(0)
        B_b, T_len, Nn, H = seq.shape
        attn_in = seq.permute(0, 2, 1, 3).reshape(B_b * Nn, T_len, H)
        attn_out, _ = model.self_attn(attn_in, attn_in, attn_in)
        attn_last = attn_out[:, -1, :].reshape(B_b, Nn, H)
        update = model.head(attn_last.reshape(-1, H)).reshape(B_b, Nn) * stds[-1]
        x = x + model.eta * update
        traj[-1] = x.squeeze().numpy().copy()
    return np.array(traj)


# ============================================================
# 推理时间测量
# ============================================================

@torch.no_grad()
def measure_inference_time(model, A_np, B_np, needs_A, n_runs=50):
    """测量模型在 n_runs 次前向传播上的平均推理时间 (ms)。"""
    model.eval()
    A = torch.from_numpy(A_np).to('cpu')
    B = torch.from_numpy(B_np).to('cpu')
    # 预热
    for _ in range(5):
        model(B, A) if needs_A else model(B)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(B, A) if needs_A else model(B)
    elapsed = (time.perf_counter() - t0) / n_runs * 1000  # ms
    return elapsed


def measure_classical_time(A_np, B_np, T, n_runs=20):
    """测量经典方法的推理时间。"""
    lam = 0.01 * np.max(np.abs(A_np.T @ B_np[0]))
    # 预热
    for i in range(3):
        ista(A_np, B_np[i], lam, max_iter=T)
        fista(A_np, B_np[i], lam, max_iter=T)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(len(B_np)):
            ista(A_np, B_np[i], lam, max_iter=T)
    t_ista = (time.perf_counter() - t0) / n_runs * 1000
    t0 = time.perf_counter()
    for _ in range(n_runs):
        for i in range(len(B_np)):
            fista(A_np, B_np[i], lam, max_iter=T)
    t_fista = (time.perf_counter() - t0) / n_runs * 1000
    return t_ista, t_fista


# ============================================================
# PCA 可视化
# ============================================================

def plot_pca_3d(x_true, all_trajs, all_labels, all_colors, A, b, lam, save):
    """PCA 投影到 2D + 目标函数值作 z 轴，画 3D 图。"""
    # 拼接所有轨迹点做 PCA
    all_pts = np.vstack(all_trajs)
    pca = PCA(n_components=2)
    pca.fit(all_pts)

    # 用 PCA 坐标表示各轨迹
    trajs_2d = [pca.transform(traj) for traj in all_trajs]
    x_true_2d = pca.transform(x_true.reshape(1, -1))[0]

    # 网格 (在 PCA 空间)
    all_2d = np.vstack(trajs_2d)
    margin_x = max(abs(all_2d[:, 0].min()), abs(all_2d[:, 0].max())) * 1.5
    margin_y = max(abs(all_2d[:, 1].min()), abs(all_2d[:, 1].max())) * 1.5
    margin = max(margin_x, margin_y, 2.0)
    res = 60
    gx = np.linspace(x_true_2d[0] - margin, x_true_2d[0] + margin, res)
    gy = np.linspace(x_true_2d[1] - margin, x_true_2d[1] + margin, res)
    GX, GY = np.meshgrid(gx, gy)

    # 网格点逆投影到原始空间算目标函数
    grid_2d = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    grid_nd = pca.inverse_transform(grid_2d)
    Z = lasso_obj(A, b, lam, grid_nd).reshape(res, res)
    f_opt = lasso_obj(A, b, lam, x_true)

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(GX, GY, Z, cmap='coolwarm', alpha=0.2, linewidth=0, antialiased=True)
    cz = Z.min() - 0.5
    ax.contourf(GX, GY, Z, levels=20, zdir='z', offset=cz, cmap='coolwarm', alpha=0.3)

    for traj_2d, lab, col in zip(trajs_2d, all_labels, all_colors):
        traj_nd = [pca.inverse_transform(p.reshape(1, -1))[0] for p in traj_2d]
        zs = np.array([lasso_obj(A, b, lam, p) for p in traj_nd])
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], zs, color=col, linewidth=2.5, label=lab,
                marker='o', markersize=4, zorder=5)
        ax.scatter(*traj_2d[0], zs[0], color=col, s=80, marker='s',
                   edgecolors='black', linewidth=1, zorder=6)
        ax.scatter(*traj_2d[-1], zs[-1], color=col, s=120, marker='*',
                   edgecolors='black', linewidth=1, zorder=6)

    ax.scatter(*x_true_2d, f_opt, color='gold', s=200, marker='*',
               edgecolors='black', linewidth=2, label='Optimal', zorder=7)
    ax.set_xlabel('PC1', fontsize=13); ax.set_ylabel('PC2', fontsize=13)
    ax.set_zlabel('Objective $f(x)$', fontsize=13)
    ax.set_title('Optimization Trajectories (PCA projection, n=200)\n'
                 f'T={T}, m={M}, k={K}, \\lambda={lam:.4f}', fontsize=14)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.view_init(elev=35, azim=-50); ax.set_zlim(cz, Z.max())
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  3D PCA 轨迹图 → {save}")


def plot_pca_2d(x_true, all_trajs, all_labels, all_colors, A, b, lam, save):
    """PCA 投影到 2D，画等高线 + 轨迹。"""
    all_pts = np.vstack(all_trajs)
    pca = PCA(n_components=2)
    pca.fit(all_pts)

    trajs_2d = [pca.transform(traj) for traj in all_trajs]
    x_true_2d = pca.transform(x_true.reshape(1, -1))[0]

    all_2d = np.vstack(trajs_2d)
    margin_x = max(abs(all_2d[:, 0].min()), abs(all_2d[:, 0].max())) * 1.5
    margin_y = max(abs(all_2d[:, 1].min()), abs(all_2d[:, 1].max())) * 1.5
    margin = max(margin_x, margin_y, 2.0)
    res = 80
    gx = np.linspace(x_true_2d[0] - margin, x_true_2d[0] + margin, res)
    gy = np.linspace(x_true_2d[1] - margin, x_true_2d[1] + margin, res)
    GX, GY = np.meshgrid(gx, gy)

    grid_2d = np.stack([GX.ravel(), GY.ravel()], axis=-1)
    grid_nd = pca.inverse_transform(grid_2d)
    Z = lasso_obj(A, b, lam, grid_nd).reshape(res, res)

    fig, ax = plt.subplots(figsize=(11, 9))
    cs = ax.contourf(GX, GY, Z, levels=30, cmap='coolwarm', alpha=0.6)
    ax.contour(GX, GY, Z, levels=30, colors='white', linewidths=0.3, alpha=0.4)
    plt.colorbar(cs, ax=ax, label='Objective $f(x)$')

    for traj_2d, lab, col in zip(trajs_2d, all_labels, all_colors):
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-o', color=col, linewidth=2.5,
                markersize=5, label=lab, zorder=5)
        ax.plot(*traj_2d[0], 's', color=col, markersize=8,
                markeredgecolor='black', markeredgewidth=1, zorder=6)
        ax.plot(*traj_2d[-1], '*', color=col, markersize=12,
                markeredgecolor='black', markeredgewidth=1, zorder=6)

    ax.plot(*x_true_2d, '*', color='gold', markersize=18,
            markeredgecolor='black', markeredgewidth=2, label='Optimal', zorder=7)
    ax.set_xlabel('PC1', fontsize=14); ax.set_ylabel('PC2', fontsize=14)
    ax.set_title('Optimization Trajectories (PCA projection, n=200)\n'
                 f'T={T}, m={M}, k={K}, all models with trained weights', fontsize=13)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout(); plt.savefig(save, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  2D PCA 等高线图 → {save}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=== 优化轨迹 PCA 可视化 (全部模型, n=200) ===\n")

    A, b, x_true, lam = make_problem(seed=42)
    print(f"问题: n={N}, m={M}, k={K}, λ={lam:.4f}\n")

    # 构建全部模型并加载权重
    eta0 = 1.0 / float(np.linalg.norm(A.T @ A, ord=2))
    A_tensor = torch.from_numpy(A)

    model_info = [
        # (name, model, needs_A, collector, label, color)
        ('lista',          LISTA(M, N, T, init_threshold=1e-3, A_init=A_tensor, init_eta=eta0),
                           False, collect_lista,       'LISTA',          '#1f77b4'),
        ('lista_cp',       LISTACP(A_tensor, T, init_eta=eta0, init_threshold=1e-3),
                           False, collect_cp,          'LISTA-CP',       '#ff7f0e'),
        ('lista_momentum', LISTAMomentum(M, N, T, init_eta=eta0, init_threshold=1e-3),
                           True,  collect_momentum,    'LISTA-Momentum', '#2ca02c'),
        ('da_lista',       DimensionAgnosticLISTA(T),
                           True,  collect_da,          'DA-LISTA',       '#d62728'),
        ('rnn_lista',      RNNLISTA(T),
                           True,  collect_rnn,         'RNN-LISTA',      '#9467bd'),
        ('lstm_lista',     LSTMLISTA(T),
                           True,  collect_lstm,        'LSTM-LISTA',     '#8c564b'),
        ('transformer_lista', TransformerLISTA(T),
                           True,  collect_transformer, 'Transformer',    '#e377c2'),
    ]

    models_dir = 'models'
    all_trajs = []
    all_labels = []
    all_colors = []
    inf_times = {}

    # ISTA / FISTA
    print("收集轨迹...")
    traj_ista = collect_ista(A, b, lam, T)
    all_trajs.append(traj_ista); all_labels.append('ISTA'); all_colors.append('#7f7f7f')
    traj_fista = collect_fista(A, b, lam, T)
    all_trajs.append(traj_fista); all_labels.append('FISTA'); all_colors.append('#bcbd22')

    # 学习型模型
    for name, model, needs_A, collector, label, color in model_info:
        path = f'{models_dir}/{name}_noiseless.pt'
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        else:
            print(f"  警告: {path} 未找到，使用随机初始化")
        traj = collector(model, A, b, T)
        err = float(np.linalg.norm(x_true - traj[-1]) / (np.linalg.norm(x_true) + 1e-10))
        print(f"  {label:18s}: 终点相对误差={err:.4f}")
        all_trajs.append(traj)
        all_labels.append(label)
        all_colors.append(color)

    # 推理时间测量
    print("\n测量推理时间...")
    B_batch = np.random.randn(32, M).astype(np.float32)
    t_ista_ms, t_fista_ms = measure_classical_time(A, B_batch, T, n_runs=20)
    inf_times['ista'] = round(t_ista_ms, 2)
    inf_times['fista'] = round(t_fista_ms, 2)
    print(f"  {'ISTA':18s}: {t_ista_ms:.2f} ms/batch (32 samples)")
    print(f"  {'FISTA':18s}: {t_fista_ms:.2f} ms/batch (32 samples)")

    for name, model, needs_A, _, label, _ in model_info:
        t_ms = measure_inference_time(model, A, B_batch, needs_A, n_runs=50)
        inf_times[name] = round(t_ms, 2)
        print(f"  {label:18s}: {t_ms:.2f} ms/batch (32 samples)")

    # 绘图
    print("\n生成可视化...")
    os.makedirs('report', exist_ok=True)
    plot_pca_3d(x_true, all_trajs, all_labels, all_colors, A, b, lam,
                'report/optimization_trajectory_3d.png')
    plot_pca_2d(x_true, all_trajs, all_labels, all_colors, A, b, lam,
                'report/optimization_trajectory_2d.png')

    # 打印总结
    print("\n各模型终点误差:")
    for lab, traj in zip(all_labels, all_trajs):
        err = float(np.linalg.norm(x_true - traj[-1]) / (np.linalg.norm(x_true) + 1e-10))
        print(f"  {lab:18s}: {err:.4f}")

    print("\n推理时间 (ms/batch, 32 samples):")
    for name, t in inf_times.items():
        print(f"  {name:18s}: {t:.2f} ms")

    print("\n完成!")
    return inf_times


if __name__ == '__main__':
    main()
