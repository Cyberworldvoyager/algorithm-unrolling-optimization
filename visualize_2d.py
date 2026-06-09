"""
visualize_2d.py — 2D LASSO 问题下各模型的优化轨迹可视化

以 n=2 的 LASSO 问题为例，画出:
1. 目标函数 f(x) = ½‖Ax-b‖² + λ‖x‖₁ 的 3D 曲面
2. 各模型在 T=10 步迭代中的 x 估计轨迹 (叠在曲面上)
3. 2D 等高线投影图 (底部)

用法:
    python visualize_2d.py
输出:
    report/optimization_trajectory_3d.png
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from common.metrics import relative_error
from lasso import (ista, fista, LISTA, LISTACP, LISTAMomentum,
                   DimensionAgnosticLISTA, RNNLISTA, LSTMLISTA, TransformerLISTA)


# ============================================================
# 问题设置: n=2, m=6, k=1 (单稀疏分量)
# ============================================================

def make_2d_problem(seed=42):
    """构造一个有代表性的 2D LASSO 问题。"""
    rng = np.random.RandomState(seed)
    m, n = 6, 2
    A = rng.randn(m, n).astype(np.float32)
    A /= np.linalg.norm(A, axis=0, keepdims=True)
    x_true = np.array([1.5, 0.0], dtype=np.float32)  # 稀疏: 只有第一个分量非零
    b = (A @ x_true).astype(np.float32)
    lam = 0.1
    return A, b, x_true, lam


def lasso_objective(A, b, lam, x1_grid, x2_grid):
    """计算 LASSO 目标函数值 (向量化, 用于画曲面)。"""
    # x1_grid, x2_grid: shape (N, N)
    N = x1_grid.shape[0]
    X = np.stack([x1_grid.ravel(), x2_grid.ravel()], axis=-1)  # (N*N, 2)
    AX = X @ A.T  # (N*N, m)
    residual = AX - b[None, :]  # (N*N, m)
    data_term = 0.5 * np.sum(residual ** 2, axis=-1)  # (N*N,)
    reg_term = lam * np.sum(np.abs(X), axis=-1)  # (N*N,)
    return (data_term + reg_term).reshape(N, N)


# ============================================================
# 轨迹收集: 每个模型返回 T 步的 x 估计
# ============================================================

def collect_ista_traj(A, b, lam, T=10):
    """ISTA 轨迹。"""
    m, n = A.shape
    L = np.linalg.norm(A.T @ A, ord=2)
    eta = 1.0 / L
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n)
    traj = [x.copy()]
    for _ in range(T):
        grad = AtA @ x - Atb
        z = x - eta * grad
        x = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        traj.append(x.copy())
    return np.array(traj)


def collect_fista_traj(A, b, lam, T=10):
    """FISTA 轨迹。"""
    m, n = A.shape
    L = np.linalg.norm(A.T @ A, ord=2)
    eta = 1.0 / L
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n)
    y = x.copy()
    t = 1.0
    traj = [x.copy()]
    for _ in range(T):
        grad = AtA @ y - Atb
        z = y - eta * grad
        x_new = np.sign(z) * np.maximum(np.abs(z) - eta * lam, 0)
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        y = x_new + ((t - 1) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        traj.append(x.copy())
    return np.array(traj)


def collect_lista_traj(model, A, b, T=10):
    """收集 LISTA/CP 模型的逐层输出。"""
    model.eval()
    A_tensor = torch.from_numpy(A).unsqueeze(0)  # (1, m, n)
    b_tensor = torch.from_numpy(b).unsqueeze(0)  # (1, m)
    n = A.shape[1]
    x = torch.zeros(1, n)
    traj = [x.squeeze(0).detach().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x = layer(b_tensor, x)
            traj.append(x.squeeze(0).detach().numpy().copy())
    return np.array(traj)


def collect_momentum_traj(model, A, b, T=10):
    """收集 LISTA-Momentum 的逐层输出。"""
    model.eval()
    A_tensor = torch.from_numpy(A).unsqueeze(0)
    b_tensor = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n)
    x_prev = x.clone()
    traj = [x.squeeze(0).detach().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_tensor, x, x_prev, A_tensor)
            x_prev = x
            x = x_new
            traj.append(x.squeeze(0).detach().numpy().copy())
    return np.array(traj)


def collect_da_traj(model, A, b, T=10):
    """收集 DA-LISTA 的逐层输出。"""
    model.eval()
    A_tensor = torch.from_numpy(A).unsqueeze(0)
    b_tensor = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]
    x = torch.zeros(1, n)
    x_prev = x.clone()
    traj = [x.squeeze(0).detach().numpy().copy()]
    with torch.no_grad():
        for layer in model.layers:
            x_new = layer(b_tensor, x, x_prev, A_tensor)
            x_prev = x
            x = x_new
            traj.append(x.squeeze(0).detach().numpy().copy())
    return np.array(traj)


def collect_sequence_traj(model, A, b, T=10):
    """收集 RNN/LSTM/Transformer 的逐步输出 (需要内部 hook)。"""
    model.eval()
    A_tensor = torch.from_numpy(A).unsqueeze(0)
    b_tensor = torch.from_numpy(b).unsqueeze(0)
    n = A.shape[1]

    # 手动展开前向传播以捕获每步输出
    x = torch.zeros(1, n, device='cpu')
    traj = [x.squeeze(0).detach().numpy().copy()]

    if hasattr(model, 'rnn'):  # RNN-LISTA
        h = torch.zeros(1, n, model.hidden_dim, device='cpu')
        with torch.no_grad():
            for _ in range(T):
                Ax = torch.bmm(A_tensor, x.unsqueeze(-1)).squeeze(-1)
                grad = torch.bmm(A_tensor.transpose(1, 2), (Ax - b_tensor).unsqueeze(-1)).squeeze(-1)
                g_std = grad.std(dim=-1, keepdim=True) + 1e-6
                g_norm = ((grad - grad.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
                h = model.rnn(g_norm, h.reshape(-1, model.hidden_dim))
                update = model.head(h).reshape(1, n) * g_std
                x = x + model.eta * update
                traj.append(x.squeeze(0).detach().numpy().copy())
    elif hasattr(model, 'lstm'):  # LSTM-LISTA
        h = torch.zeros(1, n, model.hidden_dim, device='cpu')
        c = torch.zeros(1, n, model.hidden_dim, device='cpu')
        with torch.no_grad():
            for _ in range(T):
                Ax = torch.bmm(A_tensor, x.unsqueeze(-1)).squeeze(-1)
                grad = torch.bmm(A_tensor.transpose(1, 2), (Ax - b_tensor).unsqueeze(-1)).squeeze(-1)
                g_std = grad.std(dim=-1, keepdim=True) + 1e-6
                g_norm = ((grad - grad.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
                h_new, c_new = model.lstm(g_norm, (h.reshape(-1, model.hidden_dim),
                                                    c.reshape(-1, model.hidden_dim)))
                h, c = h_new.reshape(1, n, -1), c_new.reshape(1, n, -1)
                update = model.head(h.reshape(-1, model.hidden_dim)).reshape(1, n) * g_std
                x = x + model.eta * update
                traj.append(x.squeeze(0).detach().numpy().copy())
    elif hasattr(model, 'self_attn'):  # Transformer-LISTA
        hs = []
        with torch.no_grad():
            for _ in range(T):
                Ax = torch.bmm(A_tensor, x.unsqueeze(-1)).squeeze(-1)
                grad = torch.bmm(A_tensor.transpose(1, 2), (Ax - b_tensor).unsqueeze(-1)).squeeze(-1)
                g_std = grad.std(dim=-1, keepdim=True) + 1e-6
                g_norm = ((grad - grad.mean(dim=-1, keepdim=True)) / g_std).reshape(-1, 1)
                h_t = model.mlp_in(g_norm).reshape(1, n, -1)
                hs.append(h_t)
                traj.append(x.squeeze(0).detach().numpy().copy())
            # 最后用 attention 更新
            seq = torch.cat(hs, dim=0).unsqueeze(0)  # (1, T, n, hidden)
            B, T_len, N, H = seq.shape
            attn_in = seq.reshape(B * N, T_len, H)
            attn_out, _ = model.self_attn(attn_in, attn_in, attn_in)
            attn_last = attn_out[:, -1, :].reshape(B, N, H)
            update = model.head(attn_last.reshape(-1, H)).reshape(B, N) * g_std
            x = x + model.eta * update
            traj[-1] = x.squeeze(0).detach().numpy().copy()  # 替换最后一步
    return np.array(traj)


# ============================================================
# 可视化
# ============================================================

def plot_3d_trajectory(A, b, lam, x_true, trajectories, labels, colors,
                       save='report/optimization_trajectory_3d.png'):
    """画 3D 优化轨迹图。"""
    # 网格范围: 以 x_true 为中心, 覆盖原点和 x_true
    margin = 2.5
    grid_res = 80
    x1_range = np.linspace(-margin, margin, grid_res)
    x2_range = np.linspace(-margin, margin, grid_res)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = lasso_objective(A, b, lam, X1, X2)

    # 最优值 (用于归一化 z 轴)
    f_opt = lasso_objective(A, b, lam, x_true[0:1], x_true[1:2])[0, 0]

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 画曲面 (半透明)
    surf = ax.plot_surface(X1, X2, Z, cmap='coolwarm', alpha=0.25,
                           linewidth=0, antialiased=True, zorder=0)

    # 画等高线投影 (底部)
    contour_offset = Z.min() - 0.3
    ax.contourf(X1, X2, Z, levels=20, zdir='z', offset=contour_offset,
                cmap='coolwarm', alpha=0.3)

    # 画各模型轨迹
    for traj, label, color in zip(trajectories, labels, colors):
        xs, ys = traj[:, 0], traj[:, 1]
        zs = np.array([lasso_objective(A, b, lam, np.array([x1]), np.array([x2]))[0, 0]
                        for x1, x2 in zip(xs, ys)])
        # 轨迹线
        ax.plot(xs, ys, zs, color=color, linewidth=2.5, label=label,
                marker='o', markersize=4, zorder=5)
        # 起点 (大圆)
        ax.scatter([xs[0]], [ys[0]], [zs[0]], color=color, s=80, marker='s',
                   edgecolors='black', linewidth=1, zorder=6)
        # 终点 (大星)
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color=color, s=120, marker='*',
                   edgecolors='black', linewidth=1, zorder=6)

    # 标注最优点
    ax.scatter([x_true[0]], [x_true[1]], [f_opt], color='gold', s=200,
               marker='*', edgecolors='black', linewidth=2, label='Optimal', zorder=7)

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_zlabel('Objective $f(x)$', fontsize=14)
    ax.set_title('Optimization Trajectories on 2D LASSO Landscape\n'
                 f'($n=2, m=6, k=1, \\lambda={lam}$)', fontsize=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.view_init(elev=35, azim=-50)
    ax.set_zlim(contour_offset, Z.max())

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  3D 轨迹图已保存到 {save}")


def plot_2d_contour(A, b, lam, x_true, trajectories, labels, colors,
                    save='report/optimization_trajectory_2d.png'):
    """画 2D 等高线投影图。"""
    margin = 2.5
    grid_res = 100
    x1_range = np.linspace(-margin, margin, grid_res)
    x2_range = np.linspace(-margin, margin, grid_res)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = lasso_objective(A, b, lam, X1, X2)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X1, X2, Z, levels=30, cmap='coolwarm', alpha=0.6)
    ax.contour(X1, X2, Z, levels=30, colors='white', linewidths=0.3, alpha=0.4)
    plt.colorbar(cs, ax=ax, label='Objective $f(x)$')

    for traj, label, color in zip(trajectories, labels, colors):
        xs, ys = traj[:, 0], traj[:, 1]
        ax.plot(xs, ys, '-o', color=color, linewidth=2.5, markersize=5,
                label=label, zorder=5)
        ax.plot(xs[0], ys[0], 's', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1, zorder=6)
        ax.plot(xs[-1], ys[-1], '*', color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=1, zorder=6)

    ax.plot(x_true[0], x_true[1], '*', color='gold', markersize=18,
            markeredgecolor='black', markeredgewidth=2, label='Optimal', zorder=7)

    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    ax.set_title('Optimization Trajectories on 2D LASSO Landscape\n'
                 f'($n=2, m=6, k=1, \\lambda={lam}$, T=10 steps)', fontsize=14)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  2D 等高线图已保存到 {save}")


# ============================================================
# 主流程
# ============================================================

def main():
    print("=== 2D LASSO 优化轨迹可视化 ===\n")

    A, b, x_true, lam = make_2d_problem(seed=42)
    T = 10
    print(f"问题: n={A.shape[1]}, m={A.shape[0]}, k=1, λ={lam}")
    print(f"x_true = {x_true}")

    # 训练所有学习型模型 (在 2D 问题上重新训练)
    print("\n训练模型...")
    A_tensor = torch.from_numpy(A)
    eta0 = 1.0 / float(np.linalg.norm(A.T @ A, ord=2))
    th0 = 1e-3

    # 生成 2D 训练数据 (固定字典 A)
    rng = np.random.RandomState(42)
    train_B, train_X = [], []
    for _ in range(500):
        x = np.zeros(2)
        support = rng.choice(2, 1, replace=False)
        x[support] = rng.randn(1) * 1.5
        train_B.append((A @ x).astype(np.float32))
        train_X.append(x.astype(np.float32))
    train_B = np.array(train_B)
    train_X = np.array(train_X)

    device = 'cpu'  # 2D 问题很小, 用 CPU 即可

    def train_2d(model, needs_A=False, epochs=200):
        model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = torch.nn.MSELoss()
        B_t = torch.from_numpy(train_B).to(device)
        X_t = torch.from_numpy(train_X).to(device)
        A_t = torch.from_numpy(A).to(device)
        for ep in range(epochs):
            perm = torch.randperm(500, device=device)
            for i in range(0, 500, 64):
                idx = perm[i:i+64]
                pred = model(B_t[idx], A_t) if needs_A else model(B_t[idx])
                loss = crit(pred, X_t[idx])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        return model

    models = {}

    # LISTA (ISTA-init)
    m_lista = LISTA(6, 2, T, init_threshold=th0, A_init=A_tensor, init_eta=eta0)
    models['LISTA'] = train_2d(m_lista, needs_A=False)

    # LISTA-CP
    m_cp = LISTACP(A_tensor, T, init_eta=eta0, init_threshold=th0)
    models['LISTA-CP'] = train_2d(m_cp, needs_A=False)

    # LISTA-Momentum
    m_mom = LISTAMomentum(6, 2, T, init_eta=eta0, init_threshold=th0)
    models['LISTA-Mom'] = train_2d(m_mom, needs_A=True)

    # DA-LISTA
    m_da = DimensionAgnosticLISTA(T)
    models['DA-LISTA'] = train_2d(m_da, needs_A=True)

    # RNN-LISTA
    m_rnn = RNNLISTA(T)
    models['RNN-LISTA'] = train_2d(m_rnn, needs_A=True)

    # LSTM-LISTA
    m_lstm = LSTMLISTA(T)
    models['LSTM-LISTA'] = train_2d(m_lstm, needs_A=True)

    print("  训练完成\n")

    # 收集轨迹
    print("收集优化轨迹...")
    trajectories, labels, colors = [], [], []
    cmap = plt.cm.tab10

    # 经典方法
    traj_ista = collect_ista_traj(A, b, lam, T)
    trajectories.append(traj_ista); labels.append('ISTA'); colors.append(cmap(0))

    traj_fista = collect_fista_traj(A, b, lam, T)
    trajectories.append(traj_fista); labels.append('FISTA'); colors.append(cmap(1))

    # 学习型方法
    model_trajs = {
        'LISTA': lambda: collect_lista_traj(models['LISTA'], A, b, T),
        'LISTA-CP': lambda: collect_lista_traj(models['LISTA-CP'], A, b, T),
        'LISTA-Mom': lambda: collect_momentum_traj(models['LISTA-Mom'], A, b, T),
        'DA-LISTA': lambda: collect_da_traj(models['DA-LISTA'], A, b, T),
        'RNN-LISTA': lambda: collect_sequence_traj(models['RNN-LISTA'], A, b, T),
        'LSTM-LISTA': lambda: collect_sequence_traj(models['LSTM-LISTA'], A, b, T),
    }
    for i, (name, fn) in enumerate(model_trajs.items()):
        traj = fn()
        trajectories.append(traj)
        labels.append(name)
        colors.append(cmap(i + 2))

    # 打印终点误差
    print("\n各模型终点相对误差:")
    for traj, label in zip(trajectories, labels):
        err = relative_error(x_true, traj[-1])
        print(f"  {label:14s}: {err:.4f}  (终点: [{traj[-1][0]:.3f}, {traj[-1][1]:.3f}])")

    # 画图
    print("\n生成可视化...")
    os.makedirs('report', exist_ok=True)
    plot_3d_trajectory(A, b, lam, x_true, trajectories, labels, colors)
    plot_2d_contour(A, b, lam, x_true, trajectories, labels, colors)

    print("\n完成!")


if __name__ == '__main__':
    main()
