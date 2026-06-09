"""
socp/problem.py — SOCP 标准锥规划数据生成

标准形式 (PDHG 友好):
    min   c^T x
    s.t.  A x = b
          x ∈ K = Q^{n_1} × Q^{n_2} × ... × Q^{n_p}
其中 Q^{k} = {(t, u) ∈ R × R^{k-1} : ||u||_2 <= t} 是二阶锥；
约定 k = 1 时退化为 R_+ (非负锥)。

数据采样 (feasibility-preserving):
    1. 随机生成 x* ∈ int(K), s* ∈ int(K*) 满足互补 <x*, s*> = 0
       (默认: 同一块要么 x* 严格内部 + s* = 0, 要么 s* 严格内部 + x* = 0,
       一部分块在锥边界上同时非零, 互补的 SOC 对偶关系会自动满足)。
    2. 随机 y* ∈ R^m。
    3. 反推 c = A^T y* + s*,  b = A x*.

这样保证 (x*, y*, s*) 是 KKT 点 -> x* 即为最优主解，y* 为最优对偶。
"""

import numpy as np
from typing import List, Tuple, Optional


def _sample_soc_interior(k: int, rng: np.random.RandomState,
                         scale: float = 1.0) -> np.ndarray:
    """从二阶锥严格内部 (||u|| < t) 采样一个点。"""
    if k == 1:
        return np.array([scale * (0.1 + rng.rand())])
    u = rng.randn(k - 1)
    t = np.linalg.norm(u) + scale * (0.1 + rng.rand())  # 保证 ||u|| < t
    z = np.empty(k)
    z[0] = t
    z[1:] = u
    return z


def _sample_soc_boundary(k: int, rng: np.random.RandomState,
                         scale: float = 1.0) -> np.ndarray:
    """从二阶锥边界 (||u|| = t > 0) 采样一个点 (active 约束)。"""
    if k == 1:
        return np.array([0.0])  # 1D 锥边界即 0
    u = rng.randn(k - 1)
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        u = np.ones(k - 1) / np.sqrt(k - 1)
        nu = 1.0
    t = scale * (0.1 + rng.rand())
    z = np.empty(k)
    z[0] = t
    z[1:] = (t / nu) * u
    return z


def _soc_dual_at_boundary(z: np.ndarray, rng: np.random.RandomState,
                          scale: float = 1.0) -> np.ndarray:
    """给定边界点 z = (t, u), ||u||=t > 0，
    生成与之互补的对偶 s = (s0, s_rest)，要求:
        s ∈ K* = K (自对偶), <z, s> = 0。
    满足互补的边界 s 形如 s0 > 0, s_rest = -(s0/t) u (反向同长度)。
    """
    k = z.shape[0]
    if k == 1:
        return np.array([scale * (0.1 + rng.rand())])  # 1D: s > 0 与 z=0 互补
    t = z[0]
    u = z[1:]
    s0 = scale * (0.1 + rng.rand())
    s_rest = -(s0 / t) * u
    s = np.empty(k)
    s[0] = s0
    s[1:] = s_rest
    return s


def generate_socp_data(
    block_sizes: List[int],
    m: int,
    seed: Optional[int] = None,
    p_active: float = 0.4,
    A_scale: float = 1.0,
) -> dict:
    """生成一个 SOCP 实例。

    Parameters
    ----------
    block_sizes : list of int
        K = Q^{n_1} × ... × Q^{n_p} 的各块维度。
    m : int
        等式约束数 (A 的行数)。
    seed : int, optional
        随机种子。
    p_active : float
        块在最优处约束 active (落在锥边界，且互补对偶非零) 的概率；
        其余块要么 x* 严格内部 (s*=0)，要么 x*=0 (s* 严格内部)。
    A_scale : float
        A 的列归一化前的随机幅度 (实测影响有限)。

    Returns
    -------
    data : dict
        {'A': (m, N), 'b': (m,), 'c': (N,),
         'block_sizes': list, 'N': int, 'm': int,
         'x_star': (N,), 'y_star': (m,), 's_star': (N,)}
    """
    rng = np.random.RandomState(seed)
    N = sum(block_sizes)

    # ===== 1. 采样最优主/对偶/松弛 =====
    x_star = np.zeros(N)
    s_star = np.zeros(N)
    offset = 0
    for k in block_sizes:
        end = offset + k
        # 三种状态: 内部主 (s=0), 边界互补 (x,s 都非零), 内部对偶 (x=0)
        r = rng.rand()
        if r < (1 - p_active) / 2:
            # 主严格内部, s = 0
            x_star[offset:end] = _sample_soc_interior(k, rng)
        elif r < 1 - p_active:
            # x = 0, 对偶严格内部
            s_star[offset:end] = _sample_soc_interior(k, rng)
        else:
            # 边界互补 (仅当 k >= 2 才能同时非零)
            if k == 1:
                # 1D 时退化: 随机选 x>0 或 s>0
                if rng.rand() < 0.5:
                    x_star[offset:end] = _sample_soc_interior(k, rng)
                else:
                    s_star[offset:end] = _sample_soc_interior(k, rng)
            else:
                z = _sample_soc_boundary(k, rng)
                x_star[offset:end] = z
                s_star[offset:end] = _soc_dual_at_boundary(z, rng)
        offset = end

    # ===== 2. 随机 A 与 y_star =====
    A = A_scale * rng.randn(m, N)
    # 适度归一化列, 防止极端尺度 (但保留一定异质性)
    col_norm = np.linalg.norm(A, axis=0, keepdims=True)
    A = A / np.maximum(col_norm, 1e-8)
    y_star = rng.randn(m)

    # ===== 3. 反推 b, c =====
    b = A @ x_star
    c = A.T @ y_star + s_star

    return {
        'A': A, 'b': b, 'c': c,
        'block_sizes': list(block_sizes),
        'N': N, 'm': m,
        'x_star': x_star, 'y_star': y_star, 's_star': s_star,
    }


def socp_objective(c: np.ndarray, x: np.ndarray) -> float:
    """线性目标值 c^T x。"""
    return float(c @ x)


def kkt_residuals(data: dict, x: np.ndarray, y: np.ndarray) -> dict:
    """评估候选解 (x, y) 的 KKT 残差。

    KKT 条件 (min c^T x s.t. Ax=b, x∈K; SOC 自对偶 K* = K):
        primal feas : A x = b,  x ∈ K
        dual   feas : s := c - A^T y ∈ K
        互补       : <s, x> = 0
    """
    from common.numerical import cone_dist as _cone_dist, project_cone

    A, b, c = data['A'], data['b'], data['c']
    block = data['block_sizes']
    s = c - A.T @ y
    s_proj = project_cone(s, block)
    primal_feas = float(np.linalg.norm(A @ x - b))
    cd = _cone_dist(x, block)
    dual_resid = float(np.linalg.norm(s - s_proj))
    gap = float(s @ x)
    out = {
        'primal_feas': primal_feas,
        'cone_dist': cd,
        'dual_resid': dual_resid,
        'gap': gap,
    }
    if 'x_star' in data:
        out['obj_gap'] = float(c @ x - c @ data['x_star'])
        out['rel_x_err'] = float(
            np.linalg.norm(x - data['x_star']) / (np.linalg.norm(data['x_star']) + 1e-12)
        )
    return out


def solve_with_scs(data: dict, verbose: bool = False) -> dict:
    """用 SCS (via CVXPY) 求精确解作为 ground truth 参照。"""
    import cvxpy as cp

    A, b, c = data['A'], data['b'], data['c']
    block = data['block_sizes']
    N = data['N']

    x = cp.Variable(N)
    constraints = [A @ x == b]
    offset = 0
    for k in block:
        end = offset + k
        if k == 1:
            constraints.append(x[offset] >= 0)
        else:
            constraints.append(cp.norm(x[offset + 1:end], 2) <= x[offset])
        offset = end
    prob = cp.Problem(cp.Minimize(c @ x), constraints)
    prob.solve(solver=cp.SCS, verbose=verbose)
    return {
        'x': np.array(x.value),
        'obj': float(prob.value),
        'status': prob.status,
    }
