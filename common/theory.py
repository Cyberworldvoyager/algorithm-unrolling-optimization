"""
common/theory.py — 算法展开的理论分析工具

提供:
1. 谱半径分析 — 验证展开网络的稳定性条件
2. 收敛性证书 — 验证展开网络是否满足收敛条件
3. 近似误差界 — 量化展开网络与经典算法的差距
4. 参数统计 — 分析可学习参数的分布
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


def spectral_radius(W: torch.Tensor) -> float:
    """计算矩阵的谱半径 ρ(W) = max|λ_i(W)|。

    对于 LISTA 的 W₂ = I - ηBA，需要 ρ(W₂) < 1 保证收敛。
    """
    eigvals = torch.linalg.eigvals(W)
    return float(eigvals.abs().max())


def check_lista_stability(model: nn.Module, A: torch.Tensor) -> Dict[str, any]:
    """检查 LISTA-CP 模型的稳定性条件。

    理论条件: ρ(I - ηBA) < 1

    Returns
    -------
    results : dict
        包含每层的谱半径和是否满足稳定性条件。
    """
    results = {'layers': [], 'all_stable': True}

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'B') and hasattr(layer, 'A'):
            # LISTA-CP: W₂ = I - η·B·A
            eta = layer.eta if hasattr(layer, 'eta') else 1.0
            W2 = torch.eye(A.shape[1], device=A.device) - eta * layer.B @ layer.A
            rho = spectral_radius(W2)
            stable = rho < 1.0
            results['layers'].append({
                'layer': i,
                'spectral_radius': rho,
                'stable': stable,
                'eta': float(eta) if torch.is_tensor(eta) else eta,
            })
            if not stable:
                results['all_stable'] = False
        elif hasattr(layer, 'W2'):
            # 基本 LISTA: W₂ 直接学习
            rho = spectral_radius(layer.W2.weight)
            stable = rho < 1.0
            results['layers'].append({
                'layer': i,
                'spectral_radius': rho,
                'stable': stable,
            })
            if not stable:
                results['all_stable'] = False

    return results


def convergence_certificate_lista(model: nn.Module, A: torch.Tensor,
                                  b: torch.Tensor, x_true: torch.Tensor,
                                  lam: float) -> Dict[str, float]:
    """为 LISTA 模型生成收敛性证书。

    计算:
    1. 每层的残差 ||Ax - b||
    2. 每层的相对误差 ||x - x*|| / ||x*||
    3. 每层的目标函数值

    Returns
    -------
    certificate : dict
    """
    model.eval()
    with torch.no_grad():
        _, intermediates = model(b, return_intermediates=True)

    certificate = {'layers': []}
    for i, x in enumerate(intermediates):
        residual = torch.norm(A @ x.T - b.T, dim=0).mean().item()
        rel_err = torch.norm(x - x_true, dim=1).mean().item() / (torch.norm(x_true, dim=1).mean().item() + 1e-10)
        obj = 0.5 * torch.sum((A @ x.T - b.T) ** 2, dim=0).mean().item() + \
              lam * torch.sum(torch.abs(x), dim=1).mean().item()

        certificate['layers'].append({
            'layer': i,
            'residual': residual,
            'relative_error': rel_err,
            'objective': obj,
        })

    return certificate


def parameter_distribution(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """分析模型参数的统计分布。

    Returns
    -------
    stats : dict
        每个参数的均值、标准差、最小值、最大值、范数。
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            data = param.data.flatten()
            stats[name] = {
                'mean': data.mean().item(),
                'std': data.std().item(),
                'min': data.min().item(),
                'max': data.max().item(),
                'norm': data.norm().item(),
                'numel': data.numel(),
            }
    return stats


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量。"""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            by_layer[name] = param.numel()
    return {'total': total, 'by_layer': by_layer}


def compare_parameter_efficiency(models: Dict[str, nn.Module]) -> Dict[str, Dict]:
    """对比多个模型的参数效率。

    Parameters
    ----------
    models : dict
        {模型名: 模型实例}

    Returns
    -------
    comparison : dict
    """
    comparison = {}
    for name, model in models.items():
        param_stats = count_parameters(model)
        comparison[name] = {
            'total_params': param_stats['total'],
            'param_distribution': parameter_distribution(model),
        }
    return comparison
