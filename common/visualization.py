"""
common/visualization.py — 论文级可视化工具
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, List, Tuple, Dict


def setup_figure(
    figsize: Tuple[float, float] = (8, 5),
    font_size: int = 12,
    latex: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """初始化论文级图表参数，返回 (fig, ax)。"""
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1,
        'legend.fontsize': font_size - 1,
        'figure.figsize': figsize,
        'lines.linewidth': 1.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    if latex:
        plt.rcParams.update({
            'text.usetex': False,  # 默认关闭，避免环境问题
            'font.family': 'serif',
        })
    fig, ax = plt.subplots()
    return fig, ax


def convergence_plot(
    histories: Dict[str, np.ndarray],
    xlabel: str = 'Iteration',
    ylabel: str = 'Objective Value',
    title: str = 'Convergence Comparison',
    log_scale: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制多算法收敛曲线对比图。

    Parameters
    ----------
    histories : dict
        {算法名: 数组}, 每个数组记录每步的指标值。
    log_scale : bool
        是否使用对数 y 轴。
    """
    fig, ax = setup_figure()
    markers = ['o', 's', '^', 'D', 'v', 'p']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (name, values) in enumerate(histories.items()):
        iters = np.arange(1, len(values) + 1)
        ax.plot(
            iters, values,
            label=name,
            marker=markers[i % len(markers)],
            markevery=max(1, len(values) // 10),
            color=colors[i % len(colors)],
            markersize=6,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale('log')
    ax.legend(frameon=True, fancybox=True, shadow=True)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def heatmap(
    data: np.ndarray,
    title: str = '',
    xlabel: str = '',
    ylabel: str = '',
    cmap: str = 'viridis',
    annot: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制热力图，适用于矩阵可视化。"""
    fig, ax = setup_figure()
    im = ax.imshow(data, aspect='auto', cmap=cmap, interpolation='nearest')
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        color='white' if data[i, j] > data.mean() else 'black', fontsize=8)

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def bar_comparison(
    groups: Dict[str, List[float]],
    group_labels: List[str],
    title: str = '',
    ylabel: str = '',
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """分组柱状图，用于算法指标对比。

    Parameters
    ----------
    groups : dict
        {算法名: 各组的值列表}
    group_labels : list
        每组的标签
    """
    fig, ax = setup_figure()
    n_groups = len(group_labels)
    n_bars = len(groups)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (name, values) in enumerate(groups.items()):
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=name, color=colors[i % len(colors)])

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig, ax


def layer_effect_plot(
    layer_counts: List[int],
    metrics: Dict[str, List[float]],
    ylabel: str = 'Relative Error',
    title: str = 'Effect of Number of Layers',
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """绘制层数对性能影响的折线图。"""
    fig, ax = setup_figure()
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, values) in enumerate(metrics.items()):
        ax.plot(layer_counts, values, label=name, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], markersize=8, linewidth=2)

    ax.set_xlabel('Number of Layers (T)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layer_counts)
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    return fig, ax
