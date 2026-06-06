"""
lasso/train.py — LISTA 及其变体的训练脚本

支持训练:
- LISTA (基本)
- LISTA-CP (耦合权重)
- LISTA-CP-SS (逐维度阈值)
- LISTA-CP-FISTA (带动量)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import set_seed, get_device, to_numpy, count_parameters
from common.metrics import relative_error
from common.training import Trainer, InferenceTimer
from lasso.lista import LISTA, LISTACP, LISTACPSS, LISTACPFISTA, create_lista
from lasso.problem import generate_lasso_data


def prepare_data(m: int = 50, n: int = 200, sparsity: int = 10,
                 num_train: int = 1000, num_val: int = 200,
                 noise_std: float = 0.01, batch_size: int = 64,
                 seed: int = 42):
    """准备训练和验证数据。

    注意: 所有样本共享同一个 A 矩阵 (LISTA 的标准设置)。
    """
    rng = np.random.RandomState(seed)

    # 生成共享的 A 矩阵
    A = rng.randn(m, n)
    A /= np.linalg.norm(A, axis=0, keepdims=True)

    # 生成训练数据
    X_train, B_train = [], []
    for _ in range(num_train):
        x = np.zeros(n)
        support = rng.choice(n, sparsity, replace=False)
        x[support] = rng.randn(sparsity)
        b = A @ x + noise_std * rng.randn(m)
        X_train.append(x)
        B_train.append(b)

    # 生成验证数据
    X_val, B_val = [], []
    for _ in range(num_val):
        x = np.zeros(n)
        support = rng.choice(n, sparsity, replace=False)
        x[support] = rng.randn(sparsity)
        b = A @ x + noise_std * rng.randn(m)
        X_val.append(x)
        B_val.append(b)

    # 转换为 tensor
    train_dataset = TensorDataset(
        torch.FloatTensor(np.array(B_train)),
        torch.FloatTensor(np.array(X_train)),
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(np.array(B_val)),
        torch.FloatTensor(np.array(X_val)),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, A


def train_lista(
    variant: str = 'cp',
    m: int = 50,
    n: int = 200,
    sparsity: int = 10,
    T: int = 10,
    num_epochs: int = 100,
    lr: float = 1e-3,
    grad_clip: float = 1.0,
    patience: int = 20,
    seed: int = 42,
    verbose: bool = True,
):
    """训练 LISTA 变体。

    Parameters
    ----------
    variant : str — 'basic', 'cp', 'ss', 'fista'
    """
    set_seed(seed)
    device = get_device()
    if verbose:
        print(f"Device: {device}")
        print(f"Training LISTA-{variant.upper()} with T={T} layers")

    # 准备数据
    train_loader, val_loader, A = prepare_data(m, n, sparsity, seed=seed)
    A_tensor = torch.FloatTensor(A)

    # 创建模型
    model = create_lista(A_tensor, variant=variant, T=T, init_eta=0.1)
    if verbose:
        print(f"Parameters: {count_parameters(model)}")

    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=10)

    # 训练器
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=nn.MSELoss(),
        scheduler=scheduler,
        grad_clip=grad_clip,
        patience=patience,
        device=device,
        verbose=verbose,
    )

    # 定义前向函数
    def forward_fn(model, batch):
        b, x = batch
        x_pred = model(b)
        return x_pred, x

    # 定义评估指标
    def metric_fn(pred, target):
        # 平均相对误差
        errors = []
        for i in range(pred.shape[0]):
            errors.append(relative_error(to_numpy(target[i]), to_numpy(pred[i])))
        return torch.tensor(np.mean(errors))

    # 训练
    history = trainer.fit(train_loader, val_loader, num_epochs, forward_fn, metric_fn)

    # 测量推理时间
    b_test = torch.randn(1, m, device=device)
    timing = InferenceTimer.measure_inference_time(
        model, lambda m, b: m(b), b_test, num_runs=100
    )

    # 保存
    save_path = os.path.join(os.path.dirname(__file__), f'lista_{variant}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'timing': timing,
        'params': {'variant': variant, 'm': m, 'n': n, 'T': T},
        'A': A,
    }, save_path)
    if verbose:
        print(f"Model saved to {save_path}")
        print(f"Inference time: {timing['mean']:.2f} ± {timing['std']:.2f} ms")

    return model, history, timing


def main():
    """训练所有 LISTA 变体并对比。"""
    variants = ['basic', 'cp', 'ss', 'fista']
    results = {}

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training LISTA-{variant.upper()}")
        print('='*60)
        model, history, timing = train_lista(variant=variant, verbose=True)
        results[variant] = {
            'history': history,
            'timing': timing,
            'params': count_parameters(model),
        }

    # 打印对比
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print('='*60)
    for variant, res in results.items():
        best_val = min(res['history']['val_loss'])
        best_metric = min(res['history'].get('val_metric', [float('inf')]))
        print(f"LISTA-{variant.upper():8s}: "
              f"params={res['params']:6d} "
              f"best_val={best_val:.6f} "
              f"best_rel_err={best_metric:.6f} "
              f"inference={res['timing']['mean']:.2f}ms")


if __name__ == '__main__':
    main()
