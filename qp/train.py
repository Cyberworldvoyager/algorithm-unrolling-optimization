"""
qp/train.py — PGD-Net 及其变体的训练脚本
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
from common.metrics import relative_error, optimality_gap
from common.training import Trainer, InferenceTimer
from qp.pgd_net import PGDNet, PGDMomentumNet, PGDLearnedQNet, create_pgd_net
from qp.problem import generate_qp_data, compute_optimal_solution, qp_objective


def prepare_data(n: int = 50, constraint_type: str = 'box',
                 num_train: int = 1000, num_val: int = 200,
                 batch_size: int = 64, seed: int = 42):
    """准备训练和验证数据。"""
    rng = np.random.RandomState(seed)

    def gen_batch(num):
        Q_list, c_list, x_opt_list = [], [], []
        for _ in range(num):
            data = generate_qp_data(n, constraint_type, seed=rng.randint(1e6))
            Q_list.append(data['Q'])
            c_list.append(data['c'])
            x_opt = compute_optimal_solution(data['Q'], data['c'],
                                              constraint_type, data['constraint_params'])
            x_opt_list.append(x_opt)
        return (torch.FloatTensor(np.array(Q_list)),
                torch.FloatTensor(np.array(c_list)),
                torch.FloatTensor(np.array(x_opt_list)))

    Q_train, c_train, x_train = gen_batch(num_train)
    Q_val, c_val, x_val = gen_batch(num_val)

    train_dataset = TensorDataset(Q_train, c_train, x_train)
    val_dataset = TensorDataset(Q_val, c_val, x_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_pgd_net(
    variant: str = 'basic',
    n: int = 50, constraint_type: str = 'box',
    T: int = 10, num_epochs: int = 100, lr: float = 1e-3,
    grad_clip: float = 1.0, patience: int = 20, seed: int = 42,
    verbose: bool = True,
):
    """训练 PGD-Net 变体。

    variant: 'basic', 'momentum', 'learned_q'
    """
    set_seed(seed)
    device = get_device()
    if verbose:
        print(f"Device: {device}")
        print(f"Training PGD-Net-{variant} with T={T} layers")

    train_loader, val_loader = prepare_data(n, constraint_type, seed=seed)

    model = create_pgd_net(n, variant=variant, T=T, constraint_type=constraint_type)
    if verbose:
        print(f"Parameters: {count_parameters(model)}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=10)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=nn.MSELoss(),
                      scheduler=scheduler, grad_clip=grad_clip,
                      patience=patience, device=device, verbose=verbose)

    def forward_fn(model, batch):
        Q, c, x_opt = batch
        x_pred = model(Q, c)
        return x_pred, x_opt

    def metric_fn(pred, target):
        errors = [relative_error(to_numpy(target[i]), to_numpy(pred[i]))
                  for i in range(pred.shape[0])]
        return torch.tensor(np.mean(errors))

    history = trainer.fit(train_loader, val_loader, num_epochs, forward_fn, metric_fn)

    # 推理时间
    Q_test = torch.eye(n, device=device).unsqueeze(0)
    c_test = torch.randn(1, n, device=device)
    timing = InferenceTimer.measure_inference_time(
        model, lambda m, inp: m(inp[0], inp[1]), (Q_test, c_test)
    )

    save_path = os.path.join(os.path.dirname(__file__), f'pgd_net_{variant}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'timing': timing,
        'params': {'variant': variant, 'n': n, 'T': T},
    }, save_path)
    if verbose:
        print(f"Model saved to {save_path}")
        print(f"Inference time: {timing['mean']:.2f} ± {timing['std']:.2f} ms")

    return model, history, timing


def main():
    variants = ['basic', 'momentum', 'learned_q']
    results = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training PGD-Net-{variant}")
        print('='*60)
        model, history, timing = train_pgd_net(variant=variant, verbose=True)
        results[variant] = {'history': history, 'timing': timing,
                            'params': count_parameters(model)}

    print(f"\n{'='*60}")
    print("Comparison Summary")
    print('='*60)
    for variant, res in results.items():
        best_val = min(res['history']['val_loss'])
        best_metric = min(res['history'].get('val_metric', [float('inf')]))
        print(f"PGD-{variant:12s}: params={res['params']:6d} "
              f"best_val={best_val:.6f} best_rel_err={best_metric:.6f} "
              f"inference={res['timing']['mean']:.2f}ms")


if __name__ == '__main__':
    main()
