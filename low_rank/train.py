"""
low_rank/train.py — ADMM-Net 及其变体的训练脚本
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
from low_rank.admm_net import ADMMNet, ADMMNetV2, SoftImputeNet
from low_rank.problem import generate_matrix_completion_data


def prepare_data(m: int = 50, n: int = 50, rank: int = 5, ratio: float = 0.5,
                 num_train: int = 1000, num_val: int = 200,
                 batch_size: int = 64, seed: int = 42):
    """准备训练和验证数据。"""
    rng = np.random.RandomState(seed)

    def gen_batch(num):
        M_list, M_obs_list, mask_list = [], [], []
        for _ in range(num):
            data = generate_matrix_completion_data(m, n, rank, ratio, seed=rng.randint(1e6))
            M_list.append(data['M'])
            M_obs_list.append(data['M_observed'])
            mask_list.append(data['mask'])
        return (torch.FloatTensor(np.array(M_list)),
                torch.FloatTensor(np.array(M_obs_list)),
                torch.FloatTensor(np.array(mask_list)))

    M_train, M_obs_train, mask_train = gen_batch(num_train)
    M_val, M_obs_val, mask_val = gen_batch(num_val)

    train_dataset = TensorDataset(M_obs_train, mask_train, M_train)
    val_dataset = TensorDataset(M_obs_val, mask_val, M_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def train_admm_net(
    variant: str = 'v1',
    m: int = 50, n: int = 50, rank: int = 5, ratio: float = 0.5,
    T: int = 10, num_epochs: int = 100, lr: float = 1e-3,
    grad_clip: float = 1.0, patience: int = 20, seed: int = 42,
    verbose: bool = True,
):
    """训练 ADMM-Net 变体。

    variant: 'v1', 'v2', 'softimpute'
    """
    set_seed(seed)
    device = get_device()
    if verbose:
        print(f"Device: {device}")
        print(f"Training ADMM-Net-{variant} with T={T} layers")

    train_loader, val_loader = prepare_data(m, n, rank, ratio, seed=seed)

    # 创建模型
    if variant == 'v1':
        model = ADMMNet(m, n, T=T)
    elif variant == 'v2':
        model = ADMMNetV2(m, n, T=T)
    elif variant == 'softimpute':
        model = SoftImputeNet(m, n, T=T)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if verbose:
        print(f"Parameters: {count_parameters(model)}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      factor=0.5, patience=10)

    trainer = Trainer(model=model, optimizer=optimizer, criterion=nn.MSELoss(),
                      scheduler=scheduler, grad_clip=grad_clip,
                      patience=patience, device=device, verbose=verbose)

    def forward_fn(model, batch):
        M_obs, mask, M = batch
        M_pred = model(M_obs, mask)
        return M_pred, M

    def metric_fn(pred, target):
        errors = [relative_error(to_numpy(target[i]), to_numpy(pred[i]))
                  for i in range(pred.shape[0])]
        return torch.tensor(np.mean(errors))

    history = trainer.fit(train_loader, val_loader, num_epochs, forward_fn, metric_fn)

    # 推理时间
    M_obs_test = torch.randn(1, m, n, device=device)
    mask_test = torch.ones(1, m, n, device=device) * 0.5
    timing = InferenceTimer.measure_inference_time(
        model, lambda m, inp: m(inp[0], inp[1]), (M_obs_test, mask_test)
    )

    save_path = os.path.join(os.path.dirname(__file__), f'admm_net_{variant}_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'timing': timing,
        'params': {'variant': variant, 'm': m, 'n': n, 'T': T},
    }, save_path)
    if verbose:
        print(f"Model saved to {save_path}")
        print(f"Inference time: {timing['mean']:.2f} ± {timing['std']:.2f} ms")

    return model, history, timing


def main():
    variants = ['v1', 'v2', 'softimpute']
    results = {}
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Training ADMM-Net-{variant}")
        print('='*60)
        model, history, timing = train_admm_net(variant=variant, verbose=True)
        results[variant] = {'history': history, 'timing': timing,
                            'params': count_parameters(model)}

    print(f"\n{'='*60}")
    print("Comparison Summary")
    print('='*60)
    for variant, res in results.items():
        best_val = min(res['history']['val_loss'])
        best_metric = min(res['history'].get('val_metric', [float('inf')]))
        print(f"ADMM-{variant:12s}: params={res['params']:6d} "
              f"best_val={best_val:.6f} best_rel_err={best_metric:.6f} "
              f"inference={res['timing']['mean']:.2f}ms")


if __name__ == '__main__':
    main()
