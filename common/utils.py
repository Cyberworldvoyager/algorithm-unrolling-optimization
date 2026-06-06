"""
common/utils.py — 通用工具函数
"""

import torch
import numpy as np
import random
import os
from typing import Optional, Dict, Any


def set_seed(seed: int = 42):
    """设置全局随机种子以确保可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    """获取可用设备 (CPU/GPU)。"""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """统计模型可训练参数数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """将 PyTorch tensor 转换为 numpy 数组。"""
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def to_tensor(array: np.ndarray, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """将 numpy 数组转换为 PyTorch tensor。"""
    tensor = torch.from_numpy(array).to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def relative_error(x_true: np.ndarray, x_est: np.ndarray) -> float:
    """计算相对误差 ||x_true - x_est|| / ||x_true||。"""
    return float(np.linalg.norm(x_true - x_est) / (np.linalg.norm(x_true) + 1e-10))


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, loss: float, path: str):
    """保存训练检查点。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer],
                    path: str) -> Dict[str, Any]:
    """加载训练检查点。"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint
