"""
common/training.py — 改进的训练框架

提供:
1. EarlyStopping — 基于验证损失的早停
2. GradientClipper — 梯度裁剪
3. Trainer — 统一的训练循环，支持早停、梯度裁剪、学习率调度
4. OODEvaluator — 分布外评估
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict


class EarlyStopping:
    """早停机制。

    Parameters
    ----------
    patience : int — 容忍验证损失不下降的轮数
    min_delta : float — 最小改善量
    mode : str — 'min' (越小越好) 或 'max' (越大越好)
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class Trainer:
    """统一的训练框架。

    支持:
    - 早停 (EarlyStopping)
    - 梯度裁剪
    - 学习率调度
    - 训练历史记录
    - 墙钟时间统计

    Parameters
    ----------
    model : nn.Module
    optimizer : optim.Optimizer
    criterion : nn.Module — 损失函数
    scheduler : optional — 学习率调度器
    grad_clip : float — 梯度裁剪范数 (0 表示不裁剪)
    patience : int — 早停耐心值 (0 表示不早停)
    device : torch.device
    verbose : bool
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module = nn.MSELoss(),
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        grad_clip: float = 1.0,
        patience: int = 20,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.device = device
        self.verbose = verbose

        self.early_stopping = EarlyStopping(patience=patience) if patience > 0 else None
        self.history = defaultdict(list)
        self.best_model_state = None
        self.best_epoch = 0

    def train_epoch(self, train_loader: DataLoader,
                    forward_fn: Callable) -> float:
        """训练一个 epoch。

        Parameters
        ----------
        train_loader : DataLoader
        forward_fn : Callable
            接受 (model, batch) 返回 (prediction, target) 的函数。

        Returns
        -------
        avg_loss : float
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            # 移到设备
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(self.device)

            # 前向传播
            pred, target = forward_fn(self.model, batch)
            loss = self.criterion(pred, target)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader,
                 forward_fn: Callable,
                 metric_fn: Optional[Callable] = None) -> Tuple[float, Optional[float]]:
        """评估模型。

        Parameters
        ----------
        val_loader : DataLoader
        forward_fn : Callable
        metric_fn : Callable, optional
            额外的评估指标函数。

        Returns
        -------
        avg_loss : float
        avg_metric : float or None
        """
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        num_batches = 0

        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(self.device)

            pred, target = forward_fn(self.model, batch)
            loss = self.criterion(pred, target)

            total_loss += loss.item()
            if metric_fn is not None:
                total_metric += metric_fn(pred, target).item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_metric = total_metric / num_batches if metric_fn else None

        return avg_loss, avg_metric

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        forward_fn: Callable,
        metric_fn: Optional[Callable] = None,
    ) -> Dict[str, List[float]]:
        """完整训练循环。

        Parameters
        ----------
        train_loader : DataLoader
        val_loader : DataLoader
        num_epochs : int
        forward_fn : Callable
        metric_fn : Callable, optional

        Returns
        -------
        history : dict
            包含 train_loss, val_loss, val_metric, lr, time 等。
        """
        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # 训练
            train_loss = self.train_epoch(train_loader, forward_fn)

            # 验证
            val_loss, val_metric = self.evaluate(val_loader, forward_fn, metric_fn)

            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(current_lr)
            self.history['time'].append(time.time() - epoch_start)
            if val_metric is not None:
                self.history['val_metric'].append(val_metric)

            # 保存最佳模型
            if epoch == 0 or val_loss < min(self.history['val_loss'][:-1]):
                self.best_model_state = {k: v.cpu().clone() for k, v in
                                          self.model.state_dict().items()}
                self.best_epoch = epoch

            # 早停
            if self.early_stopping is not None:
                if self.early_stopping(val_loss):
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break

            # 打印进度
            if self.verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                msg = (f"Epoch [{epoch+1}/{num_epochs}] "
                       f"Train: {train_loss:.6f} Val: {val_loss:.6f}")
                if val_metric is not None:
                    msg += f" Metric: {val_metric:.6f}"
                msg += f" LR: {current_lr:.6f} Time: {elapsed:.1f}s"
                print(msg)

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        self.history['total_time'] = time.time() - start_time
        self.history['best_epoch'] = self.best_epoch

        return dict(self.history)


class InferenceTimer:
    """推理时间测量工具。"""

    @staticmethod
    @torch.no_grad()
    def measure_inference_time(model: nn.Module, forward_fn: Callable,
                                input_data, num_runs: int = 100,
                                warmup: int = 10) -> Dict[str, float]:
        """测量模型推理时间。

        Parameters
        ----------
        model : nn.Module
        forward_fn : Callable
            接受 (model, input_data) 的函数。
        input_data : any
        num_runs : int — 测量次数
        warmup : int — 预热次数

        Returns
        -------
        timing : dict
            mean, std, min, max 推理时间 (毫秒)。
        """
        model.eval()
        device = next(model.parameters()).device

        # 预热
        for _ in range(warmup):
            forward_fn(model, input_data)

        # 同步 CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            forward_fn(model, input_data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

        times = np.array(times)
        return {
            'mean': float(times.mean()),
            'std': float(times.std()),
            'min': float(times.min()),
            'max': float(times.max()),
            'median': float(np.median(times)),
        }


class OODEvaluator:
    """分布外 (Out-of-Distribution) 评估器。"""

    @staticmethod
    def evaluate_ood(model: nn.Module, test_fn: Callable,
                     param_ranges: Dict[str, List],
                     num_samples: int = 50) -> Dict[str, Dict]:
        """在不同参数设置下评估模型的泛化性。

        Parameters
        ----------
        model : nn.Module
        test_fn : Callable
            接受 (model, **params) 返回指标的函数。
        param_ranges : dict
            参数名 -> 参数值列表。
        num_samples : int
            每个参数组合的测试样本数。

        Returns
        -------
        results : dict
            每个参数组合的评估结果。
        """
        results = {}

        # 生成参数组合
        import itertools
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            key = '_'.join(f'{k}={v}' for k, v in params.items())

            metrics = []
            for _ in range(num_samples):
                metric = test_fn(model, **params)
                metrics.append(metric)

            results[key] = {
                'params': params,
                'mean': np.mean(metrics),
                'std': np.std(metrics),
                'min': np.min(metrics),
                'max': np.max(metrics),
            }

        return results
