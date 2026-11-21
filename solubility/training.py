"""PyTorch 训练、验证、早停和全量重训逻辑。"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS, SEED
from .models import build_model


@dataclass(frozen=True)
class ModelConfig:
    """一个实验方案的完整训练配置，对应 run_experiments.py 中的 B1–B5。"""

    name: str
    model_type: str
    hidden_dim: int
    dropout: float
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    loss: str
    augment: bool
    augment_noise: float = 0.005
    temperature_noise: float = 0.02


def set_seed(seed: int = SEED) -> None:
    """同时固定 Python、NumPy 和 Torch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """按可用硬件选择 CUDA、Apple MPS 或 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(
    drug: np.ndarray,
    solvent: np.ndarray,
    temperature: np.ndarray,
    target: np.ndarray,
    *,
    shuffle: bool,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> DataLoader:
    """把 numpy 数组包装成小批量 DataLoader。

    只有训练集需要 shuffle；验证集保持原顺序，确保预测和真实值正确对齐。
    """
    dataset = TensorDataset(
        torch.from_numpy(drug.astype(np.float32)),
        torch.from_numpy(solvent.astype(np.float32)),
        torch.from_numpy(temperature.astype(np.float32)),
        torch.from_numpy(target.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def _loss_function(name: str) -> nn.Module:
    """选择回归损失。

    SmoothL1（Huber）对大误差没有 MSE 那么敏感，更适合存在实验离群值的数据。
    """
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")


def _optimizer(model: nn.Module, config: ModelConfig) -> torch.optim.Optimizer:
    """根据配置创建 Adam 或带解耦权重衰减的 AdamW。"""
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


def _scheduler(optimizer: torch.optim.Optimizer, config: ModelConfig):
    """创建余弦退火学习率调度器；未启用时返回 None。"""
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=config.learning_rate * 0.05,
        )
    if config.scheduler == "none":
        return None
    raise ValueError(f"Unknown scheduler: {config.scheduler}")


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ModelConfig,
) -> float:
    """训练一个 epoch，并返回按样本数加权的平均损失。"""
    model.train()
    total_loss = 0.0
    total_examples = 0
    for drug, solvent, temperature, target in loader:
        drug = drug.to(device)
        solvent = solvent.to(device)
        temperature = temperature.to(device)
        target = target.to(device)

        if config.augment:
            # 训练阶段加入轻微数值扰动，等价于保守的数据增强。
            # 验证阶段不会执行这段逻辑，因此不会污染评估结果。
            drug = drug + torch.randn_like(drug) * config.augment_noise
            solvent = solvent + torch.randn_like(solvent) * config.augment_noise
            temperature = (
                temperature + torch.randn_like(temperature) * config.temperature_noise
            )

        optimizer.zero_grad(set_to_none=True)
        prediction = model(drug, solvent, temperature)
        loss = criterion(prediction, target)
        loss.backward()

        # 梯度裁剪防止小批量中的极端样本造成参数突变。
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item()) * len(target)
        total_examples += len(target)
    return total_loss / max(total_examples, 1)


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """在评估模式下预测标准化后的目标，不计算梯度。"""
    model.eval()
    predictions = []
    for drug, solvent, temperature, _ in loader:
        prediction = model(
            drug.to(device),
            solvent.to(device),
            temperature.to(device),
        )
        predictions.append(prediction.detach().cpu().numpy())
    return np.concatenate(predictions)


def _evaluate_scaled(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """计算验证集损失和预测值。"""
    predictions = predict_scaled(model, loader, device)
    targets = torch.cat([batch[-1] for batch in loader]).numpy()
    loss = criterion(
        torch.from_numpy(predictions),
        torch.from_numpy(targets),
    ).item()
    return float(loss), predictions


def train_model(
    config: ModelConfig,
    drug_train: np.ndarray,
    solvent_train: np.ndarray,
    temperature_train: np.ndarray,
    target_train: np.ndarray,
    drug_valid: np.ndarray,
    solvent_valid: np.ndarray,
    temperature_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    seed: int = SEED,
    max_epochs: int = MAX_EPOCHS,
) -> tuple[nn.Module, dict, int]:
    """训练一个模型并执行早停。

    返回：
    - 恢复到最佳验证损失时权重的模型；
    - 每个 epoch 的训练损失、验证损失和学习率；
    - 最佳 epoch。
    """
    set_seed(seed)
    device = get_device()
    model = build_model(
        name=config.model_type,
        drug_dim=drug_train.shape[1],
        solvent_dim=solvent_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    train_loader = make_loader(
        drug_train,
        solvent_train,
        temperature_train,
        target_train,
        shuffle=True,
        seed=seed,
    )
    valid_loader = make_loader(
        drug_valid,
        solvent_valid,
        temperature_valid,
        target_valid,
        shuffle=False,
    )
    criterion = _loss_function(config.loss)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    history = {"train_loss": [], "valid_loss": [], "learning_rate": []}

    best_loss = float("inf")
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        # 记录当前学习率，方便画学习率曲线或排查调度器问题。
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = _train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config,
        )
        valid_loss, _ = _evaluate_scaled(model, valid_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["learning_rate"].append(learning_rate)

        if valid_loss < best_loss - 1e-5:
            # 只在验证损失真正改善时保存权重，避免把过拟合后的参数用于预测。
            best_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            # 连续 PATIENCE 个 epoch 未改善则提前结束训练。
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                break

    # 最终部署和验证使用最佳 epoch 的权重，而不是最后一个 epoch。
    model.load_state_dict(best_state)
    model.to("cpu")
    model.eval()
    history["best_epoch"] = best_epoch
    history["best_valid_loss"] = best_loss
    return model, history, best_epoch


def fit_full_model(
    config: ModelConfig,
    drug: np.ndarray,
    solvent: np.ndarray,
    temperature: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    epochs: int,
) -> nn.Module:
    """在全部清洗数据上重新训练最终模型。

    这里没有单独验证集，因此使用交叉验证得到的典型最佳 epoch 数作为固定轮数。
    """
    set_seed(seed)
    device = get_device()
    model = build_model(
        name=config.model_type,
        drug_dim=drug.shape[1],
        solvent_dim=solvent.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    loader = make_loader(
        drug,
        solvent,
        temperature,
        target,
        shuffle=True,
        seed=seed,
    )
    criterion = _loss_function(config.loss)
    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)

    for _ in range(epochs):
        _train_epoch(model, loader, criterion, optimizer, device, config)
        if scheduler is not None:
            scheduler.step()

    model.eval()
    return model.cpu()
