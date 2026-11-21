"""训练结果和误差分析图表。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import FIGURE_DIR


COLORS = {
    "navy": "#17324D",
    "blue": "#2D6A8A",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "light": "#F4F1EA",
    "gray": "#6B7280",
    "red": "#B23A48",
}


def _finish(fig, filename: str, output_dir: Path = FIGURE_DIR) -> Path:
    """统一保存图片并关闭 Matplotlib figure，避免内存累积。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_eda_figure(frame: pd.DataFrame) -> Path:
    """生成目标分布、溶剂覆盖、温度和药物样本量四联图。"""
    sns.set_theme(style="whitegrid", font="Arial")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sns.histplot(frame["target"], bins=36, color=COLORS["teal"], ax=axes[0, 0])
    axes[0, 0].set_title("Log10 solubility distribution")
    axes[0, 0].set_xlabel("log10 solubility (mol/L)")
    axes[0, 0].set_ylabel("Conditions")

    top_solvents = frame["Solvent"].value_counts().head(15).sort_values()
    axes[0, 1].barh(top_solvents.index, top_solvents.values, color=COLORS["blue"])
    axes[0, 1].set_title("Most represented solvents")
    axes[0, 1].set_xlabel("Conditions")

    sns.scatterplot(
        data=frame,
        x="Temperature",
        y="target",
        hue="Solvent",
        legend=False,
        alpha=0.35,
        s=22,
        palette="turbo",
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Temperature and solubility")
    axes[1, 0].set_xlabel("Temperature (°C)")
    axes[1, 0].set_ylabel("log10 solubility")

    drug_counts = frame.groupby("structure_group").size()
    axes[1, 1].hist(
        drug_counts,
        bins=np.arange(0.5, drug_counts.max() + 1.5, 1),
        color=COLORS["orange"],
    )
    axes[1, 1].set_title("Conditions per drug structure")
    axes[1, 1].set_xlabel("Conditions")
    axes[1, 1].set_ylabel("Drug structures")

    fig.suptitle("Dataset audit: drug solubility in organic solvents", fontsize=15)
    return _finish(fig, "01_data_audit.png")


def make_loss_curve(histories: dict[str, list[dict]]) -> Path:
    """把多个实验的五折验证损失画成均值曲线和波动带。"""
    sns.set_theme(style="whitegrid", font="Arial")
    fig, ax = plt.subplots(figsize=(10, 5.6))
    palette = [COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["orange"]]

    for color, (name, folds) in zip(palette, histories.items()):
        if not folds:
            continue
        valid_curves = [fold["valid_loss"] for fold in folds]
        max_length = max(len(curve) for curve in valid_curves)
        padded = np.stack(
            [
                np.pad(curve, (0, max_length - len(curve)), mode="edge")
                for curve in valid_curves
            ]
        )
        mean = padded.mean(axis=0)
        lower = padded.min(axis=0)
        upper = padded.max(axis=0)
        epochs = np.arange(1, max_length + 1)
        ax.plot(epochs, mean, label=f"{name} mean", color=color, linewidth=2)
        ax.fill_between(epochs, lower, upper, color=color, alpha=0.12)

    ax.set_title("Validation loss during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 / MSE loss on standardized target")
    ax.legend(frameon=False)
    return _finish(fig, "02_loss_curves.png")


def make_ablation_figure(summary: pd.DataFrame) -> Path:
    """按 MAE 排序展示各累计消融方案，误差棒表示折间离散程度。"""
    sns.set_theme(style="whitegrid", font="Arial")
    data = summary.sort_values("mae_mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        data["experiment"],
        data["mae_mean"],
        xerr=data["mae_std"].fillna(0),
        color=[
            COLORS["teal"] if index == 0 else COLORS["blue"]
            for index in range(len(data))
        ],
        alpha=0.92,
        capsize=4,
    )
    for bar, value in zip(bars, data["mae_mean"]):
        ax.text(
            value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color=COLORS["navy"],
        )
    ax.set_title("Grouped validation MAE by model and optimization")
    ax.set_xlabel("MAE in log10 units (lower is better)")
    ax.set_ylabel("")
    ax.invert_yaxis()
    return _finish(fig, "03_ablation_mae.png")


def make_prediction_figure(target: np.ndarray, prediction: np.ndarray, label: str) -> Path:
    """生成预测—真实值散点图和残差分布图。"""
    sns.set_theme(style="whitegrid", font="Arial")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    residual = prediction - target

    axes[0].scatter(target, prediction, alpha=0.52, s=24, color=COLORS["blue"])
    limits = [
        min(float(target.min()), float(prediction.min())) - 0.2,
        max(float(target.max()), float(prediction.max())) + 0.2,
    ]
    axes[0].plot(limits, limits, "--", color=COLORS["orange"], linewidth=1.5)
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_title(f"Prediction parity: {label}")
    axes[0].set_xlabel("Observed log10 solubility")
    axes[0].set_ylabel("Predicted log10 solubility")

    sns.histplot(residual, bins=34, kde=True, color=COLORS["teal"], ax=axes[1])
    axes[1].axvline(0, color=COLORS["orange"], linestyle="--", linewidth=1.5)
    axes[1].set_title("Out-of-fold residual distribution")
    axes[1].set_xlabel("Prediction error (log10 units)")
    axes[1].set_ylabel("Conditions")
    return _finish(fig, "04_prediction_diagnostics.png")


def make_solvent_error_figure(frame: pd.DataFrame, residual: np.ndarray) -> Path:
    """展示样本量至少为 10 的溶剂上的平均绝对误差。"""
    sns.set_theme(style="whitegrid", font="Arial")
    data = frame[["Solvent"]].copy()
    data["abs_error"] = np.abs(residual)
    errors = (
        data.groupby("Solvent")["abs_error"]
        .agg(["mean", "count"])
        .query("count >= 10")
        .sort_values("mean", ascending=False)
        .head(18)
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [
        COLORS["red"] if value > errors["mean"].median() else COLORS["teal"]
        for value in errors["mean"]
    ]
    ax.barh(errors.index, errors["mean"], color=colors, alpha=0.9)
    ax.set_title("Largest solvent-level errors (at least 10 conditions)")
    ax.set_xlabel("Mean absolute error in log10 units")
    ax.set_ylabel("")
    return _finish(fig, "05_solvent_errors.png")


def make_temperature_error_figure(
    frame: pd.DataFrame,
    residual: np.ndarray,
) -> Path:
    """展示不同温度下的预测残差，用于发现系统性偏差。"""
    sns.set_theme(style="whitegrid", font="Arial")
    data = frame[["Temperature"]].copy()
    data["residual"] = residual
    fig, ax = plt.subplots(figsize=(10, 5.4))
    sns.scatterplot(
        data=data,
        x="Temperature",
        y="residual",
        color=COLORS["blue"],
        alpha=0.5,
        s=24,
        ax=ax,
    )
    ax.axhline(0, color=COLORS["orange"], linestyle="--", linewidth=1.5)
    ax.set_title("Residuals across temperature")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Prediction error (log10 units)")
    return _finish(fig, "06_temperature_residuals.png")
