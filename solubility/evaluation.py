"""回归指标和溶剂排序指标。"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    frame=None,
) -> dict:
    """计算一组预测的常用回归与业务指标。

    主指标是 MAE；同时保留 R²、RMSE、Spearman、±0.5 命中率和偏差。
    传入 frame 时还会计算 Top-3 溶剂排序命中率。
    """
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    residual = prediction - target
    spearman = spearmanr(target, prediction).statistic
    metrics = {
        "n": int(len(target)),
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(target, prediction))),
        "r2": float(r2_score(target, prediction)),
        "spearman": float(spearman) if np.isfinite(spearman) else float("nan"),
        # 回归模型没有分类 accuracy，这里把“误差不超过半个 log 单位”的比例
        # 作为业务上可解释的容忍命中率。
        "tolerance_accuracy_0_5": float(np.mean(np.abs(residual) <= 0.5)),
        "bias": float(np.mean(residual)),
    }
    if frame is not None:
        ranking = compute_top_k_solvent_accuracy(frame, target, prediction, k=3)
        metrics["top3_solvent_accuracy"] = ranking["score"]
        metrics["top3_rank_groups"] = ranking["groups"]
    return metrics


def compute_top_k_solvent_accuracy(
    frame,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    k: int = 3,
) -> dict[str, float | int]:
    """评估同一药物、同一温度下高溶解度溶剂是否进入预测 Top-k。

    只有至少包含 max(3, k) 种溶剂的组才参与计算，避免用信息不足的组
    制造虚高指标。
    """
    data = frame.copy()
    data["_target"] = target
    data["_prediction"] = prediction
    group_columns = ["Drug", "Temperature"]
    scores: list[float] = []

    for _, group in data.groupby(group_columns):
        if group["Solvent"].nunique() < max(3, k):
            continue
        # 比较真实和预测的高溶解度溶剂集合，并计算交集比例。
        actual_top = set(group.nlargest(k, "_target")["Solvent"])
        predicted_top = set(group.nlargest(k, "_prediction")["Solvent"])
        scores.append(len(actual_top & predicted_top) / k)

    if not scores:
        return {"score": float("nan"), "groups": 0}
    return {"score": float(np.mean(scores)), "groups": len(scores)}
