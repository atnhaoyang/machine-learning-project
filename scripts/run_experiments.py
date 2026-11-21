"""项目总入口：从原始 CSV 生成完整训练、评估、图表和部署模型。

推荐阅读方法：
1. 先看文件末尾的 main()，理解整体顺序；
2. 再看 EXPERIMENTS，理解 B1–B6 是怎样累计变化的；
3. 最后阅读各训练与保存函数。

运行：
    conda run --no-capture-output -n pytorch python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from solubility.config import (
    CLEANED_DATA_PATH,
    DRUG_SMILES_COLUMN,
    ENSEMBLE_SEEDS,
    EXPERIMENT_LOG_PATH,
    FIGURE_DIR,
    METRICS_DETAIL_PATH,
    METRICS_SUMMARY_PATH,
    MODEL_DIR,
    N_SPLITS,
    OUTPUT_DIR,
    PREDICTION_DIR,
    SEED,
    SOLVENT_SMILES_COLUMN,
    SPLIT_PATH,
    TEMPERATURE_COLUMN,
    ensure_output_dirs,
)
from solubility.data import (
    load_raw_data,
    prepare_dataset,
    save_prepared_data,
)
from solubility.evaluation import compute_metrics
from solubility.features import FeatureTransformer, build_feature_arrays
from solubility.models import build_model
from solubility.plots import (
    make_ablation_figure,
    make_eda_figure,
    make_loss_curve,
    make_prediction_figure,
    make_solvent_error_figure,
    make_temperature_error_figure,
)
from solubility.training import ModelConfig, fit_full_model, make_loader, predict_scaled, set_seed, train_model


# B1–B5 是“累计消融”：
# 后一个方案在前一个方案基础上继续增加优化技术，而不是每次只改一个变量。
# 因此每一行表示“当前累计配置”，不能把相邻差值全部归因于最后一项技术。
EXPERIMENTS = {
    # B1：最小 PyTorch 基线。Adam + MSE，无 Dropout、无权重衰减。
    "B1_MLP_baseline": ModelConfig(
        name="B1_MLP_baseline",
        model_type="mlp",
        hidden_dim=128,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adam",
        scheduler="none",
        loss="mse",
        augment=False,
    ),
    # B2：只把优化器换成 AdamW，并加入余弦学习率退火。
    "B2_AdamW_cosine": ModelConfig(
        name="B2_AdamW_cosine",
        model_type="mlp",
        hidden_dim=128,
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        scheduler="cosine",
        loss="mse",
        augment=False,
    ),
    # B3：保留 B2，再加入 Dropout、weight decay 和 SmoothL1。
    # 学习率也由 1e-3 调整为 8e-4，因此这是正则化组合而非单项实验。
    "B3_regularized": ModelConfig(
        name="B3_regularized",
        model_type="mlp",
        hidden_dim=128,
        dropout=0.25,
        learning_rate=8e-4,
        weight_decay=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        loss="huber",
        augment=False,
    ),
    # B4：保留 B3，再加入训练时的轻微特征和温度扰动。
    "B4_augmented": ModelConfig(
        name="B4_augmented",
        model_type="mlp",
        hidden_dim=128,
        dropout=0.25,
        learning_rate=8e-4,
        weight_decay=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        loss="huber",
        augment=True,
    ),
    # B5：保留 B4，把普通 MLP 替换为注意力融合模型。
    # 隐藏维度也由 128 改为 96，因此结构收益不能全部归因于注意力机制。
    "B5_attention": ModelConfig(
        name="B5_attention",
        model_type="attention",
        hidden_dim=96,
        dropout=0.25,
        learning_rate=8e-4,
        weight_decay=1e-3,
        optimizer="adamw",
        scheduler="cosine",
        loss="huber",
        augment=True,
    ),
}


def assign_group_folds(frame: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    """按药物结构生成尽量均衡的五折。

    同一个 structure_group 的所有条件必须进入同一折，防止同一药物
    同时出现在训练集和测试集中。算法先随机打乱组，再按组大小从大到小
    分配到当前样本数最少的折。
    """
    rng = np.random.default_rng(seed)
    group_sizes = frame.groupby("structure_group").size()
    groups = group_sizes.index.to_numpy()
    rng.shuffle(groups)
    groups = sorted(groups, key=lambda group: group_sizes.loc[group], reverse=True)
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_to_fold: dict[str, int] = {}
    for group in groups:
        fold = int(np.argmin(fold_sizes))
        group_to_fold[str(group)] = fold
        fold_sizes[fold] += int(group_sizes.loc[group])
    return frame["structure_group"].map(group_to_fold).to_numpy(dtype=int)


def scale_target(target: np.ndarray, mean: float, std: float) -> np.ndarray:
    """用当前训练折的统计量标准化目标，使网络优化更稳定。"""
    return ((target - mean) / std).astype(np.float32)


def unscale_target(target_scaled: np.ndarray, mean: float, std: float) -> np.ndarray:
    """把标准化预测转换回原始 log10(mol/L)，用于指标和展示。"""
    return np.asarray(target_scaled, dtype=float) * std + mean


def fit_baselines(
    features: dict,
    frame: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[list[dict], list[pd.DataFrame], dict[str, object]]:
    """训练 B0 Ridge 和梯度提升树，并返回每折指标及折外预测。"""
    detail_rows: list[dict] = []
    oof_frames: list[pd.DataFrame] = []
    fitted = {}
    target = features["target"]

    for fold in range(N_SPLITS):
        # 每个 fold 都重新 fit 特征转换器和目标标准化，绝不使用验证折统计量。
        train = folds != fold
        valid = folds == fold
        transformer = FeatureTransformer()
        drug_train, solvent_train, temp_train = transformer.fit_transform(
            features["drug"][train],
            features["solvent"][train],
            features["temperature"][train],
        )
        drug_valid, solvent_valid, temp_valid = transformer.transform(
            features["drug"][valid],
            features["solvent"][valid],
            features["temperature"][valid],
        )
        x_train = np.concatenate([drug_train, solvent_train, temp_train], axis=1)
        x_valid = np.concatenate([drug_valid, solvent_valid, temp_valid], axis=1)
        mean = float(target[train].mean())
        std = float(target[train].std())
        y_train = scale_target(target[train], mean, std)
        y_valid = target[valid]

        models = {
            "B0_Ridge": Ridge(alpha=10.0),
            "B0_HistGradientBoosting": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=SEED + fold,
            ),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            prediction = unscale_target(model.predict(x_valid), mean, std)
            metrics = compute_metrics(y_valid, prediction, frame=frame.loc[valid].reset_index(drop=True))
            detail_rows.append(
                {
                    "experiment": name,
                    "fold": fold,
                    **metrics,
                    "best_epoch": np.nan,
                }
            )
            oof_frames.append(
                pd.DataFrame(
                    {
                        "experiment": name,
                        "row_index": frame.index[valid].to_numpy(),
                        "fold": fold,
                        "target": y_valid,
                        "prediction": prediction,
                    }
                )
            )
            fitted[name] = model

    return detail_rows, oof_frames, fitted


def run_single_seed_experiment(
    name: str,
    config: ModelConfig,
    features: dict,
    frame: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[list[dict], pd.DataFrame, list[dict]]:
    """用一个随机种子跑完某个累计消融方案的五折。"""
    detail_rows = []
    histories = []
    target = features["target"]
    oof_prediction = np.zeros(len(frame), dtype=float)

    for fold in range(N_SPLITS):
        # 与 B0 相同：训练折拟合转换器，验证折只调用 transform。
        train = folds != fold
        valid = folds == fold
        transformer = FeatureTransformer()
        drug_train, solvent_train, temp_train = transformer.fit_transform(
            features["drug"][train],
            features["solvent"][train],
            features["temperature"][train],
        )
        drug_valid, solvent_valid, temp_valid = transformer.transform(
            features["drug"][valid],
            features["solvent"][valid],
            features["temperature"][valid],
        )
        mean = float(target[train].mean())
        std = float(target[train].std())
        y_train = scale_target(target[train], mean, std)
        y_valid = scale_target(target[valid], mean, std)

        model, history, best_epoch = train_model(
            config,
            drug_train,
            solvent_train,
            temp_train,
            y_train,
            drug_valid,
            solvent_valid,
            temp_valid,
            y_valid,
            seed=SEED + fold,
        )
        valid_loader = make_loader(
            drug_valid,
            solvent_valid,
            temp_valid,
            y_valid,
            shuffle=False,
        )
        prediction_scaled = predict_scaled(model, valid_loader, torch.device("cpu"))
        prediction = unscale_target(prediction_scaled, mean, std)
        # 把每个折的预测填回全局 row_index，最终得到完整的折外预测 OOF。
        oof_prediction[valid] = prediction
        metrics = compute_metrics(
            target[valid],
            prediction,
            frame=frame.loc[valid].reset_index(drop=True),
        )
        detail_rows.append(
            {
                "experiment": name,
                "fold": fold,
                **metrics,
                "best_epoch": best_epoch,
            }
        )
        histories.append(history)

    oof_frame = pd.DataFrame(
        {
            "experiment": name,
            "row_index": frame.index.to_numpy(),
            "fold": folds,
            "target": target,
            "prediction": oof_prediction,
        }
    )
    return detail_rows, oof_frame, histories


def run_ensemble_experiment(
    name: str,
    config: ModelConfig,
    features: dict,
    frame: pd.DataFrame,
    folds: np.ndarray,
) -> tuple[list[dict], pd.DataFrame]:
    """用五个随机种子训练同一个注意力方案，再对预测取平均。"""
    detail_rows = []
    target = features["target"]
    oof_prediction = np.zeros(len(frame), dtype=float)

    for fold in range(N_SPLITS):
        train = folds != fold
        valid = folds == fold
        transformer = FeatureTransformer()
        drug_train, solvent_train, temp_train = transformer.fit_transform(
            features["drug"][train],
            features["solvent"][train],
            features["temperature"][train],
        )
        drug_valid, solvent_valid, temp_valid = transformer.transform(
            features["drug"][valid],
            features["solvent"][valid],
            features["temperature"][valid],
        )
        mean = float(target[train].mean())
        std = float(target[train].std())
        y_train = scale_target(target[train], mean, std)
        y_valid = scale_target(target[valid], mean, std)
        valid_loader = make_loader(
            drug_valid,
            solvent_valid,
            temp_valid,
            y_valid,
            shuffle=False,
        )
        fold_predictions = []
        best_epochs = []
        for seed in ENSEMBLE_SEEDS:
            # 每个 seed 的初始化和 batch 顺序不同，集成可以降低单次训练的偶然波动。
            model, history, best_epoch = train_model(
                config,
                drug_train,
                solvent_train,
                temp_train,
                y_train,
                drug_valid,
                solvent_valid,
                temp_valid,
                y_valid,
                seed=seed + fold,
            )
            fold_predictions.append(
                predict_scaled(model, valid_loader, torch.device("cpu"))
            )
            best_epochs.append(best_epoch)
        # 先在标准化空间取均值，再统一反变换回 log10 单位。
        prediction_scaled = np.mean(np.stack(fold_predictions), axis=0)
        prediction = unscale_target(prediction_scaled, mean, std)
        oof_prediction[valid] = prediction
        metrics = compute_metrics(
            target[valid],
            prediction,
            frame=frame.loc[valid].reset_index(drop=True),
        )
        detail_rows.append(
            {
                "experiment": name,
                "fold": fold,
                **metrics,
                "best_epoch": int(np.median(best_epochs)),
            }
        )
    oof_frame = pd.DataFrame(
        {
            "experiment": name,
            "row_index": frame.index.to_numpy(),
            "fold": folds,
            "target": target,
            "prediction": oof_prediction,
        }
    )
    return detail_rows, oof_frame


def aggregate_metrics(detail: pd.DataFrame) -> pd.DataFrame:
    """把每折指标聚合成每个实验的均值和标准差，并按 MAE 排序。"""
    summary = (
        detail.groupby("experiment")
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            spearman_mean=("spearman", "mean"),
            tolerance_accuracy_mean=("tolerance_accuracy_0_5", "mean"),
            tolerance_accuracy_std=("tolerance_accuracy_0_5", "std"),
            top3_solvent_accuracy_mean=("top3_solvent_accuracy", "mean"),
            top3_rank_groups_mean=("top3_rank_groups", "mean"),
            bias_mean=("bias", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
        )
        .reset_index()
    )
    summary["mae_std"] = summary["mae_std"].fillna(0.0)
    summary["rmse_std"] = summary["rmse_std"].fillna(0.0)
    summary["r2_std"] = summary["r2_std"].fillna(0.0)
    summary["tolerance_accuracy_std"] = summary["tolerance_accuracy_std"].fillna(0.0)
    return summary.sort_values("mae_mean").reset_index(drop=True)


def save_reference_fingerprints(frame: pd.DataFrame) -> None:
    """保存训练集唯一药物和溶剂的指纹，供应用判断输入是否超出适用范围。"""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
    unique_drugs = frame.drop_duplicates(DRUG_SMILES_COLUMN)
    unique_solvents = frame.drop_duplicates(SOLVENT_SMILES_COLUMN)

    def generate(smiles_values):
        arrays = []
        for smiles in smiles_values:
            mol = Chem.MolFromSmiles(smiles)
            fp = generator.GetFingerprint(mol)
            arrays.append(np.asarray(fp, dtype=np.int8))
        return np.stack(arrays)

    np.savez_compressed(
        MODEL_DIR / "reference_fingerprints.npz",
        drug_fingerprints=generate(unique_drugs[DRUG_SMILES_COLUMN]),
        solvent_fingerprints=generate(unique_solvents[SOLVENT_SMILES_COLUMN]),
    )


def save_known_options(frame: pd.DataFrame) -> None:
    """保存 UI 下拉列表使用的药物、溶剂名称及训练覆盖数量。"""
    drugs = (
        frame[[DRUG_SMILES_COLUMN, "Drug"]]
        .drop_duplicates(DRUG_SMILES_COLUMN)
        .rename(columns={DRUG_SMILES_COLUMN: "smiles", "Drug": "label"})
        .sort_values("label")
    )
    solvents = (
        frame[[SOLVENT_SMILES_COLUMN, "Solvent"]]
        .drop_duplicates(SOLVENT_SMILES_COLUMN)
        .rename(columns={SOLVENT_SMILES_COLUMN: "smiles", "Solvent": "label"})
        .sort_values("label")
    )
    drug_counts = frame.groupby(DRUG_SMILES_COLUMN).size().rename("drug_conditions")
    solvent_counts = frame.groupby(SOLVENT_SMILES_COLUMN).size().rename("solvent_conditions")
    drugs = drugs.merge(
        drug_counts,
        left_on="smiles",
        right_index=True,
        how="left",
    )
    solvents = solvents.merge(
        solvent_counts,
        left_on="smiles",
        right_index=True,
        how="left",
    )
    drugs.to_csv(MODEL_DIR / "known_drugs.csv", index=False)
    solvents.to_csv(MODEL_DIR / "known_solvents.csv", index=False)


def train_and_save_final_model(
    best_name: str,
    summary: pd.DataFrame,
    detail: pd.DataFrame,
    features: dict,
    frame: pd.DataFrame,
    oof: pd.DataFrame,
) -> dict:
    """选择验证最优模型，在全部数据上重训，并保存应用所需文件。"""
    # 最终部署模型使用完整数据的统计量；这些统计量不能用于报告交叉验证指标。
    transformer = FeatureTransformer()
    drug, solvent, temp = transformer.fit_transform(
        features["drug"], features["solvent"], features["temperature"]
    )
    target = features["target"]
    target_mean = float(target.mean())
    target_std = float(target.std())
    target_scaled = scale_target(target, target_mean, target_std)

    metadata = {
        "best_experiment": best_name,
        "target_mean": target_mean,
        "target_std": target_std,
        "drug_dim": int(drug.shape[1]),
        "solvent_dim": int(solvent.shape[1]),
        "temperature_range": [
            float(frame[TEMPERATURE_COLUMN].min()),
            float(frame[TEMPERATURE_COLUMN].max()),
        ],
        "supported_temperature_range": [5.0, 60.0],
        # 90% 经验误差范围来自折外预测，应用用它提示历史误差量级。
        "residual_q90": float(np.quantile(np.abs(oof["target"] - oof["prediction"]), 0.9)),
        "n_training_conditions": int(len(frame)),
        "n_unique_drugs": int(frame[DRUG_SMILES_COLUMN].nunique()),
        "n_solvents": int(frame[SOLVENT_SMILES_COLUMN].nunique()),
        "metrics": {
            str(key): (
                float(value)
                if isinstance(value, (float, int, np.floating, np.integer))
                and pd.notna(value)
                else value
            )
            for key, value in summary.set_index("experiment").loc[best_name].to_dict().items()
        },
    }

    is_baseline = best_name.startswith("B0_")
    if is_baseline:
        # 如果传统基线胜出，就保存对应 sklearn 模型；应用会自动支持这种引擎。
        flat = np.concatenate([drug, solvent, temp], axis=1)
        if best_name == "B0_Ridge":
            model = Ridge(alpha=10.0)
        else:
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=SEED,
            )
        model.fit(flat, target_scaled)
        joblib.dump(model, MODEL_DIR / "sklearn_model.joblib")
        metadata["engine"] = "sklearn"
        metadata["model_config"] = {"model_type": best_name}
    else:
        # B6_ensemble_attention 复用 B5 的注意力配置。
        config_name = best_name.replace("B6_ensemble_", "")
        if config_name not in EXPERIMENTS:
            config_name = "B5_attention"
        config = EXPERIMENTS[config_name]
        rows = detail.loc[detail["experiment"] == best_name, "best_epoch"]
        # 全量数据没有验证集，因此使用交叉验证中的典型最佳 epoch 作为固定轮数。
        epochs = int(max(25, min(100, round(float(rows.median())))))
        for seed in ENSEMBLE_SEEDS:
            model = fit_full_model(
                config,
                drug,
                solvent,
                temp,
                target_scaled,
                seed=seed,
                epochs=epochs,
            )
            torch.save(model.state_dict(), MODEL_DIR / f"ensemble_seed_{seed}.pt")
        metadata["engine"] = "torch"
        metadata["ensemble_seeds"] = list(ENSEMBLE_SEEDS)
        metadata["epochs"] = epochs
        metadata["model_config"] = {
            "name": config.name,
            "model_type": config.model_type,
            "hidden_dim": config.hidden_dim,
            "dropout": config.dropout,
        }

    joblib.dump(transformer, MODEL_DIR / "feature_transformer.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    """执行完整实验流水线"""
    ensure_output_dirs()
    set_seed(SEED)

    # 读取、清洗、合并重复条件，并生成数据审计图
    raw = load_raw_data()
    frame, audit = prepare_dataset(raw)
    save_prepared_data(frame, audit)
    make_eda_figure(frame)

    # 生成分子和温度特征，并固定药物分组五折。
    features = build_feature_arrays(frame)
    folds = assign_group_folds(frame)
    split_frame = frame[
        [
            DRUG_SMILES_COLUMN,
            SOLVENT_SMILES_COLUMN,
            TEMPERATURE_COLUMN,
            "Drug",
            "Solvent",
            "structure_group",
        ]
    ].copy()
    split_frame["fold"] = folds
    split_frame.to_csv(SPLIT_PATH, index=False)

    # 先运行 Ridge 和梯度提升树基线。
    detail_rows, oof_frames, _ = fit_baselines(features, frame, folds)
    histories_for_plot: dict[str, list[dict]] = {}

    # 依次运行 B1–B5。这里是累计消融，配置顺序会影响解释方式。
    for name, config in EXPERIMENTS.items():
        rows, oof, histories = run_single_seed_experiment(
            name, config, features, frame, folds
        )
        detail_rows.extend(rows)
        oof_frames.append(oof)
        if name in {"B1_MLP_baseline", "B3_regularized", "B5_attention"}:
            histories_for_plot[name] = histories

    # B6 在 B5 配置上训练五个随机种子并集成。
    ensemble_name = "B6_ensemble_attention"
    ensemble_rows, ensemble_oof = run_ensemble_experiment(
        ensemble_name,
        EXPERIMENTS["B5_attention"],
        features,
        frame,
        folds,
    )
    detail_rows.extend(ensemble_rows)
    oof_frames.append(ensemble_oof)

    detail = pd.DataFrame(detail_rows)
    summary = aggregate_metrics(detail)
    detail.to_csv(METRICS_DETAIL_PATH, index=False)
    summary.to_csv(METRICS_SUMMARY_PATH, index=False)

    # 保存折外预测，并让每条预测可以回查药物、溶剂和温度
    oof_all = pd.concat(oof_frames, ignore_index=True)
    oof_all.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    joint = oof_all.merge(
        frame.reset_index().rename(columns={"index": "row_index"}),
        on="row_index",
        how="left",
        suffixes=("", "_condition"),
    )
    joint.to_csv(OUTPUT_DIR / "oof_predictions_with_context.csv", index=False)

    # 按平均 MAE 选择最佳模型，生成消融、预测和误差分析图
    best_name = str(summary.iloc[0]["experiment"])
    best_oof = oof_all.loc[oof_all["experiment"] == best_name].copy()
    best_oof = best_oof.sort_values("row_index")
    best_context = frame.loc[best_oof["row_index"]].reset_index(drop=True)
    residual = best_oof["prediction"].to_numpy() - best_oof["target"].to_numpy()

    make_loss_curve(histories_for_plot)
    make_ablation_figure(summary)
    make_prediction_figure(
        best_oof["target"].to_numpy(),
        best_oof["prediction"].to_numpy(),
        best_name,
    )
    make_solvent_error_figure(best_context, residual)
    make_temperature_error_figure(best_context, residual)

    # 使用最优配置在全部数据上重训，并保存可直接加载的模型。
    save_known_options(frame)
    save_reference_fingerprints(frame)
    metadata = train_and_save_final_model(
        best_name,
        summary,
        detail,
        features,
        frame,
        best_oof,
    )

    experiment_log = {
        "audit": audit,
        "folds": {
            "method": "Greedy balanced GroupKFold by canonical drug structure",
            "n_splits": N_SPLITS,
            "fold_sizes": pd.Series(folds).value_counts().sort_index().to_dict(),
            # 一个药物结构若出现多个 fold，说明分组失败。
            "group_leakage_check": bool(
                (
                    split_frame.groupby("structure_group")["fold"].nunique() == 1
                ).all()
            ),
        },
        "best_experiment": best_name,
        "best_metrics": summary.iloc[0].to_dict(),
        "model_metadata": metadata,
        "experiments": list(EXPERIMENTS),
        "ensemble_experiment": ensemble_name,
    }
    EXPERIMENT_LOG_PATH.write_text(
        json.dumps(experiment_log, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"\nSaved final {metadata['engine']} model for {best_name}")


if __name__ == "__main__":
    main()
