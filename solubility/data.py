"""数据读取、SMILES 规范化和重复实验条件处理。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .config import (
    AUDIT_PATH,
    CLEANED_DATA_PATH,
    DATA_PATH,
    DRUG_SMILES_COLUMN,
    SOLVENT_SMILES_COLUMN,
    TARGET_COLUMN,
    TEMPERATURE_COLUMN,
)


REQUIRED_COLUMNS = [
    "Drug",
    "Solvent",
    DRUG_SMILES_COLUMN,
    SOLVENT_SMILES_COLUMN,
    TEMPERATURE_COLUMN,
    TARGET_COLUMN,
    "Solubility Unit",
]


def load_raw_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """读取原始 CSV，并检查必要字段是否存在。

    文件实际使用 GB18030 编码，但程序先尝试 UTF-8，方便以后替换为
    常见的 UTF-8 CSV。
    """
    path = Path(path)
    try:
        frame = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(path, encoding="gb18030")

    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return frame


def canonicalize_smiles(smiles: str) -> str:
    """把 SMILES 转换为 RDKit 规范形式。

    规范化后，同一种分子即使原始字符串写法不同，也会得到同一个表示，
    这样才能正确识别重复条件并按药物结构进行分组。
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _structure_group(smiles: str) -> str:
    """为每个规范药物结构生成稳定的短哈希组名，用于五折分组。"""
    digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()[:16]
    return f"drug_{digest}"


def _murcko_scaffold(smiles: str) -> str:
    """计算 Murcko 骨架；复杂盐或 RDKit 无法处理的 SMILES 会回退到原结构。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    except Exception:
        return smiles
    return scaffold or "[acyclic]"


def prepare_dataset(raw: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """完成数据清洗，并返回“一个实验条件一行”的建模表和审计信息。

    核心处理：
    1. 校验缺失值、数值字段和单位；
    2. 规范化药物和溶剂 SMILES；
    3. 按“规范药物 + 规范溶剂 + 精确温度”合并重复测量；
    4. 保留重复测量数量和波动，用于后续数据质量分析。
    """
    if raw is None:
        raw = load_raw_data()

    # 只保留建模需要的字段，避免后续无意使用其他列造成信息泄漏。
    raw = raw.loc[:, REQUIRED_COLUMNS].copy()

    # 本数据集没有缺失值；如果没有显式处理，直接报错比静默填补更安全。
    if raw.isna().any().any():
        missing = raw.isna().sum()
        raise ValueError(f"Missing values are not supported: {missing[missing > 0].to_dict()}")
    if set(raw["Solubility Unit"].astype(str)) != {"mol/L"}:
        units = sorted(raw["Solubility Unit"].astype(str).unique().tolist())
        raise ValueError(f"Unexpected solubility units: {units}")

    temperature = pd.to_numeric(raw[TEMPERATURE_COLUMN], errors="coerce")
    target = pd.to_numeric(raw[TARGET_COLUMN], errors="coerce")
    if temperature.isna().any() or target.isna().any():
        raise ValueError("Temperature and target must be numeric")

    # 逐行规范化 SMILES。无效 SMILES 不自动删除，而是记录行号后整体报错，
    # 防止数据质量问题在训练过程中悄悄传播。
    canonical_drugs: list[str] = []
    canonical_solvents: list[str] = []
    invalid_rows: list[int] = []
    for index, row in raw.iterrows():
        try:
            canonical_drugs.append(canonicalize_smiles(row[DRUG_SMILES_COLUMN]))
            canonical_solvents.append(canonicalize_smiles(row[SOLVENT_SMILES_COLUMN]))
        except ValueError:
            invalid_rows.append(int(index))
            canonical_drugs.append("")
            canonical_solvents.append("")

    if invalid_rows:
        raise ValueError(f"Invalid SMILES rows: {invalid_rows[:20]}")

    exact_duplicates = int(raw.duplicated().sum())
    prepared = raw.copy()
    prepared[DRUG_SMILES_COLUMN] = canonical_drugs
    prepared[SOLVENT_SMILES_COLUMN] = canonical_solvents
    prepared[TEMPERATURE_COLUMN] = temperature.astype(float)
    prepared[TARGET_COLUMN] = target.astype(float)
    # structure_group 是分组交叉验证的关键：同一个药物结构必须始终分到同一折。
    prepared["structure_group"] = prepared[DRUG_SMILES_COLUMN].map(_structure_group)
    prepared["drug_scaffold"] = prepared[DRUG_SMILES_COLUMN].map(_murcko_scaffold)

    condition_key = [
        DRUG_SMILES_COLUMN,
        SOLVENT_SMILES_COLUMN,
        TEMPERATURE_COLUMN,
    ]
    # 先统计哪些条件被重复测量，以及同一条件的标签差异有多大。
    repeated = (
        prepared.groupby(condition_key, dropna=False)[TARGET_COLUMN]
        .agg(["size", "std", "min", "max"])
        .reset_index()
    )
    repeated_conditions = repeated.loc[repeated["size"] > 1]
    conflict_count = int(
        ((repeated_conditions["max"] - repeated_conditions["min"]) > 0.5).sum()
    )

    # 模型训练时对重复条件取均值。保留 min/std/max 是为了在审计文件中
    # 识别潜在记录错误或实验噪声。
    grouped = (
        prepared.groupby(condition_key, as_index=False, sort=False)
        .agg(
            Drug=("Drug", "first"),
            Solvent=("Solvent", "first"),
            Solubility_unit=("Solubility Unit", "first"),
            target=(TARGET_COLUMN, "mean"),
            replicates=(TARGET_COLUMN, "size"),
            target_std=(TARGET_COLUMN, "std"),
            target_min=(TARGET_COLUMN, "min"),
            target_max=(TARGET_COLUMN, "max"),
            structure_group=("structure_group", "first"),
            drug_scaffold=("drug_scaffold", "first"),
        )
    )
    grouped["target_std"] = grouped["target_std"].fillna(0.0)
    grouped["target_range"] = grouped["target_max"] - grouped["target_min"]
    grouped = grouped.reset_index().rename(columns={"index": "condition_id"})

    audit = {
        "raw_rows": int(len(raw)),
        "raw_columns": int(len(raw.columns)),
        "rows_after_condition_aggregation": int(len(grouped)),
        "exact_duplicate_rows": exact_duplicates,
        "unique_drug_names": int(raw["Drug"].nunique()),
        "unique_canonical_drug_structures": int(grouped[DRUG_SMILES_COLUMN].nunique()),
        "unique_solvents": int(grouped["Solvent"].nunique()),
        "unique_canonical_solvents": int(grouped[SOLVENT_SMILES_COLUMN].nunique()),
        "unique_structure_solvent_pairs": int(
            grouped.groupby([DRUG_SMILES_COLUMN, SOLVENT_SMILES_COLUMN]).ngroups
        ),
        "repeated_condition_groups": int(len(repeated_conditions)),
        "repeated_condition_rows": int(repeated_conditions["size"].sum()),
        "conflicting_conditions_range_gt_0_5": conflict_count,
        "max_replicate_range_log10": float(
            repeated_conditions["max"].sub(repeated_conditions["min"]).max()
        )
        if len(repeated_conditions)
        else 0.0,
        "temperature_min_c": float(grouped[TEMPERATURE_COLUMN].min()),
        "temperature_max_c": float(grouped[TEMPERATURE_COLUMN].max()),
        "target_min_log10": float(grouped["target"].min()),
        "target_max_log10": float(grouped["target"].max()),
        "target_mean_log10": float(grouped["target"].mean()),
        "target_std_log10": float(grouped["target"].std()),
        "structures_with_at_most_5_conditions": int(
            (grouped.groupby("structure_group").size() <= 5).sum()
        ),
        "solvents_with_at_most_5_conditions": int(
            (grouped.groupby(SOLVENT_SMILES_COLUMN).size() <= 5).sum()
        ),
        "canonicalization": "RDKit canonical isomeric SMILES; salt forms retained",
        "duplicate_policy": (
            "Rows sharing canonical drug/solvent SMILES and exact temperature are averaged; "
            "replicate variance is retained for auditing."
        ),
    }

    return grouped, audit


def save_prepared_data(frame: pd.DataFrame, audit: dict) -> None:
    """把清洗结果和审计指标写入 outputs/，便于检查或用于其他程序。"""
    frame.to_csv(CLEANED_DATA_PATH, index=False, encoding="utf-8")
    AUDIT_PATH.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_cleaned_data(path: Path | str = CLEANED_DATA_PATH) -> pd.DataFrame:
    """读取已经清洗的数据，并恢复容易被 CSV 读成文本的数值列。"""
    frame = pd.read_csv(path)
    numeric_columns = [
        TEMPERATURE_COLUMN,
        "target",
        "replicates",
        "target_std",
        "target_min",
        "target_max",
        "target_range",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame
