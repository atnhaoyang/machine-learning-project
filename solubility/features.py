"""把 SMILES 和温度转换成神经网络输入特征。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, DataStructs, Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator

from .config import FINGERPRINT_BITS, MORGAN_RADIUS


# 这些描述符概括分子大小、亲疏水性、极性、氢键能力、环结构和电荷等性质。
# 字典的插入顺序就是最终特征向量中的列顺序。
DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "MolLogP": Crippen.MolLogP,
    "MolMR": Crippen.MolMR,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "RingCount": rdMolDescriptors.CalcNumRings,
    "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings,
    "NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NHOHCount": Lipinski.NHOHCount,
    "NOCount": Lipinski.NOCount,
    "LabuteASA": Descriptors.LabuteASA,
    "BalabanJ": Descriptors.BalabanJ,
    "BertzCT": Descriptors.BertzCT,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "NumRadicalElectrons": Descriptors.NumRadicalElectrons,
    "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters,
    "FormalCharge": Chem.GetFormalCharge,
    "NumFragments": lambda mol: len(Chem.GetMolFrags(mol)),
}

DESCRIPTOR_NAMES = list(DESCRIPTOR_FUNCTIONS)


def _replace_nonfinite(values: np.ndarray) -> np.ndarray:
    """把 NaN 和正负无穷替换成 0，防止后续标准化和模型计算失败。"""
    values = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def featurize_smiles(smiles: str) -> tuple[np.ndarray, np.ndarray]:
    """把单个 SMILES 转为“连续描述符 + 二值 Morgan 指纹”。

    返回：
    - descriptors：固定长度的一维浮点数组；
    - fingerprint：长度为 FINGERPRINT_BITS 的 0/1 数组。
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # 少数描述符在特殊结构上可能抛异常，因此逐个计算并做安全回退。
    descriptor_values = []
    for function in DESCRIPTOR_FUNCTIONS.values():
        try:
            value = float(function(mol))
        except Exception:
            value = 0.0
        descriptor_values.append(value)
    descriptors = _replace_nonfinite(np.asarray(descriptor_values))

    # Morgan 指纹记录局部子结构是否出现。它擅长表达环、官能团和局部连接，
    # 与连续物化描述符形成互补。
    fingerprint = np.zeros(FINGERPRINT_BITS, dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=FINGERPRINT_BITS,
        includeChirality=True,
    )
    bit_vector = generator.GetFingerprint(mol)
    DataStructs.ConvertToNumpyArray(bit_vector, fingerprint)
    return descriptors, fingerprint.astype(np.float32)


def combine_smiles_features(descriptors: np.ndarray, fingerprint: np.ndarray) -> np.ndarray:
    """按“描述符在前、指纹在后”的顺序拼成一个分子特征向量。"""
    return np.concatenate(
        [np.asarray(descriptors, dtype=np.float32), np.asarray(fingerprint, dtype=np.float32)]
    )


def featurize_smiles_combined(smiles: str) -> np.ndarray:
    """直接返回拼接后的单个分子特征向量。"""
    descriptors, fingerprint = featurize_smiles(smiles)
    return combine_smiles_features(descriptors, fingerprint)


def temperature_features(temperature_celsius: np.ndarray | float) -> np.ndarray:
    """生成三个温度特征。

    T 捕获绝对温度，1/T 接近 van't Hoff 热力学形式，log(T) 描述温度比例。
    温度先统一转为开尔文，避免摄氏度直接参与倒数计算。
    """
    temperature = np.asarray(temperature_celsius, dtype=np.float64)
    kelvin = temperature + 273.15
    result = np.stack(
        [
            temperature,
            1000.0 / kelvin,
            np.log(kelvin),
        ],
        axis=-1,
    )
    return result.astype(np.float32)


def feature_names() -> list[str]:
    """返回与拼接特征一一对应的列名，便于调试和保存特征元数据。"""
    names = [f"desc_{name}" for name in DESCRIPTOR_NAMES]
    names.extend(f"ecfp_{index}" for index in range(FINGERPRINT_BITS))
    return names


@dataclass
class FeatureTransformer:
    """只使用训练折拟合的缺失值中位数、均值和标准差。

    训练集拟合、验证集复用是防止信息泄漏的必要步骤。
    指纹是 0/1 特征，不做标准化；只标准化连续描述符和温度。
    """
    drug_median: np.ndarray | None = None
    drug_mean: np.ndarray | None = None
    drug_std: np.ndarray | None = None
    solvent_median: np.ndarray | None = None
    solvent_mean: np.ndarray | None = None
    solvent_std: np.ndarray | None = None
    temperature_mean: np.ndarray | None = None
    temperature_std: np.ndarray | None = None
    n_descriptors: int = len(DESCRIPTOR_NAMES)

    @staticmethod
    def _fit_continuous(train: np.ndarray, n_descriptors: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """在训练数据上学习连续特征的缺失值中位数、均值和标准差。"""
        continuous = np.asarray(train[:, :n_descriptors], dtype=np.float64)
        median = np.nanmedian(continuous, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isfinite(continuous), continuous, median)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        # 常数列标准差为 0，会除以 0；统一改成 1 相当于保留常数列。
        std = np.where(std > 1e-8, std, 1.0)
        return median.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)

    @staticmethod
    def _transform_continuous(
        values: np.ndarray,
        median: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """使用训练集统计量标准化连续描述符，同时保留指纹的 0/1 形式。"""
        values = np.asarray(values, dtype=np.float32).copy()
        descriptors = values[:, : len(median)].astype(np.float64)
        descriptors = np.where(np.isfinite(descriptors), descriptors, median)
        descriptors = (descriptors - mean) / std
        # 少数 RDKit 描述符的尺度差异很大，标准化后可能产生极端值。
        # 将连续描述符裁切到 [-10, 10]，只影响数值稳定性，不改动指纹。
        descriptors = np.clip(descriptors, -10.0, 10.0)
        values[:, : len(median)] = descriptors.astype(np.float32)
        return values

    def fit(
        self,
        drug: np.ndarray,
        solvent: np.ndarray,
        temperature: np.ndarray,
    ) -> "FeatureTransformer":
        """在给定训练数据上拟合药物、溶剂和温度的标准化参数。"""
        (
            self.drug_median,
            self.drug_mean,
            self.drug_std,
        ) = self._fit_continuous(drug, self.n_descriptors)
        (
            self.solvent_median,
            self.solvent_mean,
            self.solvent_std,
        ) = self._fit_continuous(solvent, self.n_descriptors)
        temperature = np.asarray(temperature, dtype=np.float64)
        self.temperature_mean = temperature.mean(axis=0).astype(np.float32)
        temperature_std = temperature.std(axis=0)
        self.temperature_std = np.where(temperature_std > 1e-8, temperature_std, 1.0).astype(
            np.float32
        )
        return self

    def transform(
        self,
        drug: np.ndarray,
        solvent: np.ndarray,
        temperature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把原始三组特征转换为标准化后的三组模型输入。"""
        if self.drug_mean is None or self.solvent_mean is None or self.temperature_mean is None:
            raise RuntimeError("FeatureTransformer must be fitted before transform")
        drug_scaled = self._transform_continuous(
            drug, self.drug_median, self.drug_mean, self.drug_std
        )
        solvent_scaled = self._transform_continuous(
            solvent, self.solvent_median, self.solvent_mean, self.solvent_std
        )
        temperature = np.asarray(temperature, dtype=np.float32)
        temperature_scaled = (temperature - self.temperature_mean) / self.temperature_std
        return drug_scaled, solvent_scaled, temperature_scaled.astype(np.float32)

    def fit_transform(
        self,
        drug: np.ndarray,
        solvent: np.ndarray,
        temperature: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """拟合并立即转换，常用于每个训练折的训练集。"""
        return self.fit(drug, solvent, temperature).transform(drug, solvent, temperature)


def build_feature_arrays(frame, transformer: FeatureTransformer | None = None):
    """把完整清洗表转换成 numpy 数组。

    drug 和 solvent 的每一行分别对应一种分子特征；
    temperature 的每一行包含三个物理温度特征；
    groups 保存药物结构组名，供后续分组交叉验证使用。
    """
    import pandas as pd

    from .config import DRUG_SMILES_COLUMN, SOLVENT_SMILES_COLUMN, TEMPERATURE_COLUMN

    # 同一药物或溶剂会在很多行重复出现，用缓存避免反复计算 RDKit 特征。
    drug_cache: dict[str, np.ndarray] = {}
    solvent_cache: dict[str, np.ndarray] = {}

    def lookup(smiles: str, cache: dict[str, np.ndarray]) -> np.ndarray:
        if smiles not in cache:
            cache[smiles] = featurize_smiles_combined(smiles)
        return cache[smiles]

    drug = np.stack(
        [lookup(value, drug_cache) for value in frame[DRUG_SMILES_COLUMN].astype(str)]
    ).astype(np.float32)
    solvent = np.stack(
        [lookup(value, solvent_cache) for value in frame[SOLVENT_SMILES_COLUMN].astype(str)]
    ).astype(np.float32)
    temperature = temperature_features(frame[TEMPERATURE_COLUMN].to_numpy(dtype=float))
    target = frame["target"].to_numpy(dtype=np.float32)
    groups = frame["structure_group"].astype(str).to_numpy()

    if transformer is not None:
        # 注意：这里传入的 transformer 应该已经在当前训练折上 fit。
        drug, solvent, temperature = transformer.transform(drug, solvent, temperature)

    return {
        "drug": drug,
        "solvent": solvent,
        "temperature": temperature,
        "target": target,
        "groups": groups,
        "feature_names": feature_names(),
        "drug_cache": drug_cache,
        "solvent_cache": solvent_cache,
    }
