"""模型加载、特征转换、预测和适用范围判断。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .config import FINGERPRINT_BITS, MODEL_DIR, TEMPERATURE_COLUMN
from .features import (
    FeatureTransformer,
    combine_smiles_features,
    featurize_smiles,
    temperature_features,
)
from .models import build_model


class SolubilityPredictor:
    """供 Streamlit 或脚本调用的统一推理接口。"""

    def __init__(self, model_dir: Path | str = MODEL_DIR) -> None:
        """加载元数据、特征转换器、集成模型和训练集参考指纹。"""
        self.model_dir = Path(model_dir)
        self.metadata = json.loads(
            (self.model_dir / "model_metadata.json").read_text(encoding="utf-8")
        )
        self.transformer: FeatureTransformer = joblib.load(
            self.model_dir / "feature_transformer.joblib"
        )
        self.target_mean = float(self.metadata["target_mean"])
        self.target_std = float(self.metadata["target_std"])
        self.engine = self.metadata["engine"]
        self.models: list[torch.nn.Module] = []
        self.sklearn_model = None

        if self.engine == "torch":
            config = self.metadata["model_config"]
            # 一个 checkpoint 对应一个随机种子。预测时把它们的输出聚合起来。
            for path in sorted(self.model_dir.glob("ensemble_seed_*.pt")):
                model = build_model(
                    name=config["model_type"],
                    drug_dim=int(self.metadata["drug_dim"]),
                    solvent_dim=int(self.metadata["solvent_dim"]),
                    hidden_dim=int(config["hidden_dim"]),
                    dropout=float(config["dropout"]),
                )
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                model.eval()
                self.models.append(model)
            if not self.models:
                raise FileNotFoundError("No Torch ensemble checkpoints found")
        elif self.engine == "sklearn":
            self.sklearn_model = joblib.load(self.model_dir / "sklearn_model.joblib")
        else:
            raise ValueError(f"Unsupported model engine: {self.engine}")

        references = np.load(self.model_dir / "reference_fingerprints.npz")
        # 这两个数组只用于判断“输入分子是否接近训练集”，不参与模型预测。
        self.reference_drug_fingerprints = references["drug_fingerprints"]
        self.reference_solvent_fingerprints = references["solvent_fingerprints"]

    def _raw_flat_inputs(
        self,
        drug_smiles: list[str],
        solvent_smiles: list[str],
        temperature_celsius: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """把 SMILES 和温度转换为尚未标准化的三组模型输入。"""
        drug_values = []
        solvent_values = []
        drug_fingerprints = []
        solvent_fingerprints = []
        for smiles in drug_smiles:
            descriptors, fingerprint = featurize_smiles(smiles)
            drug_values.append(combine_smiles_features(descriptors, fingerprint))
            drug_fingerprints.append(fingerprint)
        for smiles in solvent_smiles:
            descriptors, fingerprint = featurize_smiles(smiles)
            solvent_values.append(combine_smiles_features(descriptors, fingerprint))
            solvent_fingerprints.append(fingerprint)
        return (
            np.stack(drug_values).astype(np.float32),
            np.stack(solvent_values).astype(np.float32),
            temperature_features(temperature_celsius),
        )

    def predict(
        self,
        drug_smiles: str | list[str],
        solvent_smiles: str | list[str],
        temperature_celsius: float | list[float] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        """预测一个标量条件或一组对齐条件。

        如果只传入一个药物、一个溶剂或一个温度，会自动广播到其他输入长度，
        因此可以直接用于“一个药物 × 多种溶剂”的批量排序。
        """
        if isinstance(drug_smiles, str):
            drug_list = [drug_smiles]
        else:
            drug_list = drug_smiles
        if isinstance(solvent_smiles, str):
            solvent_list = [solvent_smiles]
        else:
            solvent_list = solvent_smiles
        temperatures = np.atleast_1d(np.asarray(temperature_celsius, dtype=float))

        length = max(len(drug_list), len(solvent_list), len(temperatures))
        # 单值广播：例如固定药物和温度，只改变溶剂列表。
        if len(drug_list) == 1:
            drug_list = drug_list * length
        if len(solvent_list) == 1:
            solvent_list = solvent_list * length
        if len(temperatures) == 1:
            temperatures = np.repeat(temperatures, length)
        if not (len(drug_list) == len(solvent_list) == len(temperatures) == length):
            raise ValueError("Drug, solvent, and temperature inputs must have matching lengths")

        drug, solvent, temperature = self._raw_flat_inputs(
            drug_list, solvent_list, temperatures
        )
        drug, solvent, temperature = self.transformer.transform(drug, solvent, temperature)

        if self.engine == "torch":
            predictions = []
            with torch.no_grad():
                drug_tensor = torch.from_numpy(drug)
                solvent_tensor = torch.from_numpy(solvent)
                temperature_tensor = torch.from_numpy(temperature)
                for model in self.models:
                    # 网络输出的是标准化目标，需要乘回标准差并加回均值。
                    scaled = model(drug_tensor, solvent_tensor, temperature_tensor).numpy()
                    predictions.append(scaled * self.target_std + self.target_mean)
            prediction_matrix = np.stack(predictions)
            # 集成均值作为最终预测，集成标准差作为模型间不确定性的粗略提示。
            mean = prediction_matrix.mean(axis=0)
            std = prediction_matrix.std(axis=0)
        else:
            flat = np.concatenate([drug, solvent, temperature], axis=1)
            scaled = self.sklearn_model.predict(flat)
            mean = scaled * self.target_std + self.target_mean
            std = np.zeros_like(mean)

        return {
            "log10_solubility": mean,
            "ensemble_std": std,
            # 目标本身是 log10(mol/L)，因此可直接反变换回 mol/L。
            "concentration_mol_l": np.power(10.0, mean),
            "drug_smiles": np.asarray(drug_list),
            "solvent_smiles": np.asarray(solvent_list),
            "temperature": temperatures,
        }

    def applicability(
        self,
        drug_smiles: str,
        solvent_smiles: str,
        temperature_celsius: float,
    ) -> dict:
        """检查输入是否接近训练数据，并返回面向用户的警告。"""
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=FINGERPRINT_BITS
        )

        def new_fp(smiles: str):
            mol = Chem.MolFromSmiles(smiles)
            return None if mol is None else generator.GetFingerprint(mol)

        def max_similarity(smiles: str, references: np.ndarray) -> float:
            """计算输入指纹与训练集所有指纹的最大 Tanimoto 相似度。"""
            fp = new_fp(smiles)
            if fp is None or len(references) == 0:
                return 0.0
            query = np.asarray(fp, dtype=np.float32)
            intersection = (references.astype(bool) & query.astype(bool)).sum(axis=1)
            union = (references.astype(bool) | query.astype(bool)).sum(axis=1)
            scores = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection, dtype=float),
                where=union > 0,
            )
            return float(scores.max())

        drug_similarity = max_similarity(
            drug_smiles, self.reference_drug_fingerprints
        )
        solvent_similarity = max_similarity(
            solvent_smiles, self.reference_solvent_fingerprints
        )
        temp_min = float(self.metadata["temperature_range"][0])
        temp_max = float(self.metadata["temperature_range"][1])
        warnings = []
        # 这些阈值是工程上的保守提示，不是统计学意义上的置信区间。
        if drug_similarity < 0.30:
            warnings.append("Drug is structurally distant from the training set.")
        if solvent_similarity < 0.30:
            warnings.append("Solvent is structurally distant from the training set.")
        if temperature_celsius < 5 or temperature_celsius > 60:
            warnings.append("Temperature is outside the well-supported 5–60 °C range.")
        if temperature_celsius < temp_min or temperature_celsius > temp_max:
            warnings.append("Temperature is outside the observed training range.")

        return {
            "drug_similarity": drug_similarity,
            "solvent_similarity": solvent_similarity,
            "temperature_in_supported_range": 5 <= temperature_celsius <= 60,
            "is_warning": bool(warnings),
            "warnings": warnings,
        }


def known_options(model_dir: Path | str = MODEL_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取应用下拉框使用的已知药物和已知溶剂清单。"""
    model_dir = Path(model_dir)
    return (
        pd.read_csv(model_dir / "known_drugs.csv"),
        pd.read_csv(model_dir / "known_solvents.csv"),
    )
