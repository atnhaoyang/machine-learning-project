"""项目级配置。

所有路径、字段名、随机种子和训练超参数都集中放在这里，避免在多个模块中
重复写字符串或魔法数字。阅读其他模块时，遇到这些常量可以回到本文件查看。
"""

from pathlib import Path


# 项目根目录是 solubility/ 的上一级目录。
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data_2026.csv"

# 训练、评估和应用使用的所有生成文件都放在 outputs/ 下。
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model"
PREDICTION_DIR = OUTPUT_DIR / "predictions"

CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_data.csv"
AUDIT_PATH = OUTPUT_DIR / "data_audit.json"
SPLIT_PATH = OUTPUT_DIR / "group_folds.csv"
METRICS_DETAIL_PATH = OUTPUT_DIR / "metrics_detail.csv"
METRICS_SUMMARY_PATH = OUTPUT_DIR / "metrics_summary.csv"
EXPERIMENT_LOG_PATH = OUTPUT_DIR / "experiment_log.json"

TARGET_COLUMN = "Solubility_log10"
TEMPERATURE_COLUMN = "Temperature"
DRUG_SMILES_COLUMN = "Drug SMILES"
SOLVENT_SMILES_COLUMN = "Solvent SMILES"

# 固定随机种子，保证数据划分和神经网络初始化可以复现。
SEED = 42
N_SPLITS = 5
ENSEMBLE_SEEDS = (11, 22, 33, 44, 55)

# 训练控制参数：
# MAX_EPOCHS 是每个模型最多训练的轮数；
# PATIENCE 是验证损失连续多少轮不改善后停止；
# BATCH_SIZE 是每次送入网络的条件数量。
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 14
BATCH_SIZE = 128

# Morgan 指纹设置。半径 2 表示考虑中心原子附近两跳范围内的子结构。
FINGERPRINT_BITS = 512
MORGAN_RADIUS = 2


def ensure_output_dirs() -> None:
    """创建程序运行时需要的输出目录；已存在的目录不会被覆盖。"""
    for path in (OUTPUT_DIR, FIGURE_DIR, MODEL_DIR, PREDICTION_DIR):
        path.mkdir(parents=True, exist_ok=True)
