"""Streamlit 演示应用。

页面只负责交互、展示和调用推理接口，核心特征与模型逻辑位于
solubility/ 包中。阅读顺序建议：
config.py -> inference.py -> 本文件的预测和溶剂筛选部分。
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from solubility.config import (
    AUDIT_PATH,
    CLEANED_DATA_PATH,
    FIGURE_DIR,
    METRICS_SUMMARY_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
)
from solubility.inference import SolubilityPredictor, known_options


ROOT_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="SolventScout | 药物溶解度预测",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# 简洁主题覆盖：统一浅色背景、低对比边框和单一青绿色强调色。
st.markdown(
    """
    <style>
    :root {
        --bg: #f8fafc;
        --surface: #ffffff;
        --text: #1f2937;
        --muted: #667085;
        --border: #e4e7ec;
        --accent: #0f766e;
        --accent-soft: #eef7f5;
        --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
        --app-header-height: 60px;
    }

    .stApp,
    div[data-testid="stAppViewContainer"],
    main {
        background: var(--bg) !important;
        color: var(--text) !important;
    }

    html,
    body {
        height: auto !important;
        min-height: 100% !important;
        overflow-y: auto !important;
    }

    .stApp,
    div[data-testid="stAppViewContainer"] {
        min-height: 100vh !important;
        height: auto !important;
        overflow: visible !important;
    }

    main[data-testid="stMain"] {
        min-height: calc(100vh - var(--app-header-height)) !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }

    main[data-testid="stMain"]::-webkit-scrollbar {
        width: 10px;
    }

    main[data-testid="stMain"]::-webkit-scrollbar-track {
        background: transparent;
    }

    main[data-testid="stMain"]::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border: 3px solid var(--bg);
        border-radius: 999px;
    }

    header[data-testid="stHeader"] {
        background: var(--bg) !important;
        border: 0 !important;
        height: var(--app-header-height) !important;
    }

    header[data-testid="stHeader"]::before,
    header[data-testid="stHeader"]::after {
        content: none !important;
        display: none !important;
    }

    .block-container {
        max-width: 1120px !important;
        min-height: auto !important;
        padding: 1.75rem 2.25rem 7rem !important;
    }

    html,
    body,
    [class*="css"],
    html body [data-testid="stMain"] p,
    html body [data-testid="stMain"] label,
    html body [data-testid="stMain"] li {
        font-family: var(--font-ui) !important;
        color: var(--text);
    }

    h1,
    h2,
    h3,
    [data-testid="stMain"] h3 {
        color: var(--text) !important;
        font-family: var(--font-ui) !important;
        letter-spacing: -0.02em !important;
    }

    h1 { font-size: 1.9rem !important; line-height: 1.2 !important; }
    h2 { font-size: 1.45rem !important; line-height: 1.25 !important; }
    h3,
    [data-testid="stMain"] h3 {
        font-size: 1.1rem !important;
        font-weight: 650 !important;
        line-height: 1.35 !important;
    }

    .hero {
        border: 0 !important;
        padding: 0.5rem 0 0.75rem !important;
        margin: 0 0 1.25rem !important;
    }

    .eyebrow {
        color: var(--accent) !important;
        font-size: 0.72rem !important;
        font-weight: 650 !important;
        letter-spacing: 0.08em !important;
    }

    .hero-title {
        color: var(--text) !important;
        font-family: var(--font-ui) !important;
        font-size: clamp(1.75rem, 3.2vw, 2.35rem) !important;
        font-weight: 720 !important;
        line-height: 1.18 !important;
        letter-spacing: -0.035em !important;
        max-width: 760px !important;
        margin: 0.4rem 0 0.65rem !important;
    }

    .hero-copy {
        color: var(--muted) !important;
        font-size: 0.94rem !important;
        line-height: 1.55 !important;
        max-width: 760px !important;
    }

    .metric-grid {
        display: grid !important;
        grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
        margin: 1rem 0 1.25rem !important;
    }

    .metric-grid.prediction-metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }

    .metric-card {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    .metric-label {
        color: var(--muted) !important;
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.045em !important;
        overflow-wrap: normal !important;
        word-break: keep-all !important;
    }

    .metric-value {
        color: var(--text) !important;
        font-family: var(--font-ui) !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        margin-top: 0.25rem !important;
    }

    .metric-note {
        color: var(--muted) !important;
        font-size: 0.76rem !important;
        line-height: 1.4 !important;
        margin-top: 0.2rem !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        width: 100% !important;
        padding: 0 !important;
        gap: 1.4rem !important;
        background: transparent !important;
        border-bottom: 1px solid var(--border) !important;
        border-radius: 0 !important;
    }

    div[data-testid="stTabs"] [role="tab"] {
        height: 2.6rem !important;
        min-height: 2.6rem !important;
        padding: 0 !important;
        color: var(--muted) !important;
        background: transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: var(--accent) !important;
    }

    div[data-testid="stTabs"] .react-aria-SelectionIndicator {
        background: var(--accent) !important;
        height: 2px !important;
    }

    .stButton > button {
        min-height: 2.25rem !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        background: var(--surface) !important;
        color: var(--text) !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        border-color: #cfd6dd !important;
        background: #f9fafb !important;
    }

    .stButton > button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #ffffff !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: #0b5f59 !important;
        border-color: #0b5f59 !important;
    }

    div[data-baseweb="select"] > div,
    .stSelectbox [role="group"],
    .stTextInput input,
    .stNumberInput input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.9rem !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    .stSelectbox [role="group"]:focus-within,
    .stTextInput input:focus,
    .stNumberInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .stSelectbox [role="combobox"],
    .stSelectbox [role="group"] button,
    .stSelectbox [role="group"] svg {
        color: var(--text) !important;
        fill: var(--text) !important;
    }

    .status-ok,
    .status-warn {
        border-radius: 8px !important;
        padding: 0.75rem 0.9rem !important;
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
        box-shadow: none !important;
    }

    .status-ok {
        background: var(--accent-soft) !important;
        border: 1px solid #cfe8e2 !important;
        color: #285f58 !important;
    }

    .status-warn {
        background: #fffaf0 !important;
        border: 1px solid #f1dfb5 !important;
        color: #7a5b19 !important;
    }

    .small-note {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        line-height: 1.55 !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        background: var(--surface) !important;
    }

    [data-testid="stVegaLiteChart"],
    [data-testid="stVegaLiteChart"] > div,
    [data-testid="stVegaLiteChart"] .vega-embed {
        background: var(--surface) !important;
        border-radius: 8px !important;
    }

    [data-testid="stAlert"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    [data-testid="stAlert"] .stAlertContainer,
    [data-testid="stAlert"] [role="status"] {
        background: #f5f7fa !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
    }

    [data-testid="stAlert"] [role="status"] p,
    [data-testid="stAlert"] [role="status"] div {
        color: var(--text) !important;
    }

    [data-testid="stAlert"] [role="status"] svg {
        color: var(--accent) !important;
        fill: var(--accent) !important;
    }

    a {
        color: var(--accent) !important;
    }

    @media (max-width: 780px) {
        .block-container {
            padding: 1.25rem 1rem 6rem !important;
        }
        .metric-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        }
        .hero-title {
            font-size: 1.75rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="正在加载模型…")
def load_predictor() -> SolubilityPredictor:
    """模型加载较慢，用 cache_resource 在页面会话期间复用同一个对象。"""
    return SolubilityPredictor()


@st.cache_data
def load_options() -> tuple[pd.DataFrame, pd.DataFrame]:
    """缓存药物和溶剂下拉列表；CSV 未变化时不会重复读取。"""
    return known_options()


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """读取训练脚本生成的模型对比表。"""
    return pd.read_csv(METRICS_SUMMARY_PATH)


@st.cache_data
def load_metadata() -> dict:
    """读取最优模型版本、指标和适用范围。"""
    return json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))


@st.cache_data
def load_audit() -> dict:
    """读取数据规模和重复测量审计结果。"""
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_cleaned_data() -> pd.DataFrame:
    """溶剂排序时用清洗表统计当前温度附近有多少训练条件。"""
    return pd.read_csv(CLEANED_DATA_PATH)


st.markdown(
    """
    <div class="hero">
        <div class="hero-title">在实验之前，先找到更值得尝试的溶剂条</div>
        <div class="hero-copy">
            模型仅在当前实验数据覆盖范围内提供筛选建议。
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    # 如果训练脚本还没运行，这些文件不存在；页面会给出明确提示并停止渲染。
    predictor = load_predictor()
    known_drugs, known_solvents = load_options()
    metrics = load_metrics()
    metadata = load_metadata()
    audit = load_audit()
except Exception as error:
    st.error(
        "模型文件尚未生成或无法加载。请先运行 "
        "`conda run --no-capture-output -n pytorch python scripts/run_experiments.py`。"
    )
    st.exception(error)
    st.stop()

best_metrics = metadata["metrics"]

# 页面顶部四个指标卡展示最终分组验证结果。
summary_cards = f"""
<div class="metric-grid">
    <div class="metric-card">
        <div class="metric-label">MAE</div>
        <div class="metric-value">{best_metrics["mae_mean"]:.3f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">R²</div>
        <div class="metric-value">{best_metrics["r2_mean"]:.3f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Rank correlation</div>
        <div class="metric-value">{best_metrics["spearman_mean"]:.3f}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">±0.5 hit rate</div>
        <div class="metric-value">{best_metrics["tolerance_accuracy_mean"]:.1%}</div>
    </div>
</div>
"""
st.markdown(summary_cards, unsafe_allow_html=True)

prediction_tab, ranking_tab, evaluation_tab, method_tab = st.tabs(
    ["单条件预测", "溶剂筛选", "模型评估", "方法与限制"]
)

# ---------------- 标签页 1：单条件预测 ----------------
with prediction_tab:
    st.subheader("预测一个实验条件")
    input_col, result_col = st.columns([0.95, 1.05], gap="large")

    with input_col:
        use_custom_drug = st.toggle("输入新的药物 SMILES", value=False)
        if use_custom_drug:
            drug_smiles = st.text_input(
                "药物 SMILES",
                value=known_drugs.iloc[0]["smiles"],
                help="支持 RDKit 可解析的 SMILES；盐型会按原结构保留。",
            )
            drug_label = "自定义药物"
        else:
            drug_options = known_drugs.reset_index(drop=True)
            drug_index = st.selectbox(
                "药物",
                options=drug_options.index,
                format_func=lambda index: (
                    f"{drug_options.loc[index, 'label']} "
                    f"· {int(drug_options.loc[index, 'drug_conditions'])} 条条件"
                ),
            )
            drug_smiles = str(drug_options.loc[drug_index, "smiles"])
            drug_label = str(drug_options.loc[drug_index, "label"])

        use_custom_solvent = st.toggle("输入新的溶剂 SMILES", value=False)
        if use_custom_solvent:
            solvent_smiles = st.text_input(
                "溶剂 SMILES",
                value=known_solvents.iloc[0]["smiles"],
            )
            solvent_label = "自定义溶剂"
        else:
            solvent_options = known_solvents.reset_index(drop=True)
            solvent_index = st.selectbox(
                "溶剂",
                options=solvent_options.index,
                format_func=lambda index: (
                    f"{solvent_options.loc[index, 'label']} "
                    f"· {int(solvent_options.loc[index, 'solvent_conditions'])} 条条件"
                ),
            )
            solvent_smiles = str(solvent_options.loc[solvent_index, "smiles"])
            solvent_label = str(solvent_options.loc[solvent_index, "label"])

        temperature = st.slider(
            "温度（°C）",
            min_value=-60.0,
            max_value=130.0,
            value=25.0,
            step=0.5,
        )
        predict_clicked = st.button("运行预测", type="primary", width="stretch")

    with result_col:
        if predict_clicked:
            try:
                # 1) 模型预测 log10 溶解度；2) 检查输入是否远离训练数据。
                result = predictor.predict(drug_smiles, solvent_smiles, temperature)
                applicability = predictor.applicability(
                    drug_smiles, solvent_smiles, temperature
                )
                log10_value = float(result["log10_solubility"][0])
                concentration = float(result["concentration_mol_l"][0])
                ensemble_std = float(result["ensemble_std"][0])
                band = float(metadata["residual_q90"])

                # log10 空间加减误差带后，再反变换为 mol/L，因此浓度区间不是对称的。
                lower_concentration = 10 ** (log10_value - band)
                upper_concentration = 10 ** (log10_value + band)

                st.markdown(
                    f"""
                    <div class="metric-grid prediction-metrics">
                        <div class="metric-card">
                            <div class="metric-label">Predicted log10 solubility</div>
                            <div class="metric-value">{log10_value:.2f}</div>
                            <div class="metric-note">±{band:.2f} empirical 90% band</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Estimated concentration</div>
                            <div class="metric-value">{concentration:.3g}</div>
                            <div class="metric-note">mol/L · range {lower_concentration:.3g}–{upper_concentration:.3g}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{drug_label} · {solvent_label} · {temperature:.1f} °C · "
                    f"ensemble SD {ensemble_std:.3f}"
                )

                if applicability["is_warning"]:
                    # 输入超出适用域时仍然显示预测，但明确降低结论强度。
                    warning_text = "<br>".join(applicability["warnings"])
                    st.markdown(
                        f'<div class="status-warn"><strong>低置信度提示</strong><br>{warning_text}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="status-ok"><strong>输入位于已验证范围。</strong></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<p class="small-note">药物最大训练集相似度：'
                    f'{applicability["drug_similarity"]:.2f}；溶剂最大训练集相似度：'
                    f'{applicability["solvent_similarity"]:.2f}。</p>',
                    unsafe_allow_html=True,
                )
            except Exception as error:
                st.error(f"无法完成预测：{error}")
        else:
            st.markdown(
                """
                <div class="status-ok">
                    选择药物、溶剂和温度后运行预测。结果会同时显示 log10 溶解度、
                    mol/L 浓度、集成模型波动和经验误差范围。
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p class="small-note">
                预测区间来自分组交叉验证中绝对残差的 90% 分位数，用于提示历史误差量级，
                不构成严格的置信区间保证。
                </p>
                """,
                unsafe_allow_html=True,
            )

# ---------------- 标签页 2：溶剂筛选 ----------------
with ranking_tab:
    st.subheader("为固定条件筛选候选溶剂")
    rank_col_1, rank_col_2 = st.columns([1.15, 0.85], gap="large")
    with rank_col_1:
        ranking_drug_options = known_drugs.reset_index(drop=True)
        ranking_drug_index = st.selectbox(
            "待筛查药物",
            options=ranking_drug_options.index,
            format_func=lambda index: (
                f"{ranking_drug_options.loc[index, 'label']} "
                f"· {int(ranking_drug_options.loc[index, 'drug_conditions'])} 条条件"
            ),
            key="ranking_drug",
        )
        ranking_temperature = st.slider(
            "筛选温度（°C）",
            min_value=5.0,
            max_value=60.0,
            value=25.0,
            step=0.5,
            key="ranking_temperature",
        )
        run_ranking = st.button("生成前 10 个候选", type="primary", width="stretch")
    with rank_col_2:
        st.markdown(
            """
            <div class="status-ok">
            排序依据是预测 log10 溶解度，不包含实验成本、毒性、沸点或法规限制。
            表中的训练记录数用于识别数据覆盖较少的溶剂。
            </div>
            """,
            unsafe_allow_html=True,
        )

    if run_ranking:
        # 固定一个药物和温度，对所有已知溶剂批量预测。
        cleaned_data = load_cleaned_data()
        ranking_drug_smiles = str(
            ranking_drug_options.loc[ranking_drug_index, "smiles"]
        )
        result = predictor.predict(
            ranking_drug_smiles,
            known_solvents["smiles"].tolist(),
            ranking_temperature,
        )
        ranking = known_solvents.copy()
        ranking["predicted_log10"] = result["log10_solubility"]
        ranking["ensemble_std"] = result["ensemble_std"]
        ranking["concentration_mol_l"] = result["concentration_mol_l"]
        nearby_counts = (
            cleaned_data.loc[
                (cleaned_data["Temperature"] - ranking_temperature).abs() <= 5
            ]
            .groupby("Solvent SMILES")
            .size()
            .rename("nearby_conditions")
        )
        # 除总训练量外，再显示±5°C范围的条件数，帮助识别局部数据不足。
        ranking = ranking.merge(
            nearby_counts,
            left_on="smiles",
            right_index=True,
            how="left",
        )
        ranking["nearby_conditions"] = (
            ranking["nearby_conditions"].fillna(0).astype(int)
        )
        ranking = ranking.sort_values("predicted_log10", ascending=False).head(10)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

        # Altair 只负责把前 10 名画成排序条形图。
        chart = (
            alt.Chart(ranking)
            .mark_bar(cornerRadiusEnd=5, color="#2a9d8f")
            .encode(
                x=alt.X("predicted_log10:Q", title="Predicted log10 solubility"),
                y=alt.Y("label:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("rank:Q", title="Rank"),
                    alt.Tooltip("label:N", title="Solvent"),
                    alt.Tooltip("predicted_log10:Q", title="log10 S", format=".3f"),
                    alt.Tooltip("concentration_mol_l:Q", title="mol/L", format=".4g"),
                    alt.Tooltip("solvent_conditions:Q", title="Training conditions"),
                    alt.Tooltip("nearby_conditions:Q", title="Conditions within ±5 °C"),
                ],
            )
            .properties(height=360)
            .configure_view(fill="#FFFFFF", stroke=None)
            .configure(background="#FFFFFF")
        )
        st.altair_chart(chart, width="stretch")
        table = ranking[
            [
                "rank",
                "label",
                "predicted_log10",
                "concentration_mol_l",
                "ensemble_std",
                "solvent_conditions",
                "nearby_conditions",
            ]
        ].rename(
            columns={
                "rank": "排名",
                "label": "溶剂",
                "predicted_log10": "预测 log10 S",
                "concentration_mol_l": "估算浓度 (mol/L)",
                "ensemble_std": "集成标准差",
                "solvent_conditions": "总训练条件数",
                "nearby_conditions": "±5°C 训练条件数",
            }
        )
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.info("选择药物和温度后生成候选溶剂排名。")

# ---------------- 标签页 3：模型评估 ----------------
with evaluation_tab:
    st.subheader("分组验证结果")
    display_metrics = metrics.rename(
        columns={
            "experiment": "实验",
            "mae_mean": "MAE",
            "mae_std": "MAE 标准差",
            "rmse_mean": "RMSE",
            "r2_mean": "R²",
            "spearman_mean": "Spearman",
            "tolerance_accuracy_mean": "±0.5 命中率",
            "top3_solvent_accuracy_mean": "Top-3 溶剂命中率",
            "mean_best_epoch": "平均最佳 epoch",
        }
    )
    st.dataframe(
        display_metrics[
            [
                "实验",
                "MAE",
                "MAE 标准差",
                "RMSE",
                "R²",
                "Spearman",
                "±0.5 命中率",
                "Top-3 溶剂命中率",
                "平均最佳 epoch",
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "MAE 标准差": st.column_config.NumberColumn(format="%.3f"),
            "RMSE": st.column_config.NumberColumn(format="%.3f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
            "Spearman": st.column_config.NumberColumn(format="%.3f"),
            "±0.5 命中率": st.column_config.NumberColumn(format="%.1%"),
            "Top-3 溶剂命中率": st.column_config.NumberColumn(format="%.1%"),
        },
    )

    figure_specs = [
        ("02_loss_curves.png", "训练过程与验证损失"),
        ("03_ablation_mae.png", "不同优化方案的 MAE 对比"),
        ("04_prediction_diagnostics.png", "最优模型的分组预测与残差"),
        ("05_solvent_errors.png", "数据覆盖较充分溶剂的误差"),
        ("06_temperature_residuals.png", "温度区间残差"),
    ]
    for filename, caption in figure_specs:
        # 只展示训练脚本实际生成过的图片，避免文件缺失时页面报错。
        path = FIGURE_DIR / filename
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")

# ---------------- 标签页 4：方法与限制 ----------------
with method_tab:
    st.subheader("实验设计")
    st.markdown(
        f"""
        原始数据包含 **{audit["raw_rows"]:,}** 条记录。规范化 SMILES 后，
        同一药物、溶剂和温度的重复测量被合并，最终形成
        **{audit["rows_after_condition_aggregation"]:,}** 个条件。
        数据覆盖 **{audit["unique_canonical_drug_structures"]}** 个唯一药物结构和
        **{audit["unique_solvents"]}** 种有机溶剂。

        主验证采用按规范药物结构分组的五折划分。每组药物只出现在一个测试折中，
        避免同一结构跨训练集和测试集造成数据泄漏。模型使用 RDKit 描述符、
        512 位 Morgan 指纹以及 `T`、`1000/(T+273.15)` 和 `log(T+273.15)`
        温度特征。最终模型为五个随机种子的轻量注意力融合网络。

        **主要限制**

        - 只有 168 个唯一药物结构，未见药物性能的折间波动较大。
        - 49 种溶剂中，有 21 种不超过 5 条记录，新溶剂外推不可靠。
        - 同一条件重复测量的最大差异达到
          {audit["max_replicate_range_log10"]:.2f} log 单位，实验噪声不可忽略。
        - 温湿度、压力、晶型、pH、纯度和测量方法等变量未提供。
        - 结果用于实验候选排序，不用于替代实测、安全评估或监管决策。
        """
    )
    st.markdown(
        f"""
        <p class="small-note">
        模型版本：{metadata["best_experiment"]} ·
        推理引擎：{metadata["engine"]} ·
        训练条件数：{metadata["n_training_conditions"]:,} ·
        经验 90% 误差带：±{metadata["residual_q90"]:.3f} log10。
        </p>
        """,
        unsafe_allow_html=True,
    )
