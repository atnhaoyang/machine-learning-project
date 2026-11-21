# SolventScout：药物–溶剂–温度溶解度预测与筛选

本项目使用 PyTorch 构建有机溶剂溶解度回归模型，输入药物结构、溶剂结构和温度，预测 `log10` 溶解度（mol/L），并提供候选溶剂排序 Web 演示。

项目定位为实验前候选条件筛选，不替代实验测量或监管决策。

## 模型训练

```bash
python -m pip install -r requirements.txt
python scripts/run_experiments.py
```
## 运行应用
```bash
streamlit run app.py
```

## 数据

- 原始数据：`data_2026.csv`
- 编码：GB18030
- 原始记录：1,904 条
- 规范化并合并重复条件：1,785 条
- 唯一药物结构：168 个
- 有机溶剂：49 种
- 目标：`Solubility_log10`，单位 log10(mol/L)

RDKit 用于校验和规范化 SMILES。同一规范药物、溶剂和温度的重复测量取均值，并保留重复波动信息用于数据审计。

## 验证设计

主评测采用五折分组划分。同一规范化药物结构只出现在一个测试折中，以避免同一分子跨训练集和测试集造成数据泄漏。

模型比较包括：

- Ridge 回归；
- 梯度提升树；
- PyTorch 描述符 MLP；
- AdamW + 余弦退火；
- Dropout、weight decay、SmoothL1；
- 温度邻域和特征扰动增强；
- 药物–溶剂–温度注意力融合；
- 五个随机种子集成。

## 输出

- `outputs/cleaned_data.csv`：清洗后的建模数据；
- `outputs/data_audit.json`: 数据审计报告
- `outputs/experiment_log.json`: 完整实验运行日志
- `outputs/group_folds.csv`: 五折验证划分表
- `outputs/metrics_detail.csv`：每折详细指标；
- `outputs/metrics_summary.csv`：模型汇总指标；
- `oof_predictions.csv`: 折外预测结果
- `outputs/oof_predictions_with_context.csv`：带药物、溶剂和温度信息的折外预测；
- `outputs/figures/`：数据审计、训练曲线、消融实验和误差分析图；
- `outputs/model/`：最终模型、特征转换器、已知药物和溶剂清单；

应用支持：

- 已知药物和溶剂条件预测；
- 自定义 SMILES 预测；
- 固定药物和温度下的候选溶剂排序；
- 训练集结构相似度与适用范围警告；
- 模型指标和误差分析展示。
