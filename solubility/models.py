"""PyTorch 回归模型定义。"""

from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    """把药物、溶剂和温度直接拼接后送入多层感知机。

    这是 B1–B4 使用的普通 MLP 基线结构。
    """
    def __init__(
        self,
        drug_dim: int,
        solvent_dim: int,
        temperature_dim: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # 药物特征、溶剂特征和温度特征在最后一维直接拼接。
        input_dim = drug_dim + solvent_dim + temperature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        drug: torch.Tensor,
        solvent: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：拼接输入，输出一个标量标准化 logS。"""
        values = torch.cat([drug, solvent, temperature], dim=-1)
        return self.network(values).squeeze(-1)


class AttentionFusionRegressor(nn.Module):
    """药物、溶剂和温度三 token 的轻量注意力融合模型。

    与普通 MLP 的区别：
    1. 三组输入先分别投影到相同隐藏维度；
    2. 通过多头注意力学习三者之间的交互；
    3. 使用门控为三个 token 学习动态权重。
    """
    def __init__(
        self,
        drug_dim: int,
        solvent_dim: int,
        temperature_dim: int = 3,
        hidden_dim: int = 96,
        heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # 三个独立投影层把不同来源的特征映射到统一表示空间。
        self.drug_projection = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.solvent_projection = nn.Sequential(
            nn.Linear(solvent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.temperature_projection = nn.Sequential(
            nn.Linear(temperature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        # 注意力结果通过残差连接和 LayerNorm 回加到原 token。
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # gate 为每个 token 生成 0–1 权重，决定该来源在融合时的相对重要性。
        self.gate = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        drug: torch.Tensor,
        solvent: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        """前向传播：三路投影 -> 自注意力 -> 门控聚合 -> 回归头。"""
        tokens = torch.stack(
            [
                self.drug_projection(drug),
                self.solvent_projection(solvent),
                self.temperature_projection(temperature),
            ],
            dim=1,
        )
        # 自注意力允许药物、溶剂和温度三个 token 相互更新。
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.layer_norm(tokens + self.dropout(attended))
        # 三个 token 在求和前的权重由网络自己学习。
        gates = torch.sigmoid(self.gate(tokens))
        fused = (tokens * gates).sum(dim=1)
        return self.head(fused).squeeze(-1)


def build_model(
    name: str,
    drug_dim: int,
    solvent_dim: int,
    hidden_dim: int,
    dropout: float,
) -> nn.Module:
    """根据实验配置创建 MLP 或注意力模型。"""
    if name == "attention":
        return AttentionFusionRegressor(
            drug_dim=drug_dim,
            solvent_dim=solvent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if name == "mlp":
        return MLPRegressor(
            drug_dim=drug_dim,
            solvent_dim=solvent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise ValueError(f"Unknown model name: {name}")
