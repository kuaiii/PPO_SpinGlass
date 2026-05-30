"""
值函数头：全局 readout + 图级特征，输出状态价值 V(s)。
"""

from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Data


class ValueHead(nn.Module):
    """
    状态值函数头。

    输入节点嵌入 Z 和图级特征，输出标量 V(s)。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mlp_hidden: int = 128,
        num_graph_features: int = 4,
    ) -> None:
        """
        Args:
            hidden_dim: 节点嵌入维度。
            mlp_hidden: MLP 隐藏层维度。
            num_graph_features: 图级特征数（边数、LCC、λ₂、H）。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_graph_features = num_graph_features
        in_dim = hidden_dim + num_graph_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        Z: torch.Tensor,
        data: Data,
        graph_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算状态价值 V(s)。

        Args:
            Z: 节点嵌入 (n, hidden_dim)。
            data: PyG Data 对象，用于 batch 索引（单图时 batch 为 None）。
            graph_features: 图级特征 (num_graph_features,)，
                            若 None 则默认用零向量填充。

        Returns:
            V(s) 标量值。
        """
        # 全局 readout
        if hasattr(data, "batch") and data.batch is not None:
            global_feat = global_mean_pool(Z, data.batch)  # (batch_size, hidden_dim)
        else:
            global_feat = Z.mean(dim=0, keepdim=True)  # (1, hidden_dim)

        if graph_features is None:
            graph_features = torch.zeros(
                1, self.num_graph_features,
                device=Z.device,
                dtype=Z.dtype,
            )
        else:
            if graph_features.dim() == 1:
                graph_features = graph_features.unsqueeze(0)

        combined = torch.cat([global_feat, graph_features], dim=-1)  # (1, hidden_dim + num_features)
        value = self.mlp(combined).squeeze(-1)  # scalar or (batch,)
        return value

    def reset_parameters(self) -> None:
        """重置参数。"""
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
