"""
TGNN 编码器：共享图神经网络编码器。

使用 3 层 GATConv（Graph Attention）提取节点嵌入。
"""

from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class TGNNEncoder(nn.Module):
    """
    基于 GATConv 的 3 层图注意力编码器。

    输入 PyG Data 对象，输出节点嵌入 Z ∈ R^{n×hidden_dim}。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            in_channels: 输入节点特征维度。
            hidden_dim: 隐藏层维度（默认 64）。
            num_layers: GAT 层数（默认 3）。
            num_heads: 注意力头数（默认 4）。
            dropout: Dropout 概率。
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim * num_heads
            # 最后一层只用 1 个头避免维度爆炸
            out_heads = 1 if i == num_layers - 1 else num_heads
            concat = i != num_layers - 1
            self.convs.append(
                GATConv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=out_heads,
                    concat=concat,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )
            if concat:
                self.bns.append(nn.BatchNorm1d(hidden_dim * out_heads))
            else:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播。

        Args:
            data: PyG Data，包含 x (节点特征) 和 edge_index (边索引)。

        Returns:
            节点嵌入 Z，shape (n, hidden_dim) 或 (n, hidden_dim * num_heads)。
            默认配置下输出 (n, hidden_dim)。
        """
        x, edge_index = data.x, data.edge_index
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = torch.relu(x)
            x = self.dropouts[i](x)
        return x

    def reset_parameters(self) -> None:
        """重置所有可学习参数。"""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
