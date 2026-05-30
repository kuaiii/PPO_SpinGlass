"""
注意力耦合层：从节点嵌入 Z 计算耦合矩阵 J。

J_{ij} = softmax( (W_Q z_i)^T (W_K z_j) / sqrt(d_k) )
"""

from typing import Optional
import math
import torch
import torch.nn as nn


class AttentionCoupling(nn.Module):
    """
    使用 Scaled Dot-Product Attention 计算自旋玻璃耦合矩阵 J。

    输出 J 满足：
      - J_{ij} >= 0
      - diag(J) = 0
      - 对行做 softmax（或全局 softmax 后去对角线）
    """

    def __init__(self, hidden_dim: int = 64, top_k: Optional[int] = None) -> None:
        """
        Args:
            hidden_dim: 节点嵌入维度。
            top_k: 大图稀疏化时只保留 Top-K 耦合，None 表示不稀疏化。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.d_k = hidden_dim

        self.W_Q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_K = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        计算耦合矩阵 J。

        Args:
            Z: 节点嵌入，shape (n, hidden_dim)。

        Returns:
            J: 耦合矩阵，shape (n, n)。
               对角线置 0，每行 softmax 归一化。
        """
        n = Z.shape[0]
        Q = self.W_Q(Z)  # (n, hidden_dim)
        K = self.W_K(Z)  # (n, hidden_dim)

        # Scaled dot-product
        scores = torch.matmul(Q, K.t()) / math.sqrt(self.d_k)  # (n, n)

        # 掩码对角线为 -inf（不允许自耦合）
        mask = torch.eye(n, device=Z.device, dtype=torch.bool)
        scores = scores.masked_fill(mask, float('-inf'))

        # Top-K 稀疏化（大图时）
        if self.top_k is not None and self.top_k < n - 1:
            # 对每行保留 top_k 个最大值
            vals, indices = torch.topk(scores, k=min(self.top_k, n - 1), dim=-1)
            # 构建稀疏掩码
            sparse_mask = torch.full_like(scores, float('-inf'))
            sparse_mask.scatter_(-1, indices, 0.0)
            scores = scores + sparse_mask

        # 按行 softmax
        J = torch.softmax(scores, dim=-1)

        # 再次确保对角线为 0（softmax 后 -inf 对应位置为 0）
        J = J.masked_fill(mask, 0.0)

        return J

    def reset_parameters(self) -> None:
        """重置参数。"""
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
