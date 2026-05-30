"""
策略头：拆解策略头与构造策略头。

- DismantlePolicyHead: 节点移除策略
- ConstructPolicyHead: 边添加策略（两阶段：物理 Top-K 预筛选 + 策略精炼）
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DismantlePolicyHead(nn.Module):
    """
    拆解策略头：输入节点嵌入、局部分子场、度数，输出节点移除概率分布。
    """

    def __init__(self, hidden_dim: int = 64, mlp_hidden: int = 128) -> None:
        """
        Args:
            hidden_dim: 节点嵌入维度。
            mlp_hidden: MLP 隐藏层维度。
        """
        super().__init__()
        # 输入特征：[z_i; h_i; deg(i)] => hidden_dim + 1 + 1
        in_dim = hidden_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(
        self,
        Z: torch.Tensor,
        local_fields: torch.Tensor,
        degrees: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算节点 logits。

        Args:
            Z: 节点嵌入 (n, hidden_dim)。
            local_fields: 局部分子场 (n,)。
            degrees: 节点度数 (n,)，已归一化或未归一化均可。
            mask: 动作掩码 (n,)，bool 型，True 表示有效动作。
                  若 None，默认所有节点有效。

        Returns:
            logits (n,)，已应用掩码。
        """
        h = local_fields.unsqueeze(-1)  # (n, 1)
        deg = degrees.unsqueeze(-1)  # (n, 1)
        features = torch.cat([Z, h, deg], dim=-1)  # (n, hidden_dim + 2)
        logits = self.mlp(features).squeeze(-1)  # (n,)

        if mask is not None:
            logits = torch.where(mask, logits, torch.tensor(-1e9, device=logits.device, dtype=logits.dtype))
        return logits

    def sample(
        self,
        Z: torch.Tensor,
        local_fields: torch.Tensor,
        degrees: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并返回 log_prob。

        Args:
            Z, local_fields, degrees, mask: 同 forward。

        Returns:
            (action, log_prob)
            - action: 选中的节点索引 (int64 scalar)。
            - log_prob: 对应 log 概率 (scalar)。
        """
        logits = self.forward(Z, local_fields, degrees, mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob

    def get_log_prob(
        self,
        Z: torch.Tensor,
        local_fields: torch.Tensor,
        degrees: torch.Tensor,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        给定动作，计算当前策略下的 log_prob。

        Args:
            action: 动作索引 (batch,)，或标量。

        Returns:
            log_prob 张量。
        """
        logits = self.forward(Z, local_fields, degrees, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action)

    def entropy(
        self,
        Z: torch.Tensor,
        local_fields: torch.Tensor,
        degrees: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算策略熵。"""
        logits = self.forward(Z, local_fields, degrees, mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()


class ConstructPolicyHead(nn.Module):
    """
    构造策略头：两阶段注意力采样解决 O(n^2) 动作空间问题。

    1. 物理预筛选：计算所有非边的 ΔH_add，取 Top-K 候选集。
    2. 策略精炼：对候选边做 MLP，输出边 logits。
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        mlp_hidden: int = 128,
        top_k_candidates: int = 100,
    ) -> None:
        """
        Args:
            hidden_dim: 节点嵌入维度。
            mlp_hidden: MLP 隐藏层维度。
            top_k_candidates: 物理预筛选保留的候选边数。
        """
        super().__init__()
        self.top_k_candidates = top_k_candidates
        # 边特征：[z_i; z_j; J_{ij}; ΔH_{ij}] => 2*hidden_dim + 2
        in_dim = 2 * hidden_dim + 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def physical_topk(
        self,
        J: torch.Tensor,
        s: torch.Tensor,
        gamma: float,
        h_field: float,
        common_neighbors: dict,
        existing_edges: set,
        n: int,
        K: int,
    ) -> List[Tuple[int, int, torch.Tensor]]:
        """
        物理预筛选：对所有非边计算 ΔH_add，取 Top-K。

        Args:
            J: 耦合矩阵 (n, n)。
            s: 自旋组态 (n, n)。
            gamma, h_field: 物理参数。
            common_neighbors: 公共邻居字典。
            existing_edges: 已存在边的集合 {(u,v), ...}。
            n: 节点数。
            K: 取前 K 个。

        Returns:
            候选边列表 [(i, j, delta_H), ...]，按 delta_H 升序排列
            （delta_H 越小越优先，因为添加边后能量下降更多）。
        """
        from utils.spin_glass import delta_add

        candidates = []
        for i in range(n):
            for j in range(i + 1, n):
                if (i, j) in existing_edges or (j, i) in existing_edges:
                    continue
                dH = delta_add(J, s, i, j, gamma, h_field, common_neighbors)
                candidates.append((i, j, dH))

        # 按 ΔH 升序排列（能量下降最多优先）
        candidates.sort(key=lambda x: x[2].item())
        return candidates[:K]

    def forward(
        self,
        Z: torch.Tensor,
        J: torch.Tensor,
        candidates: List[Tuple[int, int, torch.Tensor]],
    ) -> torch.Tensor:
        """
        对候选边计算策略 logits。

        Args:
            Z: 节点嵌入 (n, hidden_dim)。
            J: 耦合矩阵 (n, n)。
            candidates: 物理预筛选后的候选边列表 [(i, j, delta_H), ...]。

        Returns:
            logits (K,)，K 为候选边数量。
        """
        if len(candidates) == 0:
            return torch.tensor([], device=Z.device, dtype=Z.dtype)

        features = []
        for i, j, dH in candidates:
            z_i = Z[i]
            z_j = Z[j]
            J_ij = J[i, j].unsqueeze(0)
            dH_feat = dH.detach().unsqueeze(0)
            feat = torch.cat([z_i, z_j, J_ij, dH_feat], dim=-1)
            features.append(feat)

        features = torch.stack(features, dim=0)  # (K, 2*hidden_dim + 2)
        logits = self.mlp(features).squeeze(-1)  # (K,)
        return logits

    def sample(
        self,
        Z: torch.Tensor,
        J: torch.Tensor,
        candidates: List[Tuple[int, int, torch.Tensor]],
    ) -> Tuple[Tuple[int, int], torch.Tensor]:
        """
        在候选集上采样边。

        Args:
            Z, J, candidates: 同 forward。

        Returns:
            (action, log_prob)
            - action: (i, j) 元组。
            - log_prob: 标量 log 概率。
        """
        logits = self.forward(Z, J, candidates)
        if logits.numel() == 0:
            # 无可选动作，返回 dummy
            return (-1, -1), torch.tensor(0.0, device=Z.device)
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample()
        log_prob = dist.log_prob(idx)
        i, j, _ = candidates[idx.item()]
        return (i, j), log_prob

    def get_log_prob(
        self,
        Z: torch.Tensor,
        J: torch.Tensor,
        candidates: List[Tuple[int, int, torch.Tensor]],
        action_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        给定候选动作索引，计算 log_prob。

        Args:
            action_idx: 候选列表中的索引 (batch,)。

        Returns:
            log_prob 张量。
        """
        logits = self.forward(Z, J, candidates)
        if logits.numel() == 0:
            return torch.tensor(0.0, device=Z.device)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(action_idx)

    def entropy(
        self,
        Z: torch.Tensor,
        J: torch.Tensor,
        candidates: List[Tuple[int, int, torch.Tensor]],
    ) -> torch.Tensor:
        """计算策略熵。"""
        logits = self.forward(Z, J, candidates)
        if logits.numel() == 0:
            return torch.tensor(0.0, device=Z.device)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.entropy()
