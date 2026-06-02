"""
拓扑优化环境：拆解、构造、重构三种任务的 Gym-like 环境。

所有环境返回 PyG Data 对象作为 observation。
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import copy
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from utils.spin_glass import (
    adjacency_to_spin_dense,
    hamiltonian,
    delta_remove,
    delta_add,
    delta_swap,
    local_field,
    triadic_term_cache,
    spin_config,
)
from utils.graph_metrics import (
    largest_connected_component_ratio,
    algebraic_connectivity,
    compute_common_neighbors,
    list_triangles,
    nx_to_pyg_data,
)


class DismantleEnv:
    """
    拆解环境：每次移除一个节点（级联删除关联边），
    目标使最大连通分量 σ(G) ≤ θ_σ * n。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        h_field: float = -1.0,
        alpha: float = 0.1,
        step_cost: float = 0.01,
        sigma_threshold: float = 0.1,
        max_steps: Optional[int] = None,
    ) -> None:
        """
        Args:
            gamma: 三角闭合系数。
            h_field: 外部场，拆解任务为 -1.0。
            alpha: 能量奖励权重。
            step_cost: 每步固定惩罚。
            sigma_threshold: LCC 终止阈值比例。
            max_steps: 最大步数，None 时默认 n。
        """
        self.gamma = gamma
        self.h_field = h_field
        self.alpha = alpha
        self.step_cost = step_cost
        self.sigma_threshold = sigma_threshold
        self.max_steps = max_steps

        self.G: Optional[nx.Graph] = None
        self.n: int = 0
        self.s: Optional[torch.Tensor] = None
        self.J: Optional[torch.Tensor] = None
        self.H: float = 0.0
        self.sigma: float = 1.0
        self.step_count: int = 0
        self.removed_nodes: Set[int] = set()
        self.triangles: List[Tuple[int, int, int]] = []
        self.common_neighbors: Dict[Tuple[int, int], Set[int]] = {}

    def reset(self, G: nx.Graph) -> Data:
        """
        初始化环境。

        Args:
            G: 初始图，节点编号应为 0..n-1。

        Returns:
            PyG Data 对象（当前图状态）。
        """
        self.G = G.copy()
        self.n = G.number_of_nodes()
        self.removed_nodes = set()
        self.step_count = 0
        if self.max_steps is None:
            self.max_steps = self.n

        self._update_spin_state()
        self.sigma = largest_connected_component_ratio(self.G)
        return self._get_observation()

    def _update_spin_state(self, incremental: bool = False, removed_node: Optional[int] = None, removed_neighbors: Optional[List[int]] = None) -> None:
        """
        更新自旋组态、三角环、公共邻居缓存。

        Args:
            incremental: 是否增量更新（只更新被移除节点相关）。
            removed_node: 被移除的节点索引（增量模式时使用）。
            removed_neighbors: 被移除节点的邻居列表（增量模式时使用）。
        """
        if incremental and removed_node is not None:
            # 增量更新：只把被移除节点的行/列设为 -1
            self.s[removed_node, :] = -1.0
            self.s[:, removed_node] = -1.0
            # 增量更新三角环和公共邻居
            if removed_neighbors is not None:
                self._remove_node_from_triadic_cache(removed_node, removed_neighbors)
        else:
            # 全量重建（reset 时使用）
            self.s = adjacency_to_spin_dense(self.G, n_total=self.n)
            edge_list = []
            for u, v in self.G.edges():
                edge_list.append([u, v])
                edge_list.append([v, u])
            if len(edge_list) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            self.triangles, self.common_neighbors = triadic_term_cache(edge_index, self.n)
        self.H = 0.0

    def _remove_node_from_triadic_cache(self, node: int, neighbors: List[int]) -> None:
        """从三角环和公共邻居缓存中增量移除与指定节点相关的条目。
        
        对于稠密图（三角形数量 > 5000），全量重建比逐条删除更快。
        """
        # 稠密图回退到全量重建
        if len(self.triangles) > 5000:
            edge_list = []
            for u, v in self.G.edges():
                edge_list.append([u, v])
                edge_list.append([v, u])
            if len(edge_list) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
            else:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            self.triangles, self.common_neighbors = triadic_term_cache(edge_index, self.n)
            return

        # 1. 删除包含 node 的三角形（稀疏图时很快）
        if self.triangles:
            self.triangles = [t for t in self.triangles if node not in t]
        # 2. 删除以 node 为 key 的公共邻居
        keys_to_delete = [k for k in self.common_neighbors if node in k]
        for k in keys_to_delete:
            del self.common_neighbors[k]
        # 3. 对于 node 的每一对邻居 (u, v)，从 (u,v) 的公共邻居集合中移除 node
        nbrs = set(neighbors)
        for i, u in enumerate(neighbors):
            for v in neighbors[i + 1 :]:
                key = (min(u, v), max(u, v))
                if key in self.common_neighbors and node in self.common_neighbors[key]:
                    self.common_neighbors[key].discard(node)
                    if not self.common_neighbors[key]:
                        del self.common_neighbors[key]

    def _get_observation(self) -> Data:
        """构造 PyG Data 观测。"""
        return nx_to_pyg_data(self.G, n_total=self.n)

    def set_coupling(self, J: torch.Tensor) -> None:
        """
        由外部模型传入耦合矩阵 J，并计算当前哈密顿量。

        Args:
            J: 耦合矩阵 (n, n)。
        """
        self.J = J
        self.H = hamiltonian(J, self.s, self.gamma, self.h_field, self.triangles).item()

    def step(self, node_idx: int) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        移除指定节点。

        Args:
            node_idx: 待移除节点索引。

        Returns:
            (observation, reward, done, info)
        """
        assert self.G is not None
        if node_idx in self.removed_nodes:
            # 无效动作，给予大惩罚
            return self._get_observation(), -10.0, True, {"invalid": True}

        # 记录移除前的状态
        sigma_old = self.sigma
        H_old = self.H

        # 移除节点
        neighbors = list(self.G.neighbors(node_idx))
        self.G.remove_node(node_idx)
        self.removed_nodes.add(node_idx)
        self.step_count += 1

        # 增量更新自旋状态
        self._update_spin_state(incremental=True, removed_node=node_idx, removed_neighbors=neighbors)
        if self.J is not None:
            self.H = hamiltonian(self.J, self.s, self.gamma, self.h_field, self.triangles).item()
        else:
            self.H = 0.0

        sigma_new = largest_connected_component_ratio(self.G)
        self.sigma = sigma_new

        # 奖励计算
        r_sigma = (sigma_old - sigma_new)  # -Δσ，拆解希望 LCC 下降
        r_H = self.alpha * (self.H - H_old)  # α * ΔH
        reward = r_sigma + r_H - self.step_cost

        # 终止判断
        done = False
        info = {"sigma": sigma_new, "H": self.H, "step": self.step_count}
        if sigma_new <= self.sigma_threshold:
            done = True
            reward += 10.0  # 成功额外奖励
            info["success"] = True
        elif self.step_count >= self.max_steps or self.G.number_of_nodes() == 0:
            done = True
            info["success"] = False

        return self._get_observation(), reward, done, info

    def get_alive_mask(self, device: torch.device = torch.device("cuda")) -> torch.Tensor:
        """
        获取存活节点掩码。

        Returns:
            bool 张量 (n,)，True 表示节点仍存活。
        """
        mask = torch.ones(self.n, dtype=torch.bool, device=device)
        for node in self.removed_nodes:
            mask[node] = False
        return mask

    def get_local_fields(self, J: torch.Tensor) -> torch.Tensor:
        """
        计算当前存活节点的局部分子场。

        Args:
            J: 耦合矩阵 (n, n)。

        Returns:
            h (n,)，已移除节点位置为 0。
        """
        h = torch.zeros(self.n, dtype=J.dtype, device=J.device)
        for i in range(self.n):
            if i in self.removed_nodes:
                continue
            if i not in self.G:
                continue
            neighbors = list(self.G.neighbors(i))
            if len(neighbors) > 0:
                h[i] = local_field(J, self.s.to(J.device), i, neighbors)
        return h

    def get_degrees(self) -> torch.Tensor:
        """获取当前各节点度数。"""
        degs = torch.zeros(self.n, dtype=torch.float)
        for i in range(self.n):
            if i in self.removed_nodes:
                degs[i] = 0.0
            else:
                degs[i] = float(self.G.degree(i)) if i in self.G else 0.0
        return degs


class ConstructEnv:
    """
    构造环境：每次添加一条边，目标使代数连通度 λ₂ ≥ θ_λ。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        h_field: float = 1.0,
        alpha: float = 0.05,
        step_cost: float = 0.01,
        lambda_threshold: float = 0.1,
        budget: Optional[int] = None,
    ) -> None:
        """
        Args:
            gamma: 三角闭合系数。
            h_field: 外部场，构造任务为 +1.0。
            alpha: 能量惩罚权重。
            step_cost: 每步固定惩罚。
            lambda_threshold: λ₂ 终止阈值。
            budget: 最大添加边数，None 时默认 n。
        """
        self.gamma = gamma
        self.h_field = h_field
        self.alpha = alpha
        self.step_cost = step_cost
        self.lambda_threshold = lambda_threshold
        self.budget = budget

        self.G: Optional[nx.Graph] = None
        self.n: int = 0
        self.s: Optional[torch.Tensor] = None
        self.J: Optional[torch.Tensor] = None
        self.H: float = 0.0
        self.lambda2: float = 0.0
        self.step_count: int = 0
        self.added_edges: Set[Tuple[int, int]] = set()
        self.triangles: List[Tuple[int, int, int]] = []
        self.common_neighbors: Dict[Tuple[int, int], Set[int]] = {}

    def reset(self, G: nx.Graph, budget: Optional[int] = None) -> Data:
        """
        初始化环境。

        Args:
            G: 初始图。
            budget: 最大添加边数，覆盖构造函数中的设置。

        Returns:
            PyG Data 对象。
        """
        self.G = G.copy()
        self.n = G.number_of_nodes()
        self.added_edges = set()
        self.step_count = 0
        if budget is not None:
            self.budget = budget
        elif self.budget is None:
            self.budget = self.n

        self._update_spin_state()
        self.lambda2 = algebraic_connectivity(self.G)
        return self._get_observation()

    def _update_spin_state(self) -> None:
        """更新自旋组态与缓存。"""
        self.s = adjacency_to_spin_dense(self.G, n_total=self.n)
        n = self.n
        edge_list = []
        for u, v in self.G.edges():
            edge_list.append([u, v])
            edge_list.append([v, u])
        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.triangles, self.common_neighbors = triadic_term_cache(edge_index, n)
        self.H = 0.0

    def _get_observation(self) -> Data:
        return nx_to_pyg_data(self.G, n_total=self.n)

    def set_coupling(self, J: torch.Tensor) -> None:
        """由外部模型传入耦合矩阵 J。"""
        self.J = J
        self.H = hamiltonian(J, self.s, self.gamma, self.h_field, self.triangles).item()

    def step(self, edge: Tuple[int, int]) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        添加一条边。

        Args:
            edge: (i, j) 待添加边。

        Returns:
            (observation, reward, done, info)
        """
        assert self.G is not None
        i, j = edge
        if i == j or self.G.has_edge(i, j):
            return self._get_observation(), -10.0, True, {"invalid": True}

        lambda2_old = self.lambda2
        H_old = self.H

        self.G.add_edge(i, j)
        self.added_edges.add((min(i, j), max(i, j)))
        self.step_count += 1

        self._update_spin_state()
        if self.J is not None:
            self.H = hamiltonian(self.J, self.s, self.gamma, self.h_field, self.triangles).item()
        else:
            self.H = 0.0

        lambda2_new = algebraic_connectivity(self.G)
        self.lambda2 = lambda2_new

        # 奖励
        reward = (lambda2_new - lambda2_old) - self.alpha * (self.H - H_old) - self.step_cost

        done = False
        info = {"lambda2": lambda2_new, "H": self.H, "step": self.step_count}
        if lambda2_new >= self.lambda_threshold:
            done = True
            reward += 10.0
            info["success"] = True
        elif self.step_count >= self.budget:
            done = True
            info["success"] = False

        return self._get_observation(), reward, done, info

    def get_edge_mask(self, device: torch.device = torch.device("cuda")) -> torch.Tensor:
        """
        获取有效动作掩码矩阵 (n, n)。
        这里返回布尔掩码表示哪些边可以添加。
        """
        mask = torch.ones((self.n, self.n), dtype=torch.bool, device=device)
        for u, v in self.G.edges():
            mask[u, v] = False
            mask[v, u] = False
        mask.fill_diagonal_(False)
        return mask

    def get_candidate_edges(self) -> List[Tuple[int, int]]:
        """返回所有可添加的非边列表。"""
        candidates = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not self.G.has_edge(i, j):
                    candidates.append((i, j))
        return candidates


class RewiringEnv:
    """
    重构环境：每次执行一次边交换（移除一条边 + 添加一条边），
    严格保持边数不变。目标使 λ₂ ≥ θ_λ。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        h_field: float = 0.0,
        alpha: float = 0.05,
        step_cost: float = 0.01,
        lambda_threshold: float = 0.1,
        max_swaps: Optional[int] = None,
    ) -> None:
        """
        Args:
            gamma: 三角闭合系数。
            h_field: 外部场，重构任务为 0.0。
            alpha: 能量惩罚权重。
            step_cost: 每步固定惩罚。
            lambda_threshold: λ₂ 终止阈值。
            max_swaps: 最大交换次数，None 时默认 2 * |E|。
        """
        self.gamma = gamma
        self.h_field = h_field
        self.alpha = alpha
        self.step_cost = step_cost
        self.lambda_threshold = lambda_threshold
        self.max_swaps = max_swaps

        self.G: Optional[nx.Graph] = None
        self.n: int = 0
        self.initial_edge_count: int = 0
        self.s: Optional[torch.Tensor] = None
        self.J: Optional[torch.Tensor] = None
        self.H: float = 0.0
        self.lambda2: float = 0.0
        self.step_count: int = 0
        self.triangles: List[Tuple[int, int, int]] = []
        self.common_neighbors: Dict[Tuple[int, int], Set[int]] = {}

    def reset(self, G: nx.Graph, max_swaps: Optional[int] = None) -> Data:
        """
        初始化环境。

        Args:
            G: 初始图。
            max_swaps: 最大交换次数。

        Returns:
            PyG Data 对象。
        """
        self.G = G.copy()
        self.n = G.number_of_nodes()
        self.initial_edge_count = G.number_of_edges()
        self.step_count = 0
        if max_swaps is not None:
            self.max_swaps = max_swaps
        elif self.max_swaps is None:
            self.max_swaps = 2 * self.initial_edge_count

        self._update_spin_state()
        self.lambda2 = algebraic_connectivity(self.G)
        return self._get_observation()

    def _update_spin_state(self) -> None:
        """更新自旋组态与缓存。"""
        self.s = adjacency_to_spin_dense(self.G, n_total=self.n)
        n = self.n
        edge_list = []
        for u, v in self.G.edges():
            edge_list.append([u, v])
            edge_list.append([v, u])
        if len(edge_list) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        self.triangles, self.common_neighbors = triadic_term_cache(edge_index, n)
        self.H = 0.0

    def _get_observation(self) -> Data:
        return nx_to_pyg_data(self.G, n_total=self.n)

    def set_coupling(self, J: torch.Tensor) -> None:
        """由外部模型传入耦合矩阵 J。"""
        self.J = J
        self.H = hamiltonian(J, self.s, self.gamma, self.h_field, self.triangles).item()

    def step(
        self, swap: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        执行一次边交换。

        Args:
            swap: ((i_out, j_out), (i_in, j_in))。

        Returns:
            (observation, reward, done, info)
        """
        assert self.G is not None
        (i_out, j_out), (i_in, j_in) = swap

        # 合法性检查
        if i_in == j_in or self.G.has_edge(i_in, j_in):
            return self._get_observation(), -10.0, True, {"invalid": True, "reason": "add edge exists"}
        if not self.G.has_edge(i_out, j_out):
            return self._get_observation(), -10.0, True, {"invalid": True, "reason": "remove edge missing"}
        if (i_out, j_out) == (i_in, j_in) or (i_out, j_out) == (j_in, i_in):
            return self._get_observation(), -10.0, True, {"invalid": True, "reason": "same edge"}

        lambda2_old = self.lambda2
        H_old = self.H

        # 先移除，再添加
        self.G.remove_edge(i_out, j_out)
        self.G.add_edge(i_in, j_in)
        self.step_count += 1

        # 守恒断言
        assert self.G.number_of_edges() == self.initial_edge_count, (
            f"Edge count changed: {self.G.number_of_edges()} != {self.initial_edge_count}"
        )

        self._update_spin_state()
        if self.J is not None:
            self.H = hamiltonian(self.J, self.s, self.gamma, self.h_field, self.triangles).item()
        else:
            self.H = 0.0

        lambda2_new = algebraic_connectivity(self.G)
        self.lambda2 = lambda2_new

        # 奖励
        reward = (lambda2_new - lambda2_old) - self.alpha * (self.H - H_old) - self.step_cost

        done = False
        info = {
            "lambda2": lambda2_new,
            "H": self.H,
            "step": self.step_count,
            "edge_count": self.G.number_of_edges(),
        }
        if lambda2_new >= self.lambda_threshold:
            done = True
            reward += 10.0
            info["success"] = True
        elif self.step_count >= self.max_swaps:
            done = True
            info["success"] = False

        return self._get_observation(), reward, done, info

    def get_swap_candidates(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        返回所有可能的合法交换对。
        简单实现：遍历所有可移除边和所有可添加边组合。
        """
        existing = list(self.G.edges())
        non_existing = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if not self.G.has_edge(i, j):
                    non_existing.append((i, j))
        swaps = []
        for e_out in existing:
            for e_in in non_existing:
                if set(e_out) != set(e_in):
                    swaps.append((e_out, e_in))
        return swaps
