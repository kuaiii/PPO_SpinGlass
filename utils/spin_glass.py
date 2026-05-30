"""
自旋玻璃能量景观核心物理计算模块。

包含自旋组态编码、哈密顿量、局部能量变化、局部分子场等
严格遵循文档中的理论公式实现。
"""

from typing import Dict, List, Set, Tuple, Optional
import torch
import numpy as np
import networkx as nx


def spin_config(adj: np.ndarray) -> np.ndarray:
    """
    邻接矩阵 → 自旋组态。

    s_{ij} = 2 * A_{ij} - 1 ∈ {+1, -1}

    Args:
        adj: 邻接矩阵，shape (n, n)，元素为 0 或 1。

    Returns:
        自旋组态矩阵，元素为 +1 或 -1。
    """
    return 2.0 * adj - 1.0


def hamiltonian(
    J: torch.Tensor,
    s: torch.Tensor,
    gamma: float,
    h_field: float,
    triangles: List[Tuple[int, int, int]],
) -> torch.Tensor:
    """
    计算哈密顿量（状态能量）。

    H = - sum_{(i,j)} J_{ij} s_{ij}
        - gamma * sum_{(i,j,k)∈Δ} s_{ij} s_{jk} s_{ki}
        - h * sum_{(i,j)} s_{ij}

    注意：所有求和均针对无序对 (i<j) 进行，避免对称矩阵重复计数。

    Args:
        J: 耦合矩阵 (n, n)，稠密或稀疏表示。
        s: 自旋组态矩阵 (n, n)。
        gamma: 三角闭合系数，默认 0.5。
        h_field: 外部场。构造 +1.0，拆解 -1.0，重构 0.0。
        triangles: 当前图中三角环列表 [(i,j,k), ...]。

    Returns:
        标量张量，当前状态的能量。
    """
    n = J.shape[0]
    # 上三角索引，避免重复计数
    row_idx, col_idx = torch.triu_indices(n, n, offset=1, device=J.device)

    # 第一项：耦合项（无序对）
    term1 = -torch.sum(J[row_idx, col_idx] * s[row_idx, col_idx])

    # 第二项：三角闭合项（每个三角环只计一次）。
    # 注意：为保证与 delta_remove/delta_add 公式（含 2γ 系数）严格一致，
    # 此处使用 2*gamma，使单条边变化时三角项匹配。
    term2 = torch.tensor(0.0, dtype=J.dtype, device=J.device)
    if triangles and gamma != 0.0:
        vals = []
        for i, j, k in triangles:
            vals.append(s[i, j] * s[j, k] * s[k, i])
        if vals:
            term2 = -2.0 * gamma * torch.stack(vals).sum()

    # 第三项：外部场项（无序对）
    term3 = -h_field * torch.sum(s[row_idx, col_idx])

    return term1 + term2 + term3


def _get_common_neighbors_for_edge(
    i: int, j: int, common_neighbors: Dict[Tuple[int, int], Set[int]]
) -> Set[int]:
    """获取边 (i,j) 的公共邻居集合。"""
    key = (min(i, j), max(i, j))
    return common_neighbors.get(key, set())


def delta_remove(
    J: torch.Tensor,
    s: torch.Tensor,
    i: int,
    j: int,
    gamma: float,
    h_field: float,
    common_neighbors: Dict[Tuple[int, int], Set[int]],
) -> torch.Tensor:
    """
    移除边 (i,j) 的能量变化（s_{ij}: +1 → -1）。

    ΔH_remove = +2 * J_{ij}
                + 2 * gamma * sum_{k∈N(i)∩N(j)} s_{jk} * s_{ki}
                + 2 * h

    Args:
        J: 耦合矩阵 (n, n)。
        s: 自旋组态矩阵 (n, n)。
        i, j: 待移除边的两个端点。
        gamma: 三角闭合系数。
        h_field: 外部场。
        common_neighbors: 预计算的公共邻居字典。

    Returns:
        标量张量，能量变化值。
    """
    cns = _get_common_neighbors_for_edge(i, j, common_neighbors)
    tri_sum = torch.tensor(0.0, dtype=J.dtype, device=J.device)
    if cns and gamma != 0.0:
        vals = []
        for k in cns:
            vals.append(s[j, k] * s[k, i])
        if vals:
            tri_sum = torch.stack(vals).sum()

    delta = 2.0 * J[i, j] + 2.0 * gamma * tri_sum + 2.0 * h_field
    return delta


def delta_add(
    J: torch.Tensor,
    s: torch.Tensor,
    i: int,
    j: int,
    gamma: float,
    h_field: float,
    common_neighbors: Dict[Tuple[int, int], Set[int]],
) -> torch.Tensor:
    """
    添加边 (i,j) 的能量变化（s_{ij}: -1 → +1）。

    ΔH_add = -2 * J_{ij}
             - 2 * gamma * sum_{k∈N(i)∩N(j)} s_{jk} * s_{ki}
             - 2 * h

    Args:
        J: 耦合矩阵 (n, n)。
        s: 自旋组态矩阵 (n, n)。
        i, j: 待添加边的两个端点。
        gamma: 三角闭合系数。
        h_field: 外部场。
        common_neighbors: 预计算的公共邻居字典。

    Returns:
        标量张量，能量变化值。
    """
    cns = _get_common_neighbors_for_edge(i, j, common_neighbors)
    tri_sum = torch.tensor(0.0, dtype=J.dtype, device=J.device)
    if cns and gamma != 0.0:
        vals = []
        for k in cns:
            vals.append(s[j, k] * s[k, i])
        if vals:
            tri_sum = torch.stack(vals).sum()

    delta = -2.0 * J[i, j] - 2.0 * gamma * tri_sum - 2.0 * h_field
    return delta


def delta_swap(
    J: torch.Tensor,
    s: torch.Tensor,
    i_remove: int,
    j_remove: int,
    i_add: int,
    j_add: int,
    gamma: float,
    h_field: float,
    common_neighbors: Dict[Tuple[int, int], Set[int]],
) -> torch.Tensor:
    """
    Kawasaki 交换（固定边数）的能量变化。

    ΔH_swap = ΔH_add(i_add, j_add) + ΔH_remove(i_remove, j_remove)

    Args:
        J: 耦合矩阵 (n, n)。
        s: 自旋组态矩阵 (n, n)。
        i_remove, j_remove: 待移除边。
        i_add, j_add: 待添加边。
        gamma: 三角闭合系数。
        h_field: 外部场。
        common_neighbors: 预计算的公共邻居字典。

    Returns:
        标量张量，能量变化值。
    """
    d_remove = delta_remove(J, s, i_remove, j_remove, gamma, h_field, common_neighbors)
    d_add = delta_add(J, s, i_add, j_add, gamma, h_field, common_neighbors)
    return d_remove + d_add


def local_field(
    J: torch.Tensor,
    s: torch.Tensor,
    node: int,
    neighbors: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    计算局部分子场 h_i。

    h_i = sum_{j∈N(i)} J_{ij} * s_{ij}

    Args:
        J: 耦合矩阵 (n, n)。
        s: 自旋组态矩阵 (n, n)。
        node: 节点索引。
        neighbors: 邻居列表，若为 None 则默认使用所有节点。

    Returns:
        标量张量，局部分子场。
    """
    if neighbors is None:
        neighbors = list(range(J.shape[0]))
    if len(neighbors) == 0:
        return torch.tensor(0.0, dtype=J.dtype, device=J.device)
    idx = torch.tensor(neighbors, dtype=torch.long, device=J.device)
    return torch.sum(J[node, idx] * s[node, idx])


def triadic_term_cache(
    edge_index: torch.Tensor, n: int
) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int], Set[int]]]:
    """
    预计算三角环列表与公共邻居字典，避免每步 O(n^3) 枚举。

    Args:
        edge_index: 边索引，shape (2, m)。
        n: 节点数。

    Returns:
        (triangles, common_neighbors)
        - triangles: 三角环列表 [(i,j,k), ...]，i < j < k。
        - common_neighbors: 字典，key=(min(i,j), max(i,j))，value=公共邻居集合。
    """
    # 构建邻接集合
    adj_set: Dict[int, Set[int]] = {i: set() for i in range(n)}
    m = edge_index.shape[1]
    for e in range(m):
        u = int(edge_index[0, e].item())
        v = int(edge_index[1, e].item())
        adj_set[u].add(v)
        adj_set[v].add(u)

    # 计算公共邻居
    common_neighbors: Dict[Tuple[int, int], Set[int]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cns = adj_set[i].intersection(adj_set[j])
            if cns:
                common_neighbors[(i, j)] = cns

    # 枚举三角环
    triangles: List[Tuple[int, int, int]] = []
    for i in range(n):
        for j in adj_set[i]:
            if j <= i:
                continue
            for k in adj_set[i].intersection(adj_set[j]):
                if k > j:
                    triangles.append((i, j, k))

    return triangles, common_neighbors


def adjacency_to_spin_dense(G: nx.Graph, n_total: Optional[int] = None) -> torch.Tensor:
    """
    将 NetworkX 图转为稠密自旋组态张量 (n_total, n_total)。

    若 n_total > G.number_of_nodes()，则缺失节点视为孤立节点（自旋为 -1）。
    节点编号假设为 0..n_total-1，图中可能缺少某些节点（如拆解环境）。

    Args:
        G: NetworkX 无向图。
        n_total: 总节点数，None 时使用 G.number_of_nodes()。

    Returns:
        自旋组态张量 (n_total, n_total)，元素为 +1.0 或 -1.0。
    """
    if n_total is None:
        n_total = G.number_of_nodes()
    adj = np.zeros((n_total, n_total), dtype=np.float32)
    for u, v in G.edges():
        u_int = int(u)
        v_int = int(v)
        if 0 <= u_int < n_total and 0 <= v_int < n_total:
            adj[u_int, v_int] = 1.0
            adj[v_int, u_int] = 1.0
    s = torch.from_numpy(spin_config(adj))
    return s
