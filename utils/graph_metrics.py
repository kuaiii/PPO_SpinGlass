"""
图度量计算与奖励归一化工具模块。

包含 LCC、代数连通度、公共邻居缓存、RunningMeanStd 等。
"""

from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import networkx as nx
import torch
import torch_geometric
from torch_geometric.data import Data
from scipy.sparse.linalg import eigsh
from scipy.sparse import csgraph


def largest_connected_component_ratio(G: nx.Graph) -> float:
    """
    计算最大连通分量占比 σ(G) / n。

    Args:
        G: NetworkX 无向图。

    Returns:
        LCC 节点数占总节点数的比例。
    """
    n = G.number_of_nodes()
    if n == 0:
        return 0.0
    if G.number_of_edges() == 0:
        return 1.0 / n  # 每个孤立点都是大小为1的连通分量
    lcc_size = len(max(nx.connected_components(G), key=len))
    return lcc_size / n


def algebraic_connectivity(G: nx.Graph) -> float:
    """
    计算图的代数连通度 λ₂（Laplacian 第二小特征值）。

    小图使用稠密特征值分解，大图使用稀疏 eigsh。

    Args:
        G: NetworkX 无向图。

    Returns:
        λ₂ 标量值。空图或无边图返回 0.0。
    """
    n = G.number_of_nodes()
    if n <= 1 or G.number_of_edges() == 0:
        return 0.0

    # 确保节点编号连续 0..n-1
    node_list = list(range(n))
    try:
        L = nx.laplacian_matrix(G, nodelist=node_list).astype(np.float32)
    except Exception:
        # 若图节点不连续，先 relabel
        G = nx.convert_node_labels_to_integers(G)
        node_list = list(range(G.number_of_nodes()))
        L = nx.laplacian_matrix(G, nodelist=node_list).astype(np.float32)

    if n <= 200:
        # 稠密特征值分解
        evals = np.linalg.eigvalsh(L.toarray())
        evals_sorted = np.sort(evals)
        return float(evals_sorted[1]) if len(evals_sorted) > 1 else 0.0
    else:
        # 稀疏：求 2 个最小特征值
        try:
            evals = eigsh(L, k=2, which='SM', return_eigenvectors=False)
            evals_sorted = np.sort(evals)
            return float(evals_sorted[1])
        except Exception:
            # fallback
            evals = np.linalg.eigvalsh(L.toarray())
            evals_sorted = np.sort(evals)
            return float(evals_sorted[1]) if len(evals_sorted) > 1 else 0.0


def compute_common_neighbors(G: nx.Graph) -> Dict[Tuple[int, int], Set[int]]:
    """
    预计算图中所有节点对的公共邻居字典。

    Args:
        G: NetworkX 无向图，节点编号建议为 0..n-1。

    Returns:
        字典，key=(min(u,v), max(u,v))，value=公共邻居集合。
    """
    adj = {node: set(G.neighbors(node)) for node in G.nodes()}
    common: Dict[Tuple[int, int], Set[int]] = {}
    nodes = sorted(G.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            cns = adj[u].intersection(adj[v])
            if cns:
                common[(u, v)] = cns
    return common


def list_triangles(G: nx.Graph) -> List[Tuple[int, int, int]]:
    """
    枚举图中所有三角环。

    Args:
        G: NetworkX 无向图。

    Returns:
        三角环列表 [(i,j,k), ...]，满足 i < j < k。
    """
    triangles = []
    for u in G.nodes():
        neighbors = sorted(G.neighbors(u))
        for i, v in enumerate(neighbors):
            if v <= u:
                continue
            for w in neighbors[i + 1 :]:
                if w <= v:
                    continue
                if G.has_edge(v, w):
                    triangles.append((u, v, w))
    return triangles


class RunningMeanStd:
    """
    运行均值与标准差，用于奖励归一化。

    维护观测值序列的在线均值和方差估计，
    参考 OpenAI Baselines 实现。
    """

    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()) -> None:
        """
        Args:
            epsilon: 防止除零的小数。
            shape: 观测值形状，标量奖励用 ()。
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        """
        更新运行统计量。

        Args:
            x: 一批观测值，shape 与初始化时一致。
        """
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        对输入做 Z-score 归一化。

        Args:
            x: 待归一化数组。
            eps: 防止除零。

        Returns:
            归一化后的数组。
        """
        return (x - self.mean) / np.sqrt(self.var + eps)


def nx_to_pyg_data(G: nx.Graph, node_features: Optional[np.ndarray] = None, n_total: Optional[int] = None) -> Data:
    """
    将 NetworkX 图转为 PyTorch Geometric Data 对象。

    支持节点编号不连续的情况（如拆解环境）。
    若 n_total > G.number_of_nodes()，缺失节点视为孤立节点。

    Args:
        G: NetworkX 无向图，节点编号建议为 0..n_total-1。
        node_features: 节点特征数组 (n_total, feat_dim)，None 时用度作为特征。
        n_total: 总节点数，None 时取 G.number_of_nodes()。

    Returns:
        PyG Data 对象。
    """
    if n_total is None:
        n_total = G.number_of_nodes()
    if node_features is None:
        degrees = np.zeros((n_total, 1), dtype=np.float32)
        for i in range(n_total):
            degrees[i, 0] = float(G.degree(i)) if G.has_node(i) else 0.0
        node_features = degrees
    x = torch.from_numpy(node_features).float()

    edge_list = []
    for u, v in G.edges():
        edge_list.append([int(u), int(v)])
        edge_list.append([int(v), int(u)])
    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index, num_nodes=n_total)
