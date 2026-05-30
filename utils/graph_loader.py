"""
图数据加载器：从目录加载 NetworkX 图用于训练/测试。
"""

from typing import List, Optional
from pathlib import Path
import random
import logging

import networkx as nx

logger = logging.getLogger(__name__)


def load_graphs_from_dir(
    data_dir: str,
    min_nodes: int = 10,
    max_nodes: Optional[int] = None,
    max_graphs: Optional[int] = None,
    ext: str = ".gml",
    seed: Optional[int] = None,
) -> List[nx.Graph]:
    """
    从目录加载所有符合大小要求的图。

    Args:
        data_dir: 图文件目录。
        min_nodes: 最小节点数。
        max_nodes: 最大节点数，None 表示不限制。
        max_graphs: 最大加载图数，None 表示全部。
        ext: 文件扩展名。
        seed: 随机种子。

    Returns:
        NetworkX 图列表（已标准化为连续整数标签的简单图）。
    """
    data_path = Path(data_dir)
    files = sorted(data_path.glob(f"*{ext}"))
    if not files:
        logger.warning(f"No {ext} files found in {data_dir}")
        return []

    if seed is not None:
        random.seed(seed)
        random.shuffle(files)
    else:
        random.shuffle(files)

    graphs = []
    for f in files:
        if max_graphs is not None and len(graphs) >= max_graphs:
            break
        try:
            G = nx.read_gml(f, label="id")
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_nodes() == 0:
                continue
            # 标准化节点标签
            G = nx.convert_node_labels_to_integers(G)
            n = G.number_of_nodes()
            if n < min_nodes:
                continue
            if max_nodes is not None and n > max_nodes:
                continue
            graphs.append(G)
        except Exception as e:
            logger.debug(f"Failed to load {f.name}: {e}")
            continue

    logger.info(f"Loaded {len(graphs)} graphs from {data_dir} (min={min_nodes}, max={max_nodes})")
    return graphs


def sample_subgraph(G: nx.Graph, n_target: int, seed: Optional[int] = None) -> nx.Graph:
    """
    从大图中随机采样一个连通子图。

    Args:
        G: 原始图。
        n_target: 目标子图节点数。
        seed: 随机种子。

    Returns:
        连通子图。
    """
    if seed is not None:
        random.seed(seed)

    if G.number_of_nodes() <= n_target:
        return G.copy()

    # 从最大连通分量中采样
    lcc_nodes = max(nx.connected_components(G), key=len)
    if len(lcc_nodes) < n_target:
        n_target = len(lcc_nodes)

    # BFS 从随机节点开始采样
    start = random.choice(list(lcc_nodes))
    visited = {start}
    queue = [start]
    while len(visited) < n_target and queue:
        node = queue.pop(0)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= n_target:
                    break

    sub = G.subgraph(visited).copy()
    sub = nx.convert_node_labels_to_integers(sub)
    return sub
