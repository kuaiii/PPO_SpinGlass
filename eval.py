"""
评估与可视化模块。

包含策略评估、LCC 曲线、能量变化、策略熵、AUC 计算等。
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from envs.topology_env import DismantleEnv, ConstructEnv, RewiringEnv
from utils.graph_metrics import largest_connected_component_ratio, algebraic_connectivity
from models.encoder import TGNNEncoder
from models.coupling import AttentionCoupling
from models.policy import DismantlePolicyHead, ConstructPolicyHead
from models.value import ValueHead


def evaluate_dismantle_policy(
    encoder: TGNNEncoder,
    coupling: AttentionCoupling,
    policy_head: DismantlePolicyHead,
    value_head: ValueHead,
    G: nx.Graph,
    device: torch.device,
    max_steps: Optional[int] = None,
) -> Dict[str, any]:
    """
    评估拆解策略在单张图上的表现。

    Args:
        encoder, coupling, policy_head, value_head: 训练好的网络。
        G: 测试图。
        device: 计算设备。
        max_steps: 最大步数。

    Returns:
        评估结果字典，包含：
        - lcc_curve: LCC 比例变化列表。
        - energy_curve: 能量变化列表。
        - reward_sum: 累积奖励。
        - steps: 实际执行步数。
        - removed_nodes: 移除节点列表。
    """
    env = DismantleEnv(max_steps=max_steps)
    obs = env.reset(G)
    done = False

    lcc_curve = [env.sigma]
    energy_curve = [env.H]
    reward_sum = 0.0
    removed_nodes = []

    while not done:
        obs_data = obs.to(device)
        with torch.no_grad():
            Z = encoder(obs_data)
            J = coupling(Z)
            mask = env.get_alive_mask(device=device)
            h_local = env.get_local_fields(J)
            degs = env.get_degrees().to(device)
            action, _ = policy_head.sample(Z, h_local, degs, mask)
            node = action.item()

        obs_next, reward, done, info = env.step(node)
        reward_sum += reward
        removed_nodes.append(node)
        lcc_curve.append(info.get("sigma", env.sigma))
        energy_curve.append(info.get("H", env.H))
        obs = obs_next

    return {
        "lcc_curve": lcc_curve,
        "energy_curve": energy_curve,
        "reward_sum": reward_sum,
        "steps": len(removed_nodes),
        "removed_nodes": removed_nodes,
        "success": info.get("success", False),
    }


def evaluate_construct_policy(
    encoder: TGNNEncoder,
    coupling: AttentionCoupling,
    policy_head: ConstructPolicyHead,
    value_head: ValueHead,
    G: nx.Graph,
    device: torch.device,
    budget: Optional[int] = None,
    gamma: float = 0.5,
    h_field: float = 1.0,
) -> Dict[str, any]:
    """
    评估构造策略在单张图上的表现。

    Args:
        encoder, coupling, policy_head, value_head: 训练好的网络。
        G: 测试图。
        device: 计算设备。
        budget: 最大添加边数。
        gamma: 三角闭合系数。
        h_field: 外部场。

    Returns:
        评估结果字典。
    """
    env = ConstructEnv(gamma=gamma, h_field=h_field)
    obs = env.reset(G, budget=budget)
    done = False

    lambda2_curve = [env.lambda2]
    energy_curve = [env.H]
    reward_sum = 0.0
    added_edges = []

    while not done:
        obs_data = obs.to(device)
        with torch.no_grad():
            Z = encoder(obs_data)
            J = coupling(Z)
            existing_edges = set()
            for u, v in env.G.edges():
                existing_edges.add((min(u, v), max(u, v)))
            candidates = policy_head.physical_topk(
                J, env.s, env.gamma, h_field,
                env.common_neighbors, existing_edges,
                env.n, policy_head.top_k_candidates,
            )
            if len(candidates) == 0:
                break
            action, _ = policy_head.sample(Z, J, candidates)

        obs_next, reward, done, info = env.step(action)
        reward_sum += reward
        added_edges.append(action)
        lambda2_curve.append(info.get("lambda2", env.lambda2))
        energy_curve.append(info.get("H", env.H))
        obs = obs_next

    return {
        "lambda2_curve": lambda2_curve,
        "energy_curve": energy_curve,
        "reward_sum": reward_sum,
        "steps": len(added_edges),
        "added_edges": added_edges,
        "success": info.get("success", False),
    }


def random_dismantle_baseline(G: nx.Graph, max_steps: int) -> List[float]:
    """
    随机拆解基线：随机移除节点，返回 LCC 曲线。

    Args:
        G: 初始图。
        max_steps: 最大步数。

    Returns:
        LCC 比例列表。
    """
    env = DismantleEnv(max_steps=max_steps)
    env.reset(G)
    lcc_curve = [env.sigma]
    done = False
    while not done:
        alive = [i for i in range(env.n) if i not in env.removed_nodes]
        if len(alive) == 0:
            break
        node = np.random.choice(alive)
        _, _, done, info = env.step(node)
        lcc_curve.append(info.get("sigma", env.sigma))
    return lcc_curve


def compute_auc(curve: List[float]) -> float:
    """
    计算曲线下面积（梯形法则）。

    Args:
        curve: 曲线值列表。

    Returns:
        AUC 值。
    """
    return float(np.trapz(curve, dx=1.0))


def plot_evaluation(
    results: Dict[str, any],
    baseline_curve: Optional[List[float]] = None,
    save_path: str = "eval.png",
) -> None:
    """
    绘制评估曲线。

    Args:
        results: evaluate_dismantle_policy 或 evaluate_construct_policy 的输出。
        baseline_curve: 基线曲线（可选）。
        save_path: 保存路径。
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "lcc_curve" in results:
        axes[0].plot(results["lcc_curve"], label="Policy", marker="o")
        if baseline_curve is not None:
            axes[0].plot(baseline_curve, label="Random", marker="x")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("LCC Ratio")
        axes[0].set_title("Dismantle: LCC Curve")
        axes[0].legend()
        axes[0].grid(True)
    elif "lambda2_curve" in results:
        axes[0].plot(results["lambda2_curve"], label="Policy", marker="o")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Algebraic Connectivity λ₂")
        axes[0].set_title("Construct: λ₂ Curve")
        axes[0].legend()
        axes[0].grid(True)

    if "energy_curve" in results:
        axes[1].plot(results["energy_curve"], label="Energy", color="red")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Hamiltonian H")
        axes[1].set_title("Energy Landscape")
        axes[1].legend()
        axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Eval] Plot saved to {save_path}")
