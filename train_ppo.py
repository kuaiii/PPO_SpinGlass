"""
PPO 训练器：SpinGlassPPOTrainer。

支持拆解、构造、重构三种任务的 PPO 训练，
包含 GAE 优势计算、clip 损失、值函数损失、联合对齐损失、课程学习。
"""

from typing import Dict, List, Tuple, Optional, Any
import copy
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import TensorDataset, DataLoader

from models.encoder import TGNNEncoder
from models.coupling import AttentionCoupling
from models.policy import DismantlePolicyHead, ConstructPolicyHead
from models.value import ValueHead
from envs.topology_env import DismantleEnv, ConstructEnv, RewiringEnv
from utils.graph_metrics import RunningMeanStd, nx_to_pyg_data
from utils.spin_glass import local_field
from utils.graph_loader import load_graphs_from_dir, sample_subgraph


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """
    计算 GAE（Generalized Advantage Estimation）。

    Args:
        rewards: 每步奖励列表。
        values: 每步值函数估计列表（V(s_t)）。
        gamma: 折扣因子。
        lam: GAE lambda。

    Returns:
        (advantages, returns)
        - advantages: 优势函数列表，长度与输入相同。
        - returns: 累积回报列表。
    """
    T = len(rewards)
    advantages = [0.0] * T
    last_gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


class SpinGlassPPOTrainer:
    """
    基于自旋玻璃能量景观的 PPO 训练器。
    """

    def __init__(
        self,
        task: str,
        n_nodes: int = 20,
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_heads: int = 4,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        lam: float = 0.95,
        eps_clip: float = 0.2,
        K_epochs: int = 4,
        num_episodes: int = 20,
        batch_size: int = 64,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        align_loss_coef: float = 0.1,
        align_every: int = 5,
        device: str = "auto",
        curriculum: bool = True,
        max_nodes: int = 100,
        node_increment: int = 10,
        increment_every: int = 100,
        train_data_dir: Optional[str] = None,
        max_train_graphs: Optional[int] = None,
        subgraph_size: Optional[int] = None,
    ) -> None:
        """
        Args:
            task: 'dismantle', 'construct', 或 'rewiring'。
            n_nodes: 初始图节点数（课程学习起点）。
            in_channels: 输入节点特征维度。
            hidden_dim: GNN 隐藏维度。
            num_heads: GAT 头数。
            lr_policy: 策略网络学习率。
            lr_value: 值函数网络学习率。
            gamma: 折扣因子。
            lam: GAE lambda。
            eps_clip: PPO clip 参数。
            K_epochs: 每轮数据复用训练轮数。
            num_episodes: 每轮收集的 episode 数。
            batch_size: PPO 更新 batch size。
            entropy_coef: 熵奖励系数。
            value_loss_coef: 值函数损失系数。
            align_loss_coef: 联合对齐损失系数。
            align_every: 每隔多少 iteration 计算对齐损失。
            device: 计算设备。
            curriculum: 是否启用课程学习。
            max_nodes: 课程学习最大节点数。
            node_increment: 每次增加的节点数。
            increment_every: 每隔多少 iteration 增加节点数。
            train_data_dir: 训练图数据目录，None 时随机生成。
            max_train_graphs: 最大加载训练图数。
            subgraph_size: 大图子采样目标大小，None 时不采样。
        """
        assert task in ("dismantle", "construct", "rewiring")
        self.task = task
        self.n_nodes = n_nodes
        self.curriculum = curriculum
        self.max_nodes = max_nodes
        self.node_increment = node_increment
        self.increment_every = increment_every

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 网络模块
        self.encoder = TGNNEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=3,
            num_heads=num_heads,
        ).to(self.device)
        self.coupling = AttentionCoupling(hidden_dim=hidden_dim).to(self.device)
        self.value_head = ValueHead(hidden_dim=hidden_dim).to(self.device)

        if task == "dismantle":
            self.policy_head = DismantlePolicyHead(hidden_dim=hidden_dim).to(self.device)
            self.env_class = DismantleEnv
        elif task == "construct":
            self.policy_head = ConstructPolicyHead(hidden_dim=hidden_dim).to(self.device)
            self.env_class = ConstructEnv
        else:
            # rewiring 使用 dismantle + construct 的组合，这里简化用 construct head 采样 e_in
            self.policy_head_add = ConstructPolicyHead(hidden_dim=hidden_dim).to(self.device)
            self.policy_head_remove = DismantlePolicyHead(hidden_dim=hidden_dim).to(self.device)
            self.env_class = RewiringEnv

        # 优化器
        params = (
            list(self.encoder.parameters())
            + list(self.coupling.parameters())
            + list(self.value_head.parameters())
        )
        if task in ("dismantle", "construct"):
            params += list(self.policy_head.parameters())
        else:
            params += list(self.policy_head_add.parameters()) + list(self.policy_head_remove.parameters())

        self.optimizer = torch.optim.Adam(params, lr=lr_policy)
        # 值函数可单独设置学习率（PyTorch 不支持 per-parameter lr 在同个 optimizer 中轻易实现，
        # 这里统一用一个 optimizer，值函数网络内部可微调）
        self.lr_policy = lr_policy
        self.lr_value = lr_value

        self.gamma = gamma
        self.lam = lam
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.align_loss_coef = align_loss_coef
        self.align_every = align_every

        self.reward_rms = RunningMeanStd(shape=())
        self.iteration = 0

        # 加载真实训练数据
        self.train_graphs: List[nx.Graph] = []
        self.subgraph_size = subgraph_size
        if train_data_dir is not None:
            self.train_graphs = load_graphs_from_dir(
                train_data_dir,
                min_nodes=10,
                max_nodes=None,
                max_graphs=max_train_graphs,
                ext=".gml",
                seed=42,
            )
            if not self.train_graphs:
                logger.warning(f"No training graphs loaded from {train_data_dir}, falling back to random generation")

    def generate_graph(self, n: int) -> nx.Graph:
        """
        生成训练用图。优先从真实数据集中采样，不足时随机生成。

        Args:
            n: 目标节点数（课程学习使用）。

        Returns:
            NetworkX 无向图。
        """
        # 如果加载了真实数据，从中采样
        if self.train_graphs:
            # 筛选接近目标大小的图
            candidates = [G for G in self.train_graphs if abs(G.number_of_nodes() - n) <= max(20, n // 2)]
            if not candidates:
                candidates = self.train_graphs
            G = random.choice(candidates)
            # 如果图太大，采样子图
            if self.subgraph_size is not None and G.number_of_nodes() > self.subgraph_size:
                G = sample_subgraph(G, self.subgraph_size)
            return G.copy()

        # 回退到随机生成
        if random.random() < 0.5:
            m = max(1, min(3, n // 2))
            G = nx.barabasi_albert_graph(n, m)
        else:
            k = max(2, min(6, n // 2))
            p = 0.3
            G = nx.watts_strogatz_graph(n, k, p)
        G = nx.convert_node_labels_to_integers(G)
        return G

    def collect_trajectories(
        self,
        num_episodes: int,
    ) -> List[Dict[str, Any]]:
        """
        收集一批 episode 轨迹。

        Args:
            num_episodes: episode 数量。

        Returns:
            轨迹列表，每条轨迹为 dict，包含 obs, actions, rewards, values, log_probs, masks, advantages, returns。
        """
        trajectories = []
        for _ in range(num_episodes):
            G = self.generate_graph(self.n_nodes)
            env = self.env_class()
            obs = env.reset(G)
            done = False

            episode_buffer = []
            while not done:
                obs_data = obs.to(self.device)
                with torch.no_grad():
                    Z = self.encoder(obs_data)
                    J = self.coupling(Z)
                    h_field = 0.0
                    if self.task == "dismantle":
                        h_field = -1.0
                    elif self.task == "construct":
                        h_field = 1.0

                    # 计算局部场
                    if self.task == "dismantle":
                        mask = env.get_alive_mask(device=self.device)
                        h_local = env.get_local_fields(J)
                        degs = env.get_degrees().to(self.device)
                        action, log_prob = self.policy_head.sample(Z, h_local, degs, mask)
                        v = self.value_head(Z, obs_data)
                        episode_buffer.append({
                            "obs": obs,
                            "Z": Z.cpu(),
                            "J": J.cpu(),
                            "action": action.item(),
                            "log_prob": log_prob.item(),
                            "value": v.item(),
                            "mask": mask.cpu(),
                            "h_local": h_local.cpu(),
                            "degs": degs.cpu(),
                        })
                        obs_next, reward, done, info = env.step(action.item())

                    elif self.task == "construct":
                        # 物理预筛选候选边
                        existing_edges = set()
                        for u, v in env.G.edges():
                            existing_edges.add((min(u, v), max(u, v)))
                        candidates = self.policy_head.physical_topk(
                            J, env.s, env.gamma, h_field,
                            env.common_neighbors, existing_edges,
                            env.n, self.policy_head.top_k_candidates,
                        )
                        if len(candidates) == 0:
                            done = True
                            reward = -1.0
                            info = {"invalid": True}
                            break
                        action, log_prob = self.policy_head.sample(Z, J, candidates)
                        v = self.value_head(Z, obs_data)
                        episode_buffer.append({
                            "obs": obs,
                            "Z": Z.cpu(),
                            "J": J.cpu(),
                            "action": action,
                            "action_idx": None,  # construct 需要记录候选索引
                            "log_prob": log_prob.item(),
                            "value": v.item(),
                            "candidates": candidates,
                        })
                        obs_next, reward, done, info = env.step(action)

                    else:  # rewiring
                        # 简化：先采样移除边（用 dismantle head 的节点对），再采样添加边
                        mask_remove = torch.ones(env.n, dtype=torch.bool, device=self.device)
                        for u, v in env.G.edges():
                            mask_remove[u] = True
                            mask_remove[v] = True
                        # 这里简化用随机选一条边移除，再构造候选添加
                        existing_edges = list(env.G.edges())
                        if len(existing_edges) == 0:
                            done = True
                            reward = -1.0
                            break
                        e_out = random.choice(existing_edges)
                        existing_set = set()
                        for u, v in env.G.edges():
                            existing_set.add((min(u, v), max(u, v)))
                        candidates = self.policy_head_add.physical_topk(
                            J, env.s, env.gamma, h_field,
                            env.common_neighbors, existing_set,
                            env.n, self.policy_head_add.top_k_candidates,
                        )
                        if len(candidates) == 0:
                            done = True
                            reward = -1.0
                            break
                        e_in, log_prob = self.policy_head_add.sample(Z, J, candidates)
                        v = self.value_head(Z, obs_data)
                        action = (e_out, e_in)
                        episode_buffer.append({
                            "obs": obs,
                            "Z": Z.cpu(),
                            "J": J.cpu(),
                            "action": action,
                            "log_prob": log_prob.item(),
                            "value": v.item(),
                            "candidates": candidates,
                        })
                        obs_next, reward, done, info = env.step(action)

                episode_buffer[-1]["reward"] = reward
                obs = obs_next

            # GAE 计算
            if len(episode_buffer) > 0:
                rewards = [buf["reward"] for buf in episode_buffer]
                values = [buf["value"] for buf in episode_buffer]
                # 归一化奖励
                rewards_arr = np.array(rewards, dtype=np.float32)
                self.reward_rms.update(rewards_arr.reshape(-1, 1))
                rewards_norm = self.reward_rms.normalize(rewards_arr).flatten().tolist()

                advantages, returns = compute_gae(rewards_norm, values, self.gamma, self.lam)
                for i, buf in enumerate(episode_buffer):
                    buf["advantage"] = advantages[i]
                    buf["return"] = returns[i]
                trajectories.extend(episode_buffer)

        return trajectories

    def update(
        self,
        trajectories: List[Dict[str, Any]],
        align_env: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        PPO 策略更新。

        Args:
            trajectories: 收集的轨迹数据。
            align_env: 用于计算对齐损失的另一个环境实例（可选）。

        Returns:
            损失统计字典。
        """
        if len(trajectories) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "align_loss": 0.0}

        # 构建 batch
        old_log_probs = torch.tensor([t["log_prob"] for t in trajectories], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([t["advantage"] for t in trajectories], dtype=torch.float32, device=self.device)
        returns = torch.tensor([t["return"] for t in trajectories], dtype=torch.float32, device=self.device)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_batches = 0

        for epoch in range(self.K_epochs):
            indices = torch.randperm(len(trajectories))
            for start in range(0, len(trajectories), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                batch_loss = torch.tensor(0.0, device=self.device)
                batch_policy_loss = 0.0
                batch_value_loss = 0.0
                batch_entropy_val = 0.0
                count = 0

                for idx in batch_idx:
                    t = trajectories[idx.item()]
                    obs_data = t["obs"].to(self.device)
                    Z = self.encoder(obs_data)
                    J = self.coupling(Z)
                    v = self.value_head(Z, obs_data)

                    if self.task == "dismantle":
                        mask = t["mask"].to(self.device)
                        h_local = t["h_local"].to(self.device)
                        degs = t["degs"].to(self.device)
                        action = torch.tensor(t["action"], device=self.device)
                        log_prob = self.policy_head.get_log_prob(Z, h_local, degs, action, mask)
                        entropy = self.policy_head.entropy(Z, h_local, degs, mask)
                    elif self.task == "construct":
                        candidates = t["candidates"]
                        action = t["action"]
                        action_idx = None
                        for i_cand, (i, j, _) in enumerate(candidates):
                            if (i, j) == action or (j, i) == action:
                                action_idx = i_cand
                                break
                        if action_idx is None:
                            continue
                        action_idx_t = torch.tensor(action_idx, device=self.device)
                        log_prob = self.policy_head.get_log_prob(Z, J, candidates, action_idx_t)
                        entropy = self.policy_head.entropy(Z, J, candidates)
                    else:
                        candidates = t["candidates"]
                        action_idx = None
                        _, e_in = t["action"]
                        for i_cand, (i, j, _) in enumerate(candidates):
                            if (i, j) == e_in or (j, i) == e_in:
                                action_idx = i_cand
                                break
                        if action_idx is None:
                            continue
                        action_idx_t = torch.tensor(action_idx, device=self.device)
                        log_prob = self.policy_head_add.get_log_prob(Z, J, candidates, action_idx_t)
                        entropy = self.policy_head_add.entropy(Z, J, candidates)

                    adv = advantages[idx]
                    ret = returns[idx]

                    ratio = torch.exp(log_prob - old_log_probs[idx])
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
                    policy_loss = -torch.min(surr1, surr2)
                    value_loss = F.mse_loss(v.squeeze(), ret)

                    loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
                    batch_loss = batch_loss + loss

                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_entropy_val += entropy.item()
                    count += 1

                if count > 0:
                    self.optimizer.zero_grad()
                    (batch_loss / count).backward()
                    torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.coupling.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 1.0)
                    if self.task in ("dismantle", "construct"):
                        torch.nn.utils.clip_grad_norm_(self.policy_head.parameters(), 1.0)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.policy_head_add.parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(self.policy_head_remove.parameters(), 1.0)
                    self.optimizer.step()

                    total_policy_loss += batch_policy_loss / count
                    total_value_loss += batch_value_loss / count
                    total_entropy += batch_entropy_val / count
                    num_batches += 1

        stats = {
            "policy_loss": total_policy_loss / max(num_batches, 1),
            "value_loss": total_value_loss / max(num_batches, 1),
            "entropy": total_entropy / max(num_batches, 1),
            "align_loss": 0.0,
        }

        # 联合对齐损失（每 align_every 轮）
        if align_env is not None and self.iteration % self.align_every == 0 and self.iteration > 0:
            G1 = self.generate_graph(self.n_nodes)
            G2 = self.generate_graph(self.n_nodes)
            obs1 = nx_to_pyg_data(G1).to(self.device)
            obs2 = nx_to_pyg_data(G2).to(self.device)
            with torch.no_grad():
                Z1 = self.encoder(obs1)
                Z2 = self.encoder(obs2)
            loss_align = F.mse_loss(Z1.mean(dim=0), Z2.mean(dim=0))
            self.optimizer.zero_grad()
            loss_align.backward()
            self.optimizer.step()
            stats["align_loss"] = loss_align.item()

        return stats

    def train(self, max_iters: int = 200) -> Dict[str, List[float]]:
        """
        主训练循环。

        Args:
            max_iters: 最大训练轮数。

        Returns:
            训练历史字典。
        """
        history = {"policy_loss": [], "value_loss": [], "entropy": [], "align_loss": [], "reward": []}

        for iteration in range(max_iters):
            self.iteration = iteration

            # 课程学习
            if self.curriculum and iteration > 0 and iteration % self.increment_every == 0:
                new_n = min(self.n_nodes + self.node_increment, self.max_nodes)
                if new_n > self.n_nodes:
                    self.n_nodes = new_n
                    print(f"[Curriculum] Increase graph size to {self.n_nodes}")

            trajectories = self.collect_trajectories(self.num_episodes)
            stats = self.update(trajectories)

            avg_reward = np.mean([t["reward"] for t in trajectories]) if trajectories else 0.0

            history["policy_loss"].append(stats["policy_loss"])
            history["value_loss"].append(stats["value_loss"])
            history["entropy"].append(stats["entropy"])
            history["align_loss"].append(stats["align_loss"])
            history["reward"].append(avg_reward)

            if iteration % 10 == 0:
                print(
                    f"Iter {iteration:03d} | n={self.n_nodes} | "
                    f"policy_loss={stats['policy_loss']:.4f} | "
                    f"value_loss={stats['value_loss']:.4f} | "
                    f"entropy={stats['entropy']:.4f} | "
                    f"reward={avg_reward:.4f}"
                )

        return history
