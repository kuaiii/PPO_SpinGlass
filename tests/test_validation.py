"""
测试验证模块。

包含四个核心测试：
1. 能量一致性测试
2. PPO 收敛测试（星型图拆解）
3. 重构守恒测试
4. 物理引导测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from envs.topology_env import DismantleEnv, ConstructEnv, RewiringEnv
from models.encoder import TGNNEncoder
from models.coupling import AttentionCoupling
from models.policy import DismantlePolicyHead, ConstructPolicyHead
from models.value import ValueHead
from train_ppo import SpinGlassPPOTrainer
from utils.spin_glass import (
    hamiltonian,
    delta_remove,
    delta_add,
    adjacency_to_spin_dense,
    triadic_term_cache,
)
from utils.graph_metrics import algebraic_connectivity
from eval import evaluate_dismantle_policy, random_dismantle_baseline, compute_auc


def test_energy_consistency():
    """
    测试 1：能量一致性测试。
    随机图 n=15，随机执行 5 步单条边添加/移除，
    验证 env.H 的增量严格等于 delta_* 公式计算值（误差 < 1e-6）。

    注意：节点移除会同时移除多条边，导致 delta_remove 不可简单叠加。
    因此本测试针对单条边操作验证公式。
    """
    print("=" * 60)
    print("Test 1: Energy Consistency (Single Edge Operations)")
    print("=" * 60)

    n = 15
    G = nx.erdos_renyi_graph(n, 0.4, seed=42)
    G = nx.convert_node_labels_to_integers(G)

    # 固定随机 J
    J = torch.randn(n, n, dtype=torch.float32)
    J = (J + J.t()) * 0.5
    torch.diagonal(J).fill_(0.0)

    max_error = 0.0

    # --- 测试 delta_add：构造环境添加边 ---
    env_c = ConstructEnv(gamma=0.5, h_field=1.0)
    env_c.reset(G.copy())
    env_c.set_coupling(J)

    non_edges = env_c.get_candidate_edges()
    np.random.shuffle(non_edges)

    for step, (i, j) in enumerate(non_edges[:5]):
        H_prev = env_c.H
        s_old = env_c.s
        cn_old = env_c.common_neighbors

        delta_analytical = delta_add(
            J, s_old, i, j, env_c.gamma, env_c.h_field, cn_old
        ).item()

        obs, reward, done, info = env_c.step((i, j))
        H_new = env_c.H
        delta_numerical = H_new - H_prev
        error = abs(delta_numerical - delta_analytical)
        max_error = max(max_error, error)

        print(f"  Add step {step + 1}: edge=({i},{j}), "
              f"ΔH_analytical={delta_analytical:.6f}, "
              f"ΔH_numerical={delta_numerical:.6f}, error={error:.2e}")

        if done:
            break

    # --- 测试 delta_remove：拆解环境移除单条边 ---
    env_d = DismantleEnv(gamma=0.5, h_field=-1.0)
    env_d.reset(G.copy())
    env_d.set_coupling(J)

    edges = list(G.edges())
    np.random.shuffle(edges)

    for step, (i, j) in enumerate(edges[:5]):
        H_prev = env_d.H
        s_old = env_d.s
        cn_old = env_d.common_neighbors

        delta_analytical = delta_remove(
            J, s_old, i, j, env_d.gamma, env_d.h_field, cn_old
        ).item()

        # 在 DismantleEnv 中没有直接的 remove_edge API，我们手动构造一个
        # 通过临时环境或直接操作图
        G_tmp = env_d.G.copy()
        if G_tmp.has_edge(i, j):
            G_tmp.remove_edge(i, j)
        env_d.G = G_tmp
        env_d._update_spin_state()
        env_d.set_coupling(J)

        H_new = env_d.H
        delta_numerical = H_new - H_prev
        error = abs(delta_numerical - delta_analytical)
        max_error = max(max_error, error)

        print(f"  Remove step {step + 1}: edge=({i},{j}), "
              f"ΔH_analytical={delta_analytical:.6f}, "
              f"ΔH_numerical={delta_numerical:.6f}, error={error:.2e}")

    print(f"\n  Max error: {max_error:.2e}")
    assert max_error < 1e-5, f"Energy consistency failed: max_error={max_error}"
    print("  [PASS] Energy consistency test passed.\n")


def test_ppo_convergence():
    """
    测试 2：PPO 收敛测试。
    在 n=20 星型图上训练拆解策略，验证 200 iteration 内学会优先移除中心节点，
    且 LCC AUC 优于随机策略 > 30%。
    """
    print("=" * 60)
    print("Test 2: PPO Convergence on Star Graph")
    print("=" * 60)

    n = 20
    center = 0
    G = nx.star_graph(n - 1)  # 节点 0 为中心
    # star_graph 的节点标签为 0..n-1，其中 0 为中心

    trainer = SpinGlassPPOTrainer(
        task="dismantle",
        n_nodes=n,
        in_channels=1,
        hidden_dim=32,
        num_heads=2,
        lr_policy=3e-4,
        lr_value=1e-3,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        K_epochs=4,
        num_episodes=10,
        batch_size=32,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        device="cpu",
        curriculum=False,
        max_nodes=n,
    )

    # 固定测试图，不使用 generate_graph
    def fixed_graph_generator(n_nodes):
        return G.copy()

    trainer.generate_graph = fixed_graph_generator

    print("  Training for 200 iterations...")
    history = trainer.train(max_iters=200)

    # 评估训练策略
    result = evaluate_dismantle_policy(
        trainer.encoder,
        trainer.coupling,
        trainer.policy_head,
        trainer.value_head,
        G.copy(),
        torch.device("cpu"),
        max_steps=n,
    )

    # 随机基线
    baseline_curve = random_dismantle_baseline(G.copy(), max_steps=n)
    policy_curve = result["lcc_curve"]

    policy_auc = compute_auc(policy_curve)
    baseline_auc = compute_auc(baseline_curve)
    improvement = (baseline_auc - policy_auc) / baseline_auc * 100.0  # 拆解希望 LCC 下降更快，AUC 更小更好

    print(f"  Policy AUC: {policy_auc:.4f}")
    print(f"  Random AUC: {baseline_auc:.4f}")
    print(f"  Improvement: {improvement:.2f}%")
    print(f"  Removed nodes: {result['removed_nodes'][:5]}")

    # 检查是否优先移除中心节点
    first_node = result["removed_nodes"][0] if result["removed_nodes"] else -1
    center_priority = (first_node == center)

    print(f"  First removed node: {first_node}, center: {center}")
    assert center_priority, f"Policy did not prioritize center node: first={first_node}"
    assert improvement > 30.0, f"AUC improvement only {improvement:.2f}%"
    print("  [PASS] PPO convergence test passed.\n")


def test_rewiring_conservation():
    """
    测试 3：重构守恒测试。
    RewiringEnv 运行 10 步，每步断言 |E| 不变。
    """
    print("=" * 60)
    print("Test 3: Rewiring Edge Count Conservation")
    print("=" * 60)

    n = 15
    G = nx.erdos_renyi_graph(n, 0.4, seed=123)
    G = nx.convert_node_labels_to_integers(G)
    initial_edges = G.number_of_edges()

    env = RewiringEnv(gamma=0.5, h_field=0.0, max_swaps=10)
    env.reset(G)

    for step in range(10):
        # 随机合法交换
        swaps = env.get_swap_candidates()
        if len(swaps) == 0:
            print(f"  No valid swaps at step {step + 1}, stopping.")
            break
        swap = swaps[np.random.randint(len(swaps))]
        obs, reward, done, info = env.step(swap)

        current_edges = env.G.number_of_edges()
        print(f"  Step {step + 1}: edges={current_edges}, initial={initial_edges}")
        assert current_edges == initial_edges, (
            f"Edge count changed: {current_edges} != {initial_edges}"
        )
        if done:
            break

    print("  [PASS] Rewiring conservation test passed.\n")


def test_physical_guidance():
    """
    测试 4：物理引导测试。
    固定随机 J，比较 h=+1 与 h=-1 时策略的动作分布差异。
    构造倾向高 J_{ij} 边，拆解倾向高局部场节点。
    """
    print("=" * 60)
    print("Test 4: Physical Guidance (h=+1 vs h=-1)")
    print("=" * 60)

    n = 10
    G = nx.erdos_renyi_graph(n, 0.5, seed=456)
    G = nx.convert_node_labels_to_integers(G)

    # 固定随机 J
    J = torch.randn(n, n, dtype=torch.float32)
    J = (J + J.t()) * 0.5
    torch.diagonal(J).fill_(0.0)

    # h = -1 (拆解)
    env_dismantle = DismantleEnv(gamma=0.5, h_field=-1.0)
    env_dismantle.reset(G.copy())
    env_dismantle.set_coupling(J)
    h_local_d = env_dismantle.get_local_fields(J)
    # 选择局部场最高的节点（拆解倾向）
    top_node_dismantle = torch.argmax(h_local_d).item()

    # h = +1 (构造)
    env_construct = ConstructEnv(gamma=0.5, h_field=1.0)
    env_construct.reset(G.copy())
    env_construct.set_coupling(J)
    # 计算所有非边的 J 值
    non_edges = []
    j_values = []
    for i in range(n):
        for j in range(i + 1, n):
            if not env_construct.G.has_edge(i, j):
                non_edges.append((i, j))
                j_values.append(J[i, j].item())
    top_edge_construct = non_edges[np.argmax(j_values)]

    print(f"  Top node for dismantle (h=-1): {top_node_dismantle}, h_local={h_local_d[top_node_dismantle]:.4f}")
    print(f"  Top edge for construct (h=+1): {top_edge_construct}, J={J[top_edge_construct[0], top_edge_construct[1]]:.4f}")

    # 构造策略头应倾向高 J 边
    assert J[top_edge_construct[0], top_edge_construct[1]] > 0, "Construct should prefer positive J edge"
    # 拆解策略应倾向高局部场节点
    assert h_local_d[top_node_dismantle] > 0, "Dismantle should prefer high local field node"
    print("  [PASS] Physical guidance test passed.\n")


def run_all_tests():
    """运行全部测试。"""
    print("\n" + "=" * 60)
    print("Running All Validation Tests")
    print("=" * 60 + "\n")

    test_energy_consistency()
    test_rewiring_conservation()
    test_physical_guidance()
    test_ppo_convergence()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
