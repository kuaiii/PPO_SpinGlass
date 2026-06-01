"""
主入口脚本。

支持命令行参数启动训练、评估或测试。
"""

import argparse
import random
import numpy as np
import torch
import networkx as nx

from train_ppo import SpinGlassPPOTrainer
from eval import (
    evaluate_dismantle_policy,
    evaluate_construct_policy,
    random_dismantle_baseline,
    compute_auc,
    plot_evaluation,
)
from tests.test_validation import run_all_tests


def set_seed(seed: int) -> None:
    """设置随机种子保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # 小图 GNN 在 CPU 上多线程开销极大，默认设为单线程
    if not torch.cuda.is_available():
        torch.set_num_threads(1)

    parser = argparse.ArgumentParser(
        description="基于自旋玻璃能量景观与PPO的网络拓扑DRL优化引擎"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "test"],
        help="运行模式: train(训练), eval(评估), test(测试)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="dismantle",
        choices=["dismantle", "construct", "rewiring"],
        help="任务类型",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=20,
        help="初始图节点数（课程学习起点）",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=200,
        help="最大训练轮数",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备: auto, cpu, cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="GNN 隐藏维度",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="GAT 注意力头数",
    )
    parser.add_argument(
        "--lr_policy",
        type=float,
        default=3e-4,
        help="策略网络学习率",
    )
    parser.add_argument(
        "--lr_value",
        type=float,
        default=1e-3,
        help="值函数网络学习率",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="每轮收集的 episode 数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="PPO batch size",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        default=True,
        help="启用课程学习",
    )
    parser.add_argument(
        "--no_curriculum",
        action="store_true",
        help="禁用课程学习",
    )
    parser.add_argument(
        "--eval_graph",
        type=str,
        default=None,
        help="评估用的图文件路径（pickle 或 edgelist），None 时随机生成",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default="eval.png",
        help="评估结果保存路径",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="训练图数据目录（包含 .gml 文件），None 时随机生成",
    )
    parser.add_argument(
        "--max_train_graphs",
        type=int,
        default=None,
        help="最大加载训练图数",
    )
    parser.add_argument(
        "--subgraph_size",
        type=int,
        default=100,
        help="大图子采样目标大小",
    )
    parser.add_argument(
        "--random_graph_mix",
        type=float,
        default=0.0,
        help="随机图混合比例(0.0~1.0)，用于密度适应训练",
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=100,
        help="课程学习最大节点数",
    )
    parser.add_argument(
        "--node_increment",
        type=int,
        default=10,
        help="课程学习每次增加的节点数",
    )
    parser.add_argument(
        "--increment_every",
        type=int,
        default=100,
        help="课程学习每隔多少iteration增量",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50,
        help="每隔多少iteration保存一次checkpoint，0表示不保存",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    if args.no_curriculum:
        args.curriculum = False

    if args.mode == "test":
        print("[Mode] Running validation tests...")
        run_all_tests()
        return

    if args.mode == "train":
        print(f"[Mode] Training task={args.task}, n_nodes={args.n_nodes}, max_iters={args.max_iters}")
        if args.train_data_dir:
            print(f"[Mode] Loading training graphs from {args.train_data_dir}")
        trainer = SpinGlassPPOTrainer(
            task=args.task,
            n_nodes=args.n_nodes,
            in_channels=1,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            device=args.device,
            curriculum=args.curriculum,
            max_nodes=args.max_nodes,
            node_increment=args.node_increment,
            increment_every=args.increment_every,
            train_data_dir=args.train_data_dir,
            max_train_graphs=args.max_train_graphs,
            subgraph_size=args.subgraph_size,
            random_graph_mix=args.random_graph_mix,
        )
        ckpt_path = f"checkpoint_{args.task}.pt"
        history = trainer.train(max_iters=args.max_iters, save_every=args.save_every, ckpt_path=ckpt_path)
        print("\n[Train] Training completed.")
        print(f"[Train] Final avg reward: {history['reward'][-1]:.4f}")

        # 保存最终模型
        trainer.save_checkpoint(ckpt_path)
        print(f"[Train] Checkpoint saved to {ckpt_path}")

    elif args.mode == "eval":
        print(f"[Mode] Evaluating task={args.task}")
        # 加载 checkpoint
        ckpt_path = f"checkpoint_{args.task}.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Error] Checkpoint not found: {ckpt_path}")
            return

        checkpoint = torch.load(ckpt_path, map_location="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

        from models.encoder import TGNNEncoder
        from models.coupling import AttentionCoupling
        from models.policy import DismantlePolicyHead, ConstructPolicyHead
        from models.value import ValueHead

        encoder = TGNNEncoder(in_channels=1, hidden_dim=args.hidden_dim, num_heads=args.num_heads).to(device)
        coupling = AttentionCoupling(hidden_dim=args.hidden_dim).to(device)
        value_head = ValueHead(hidden_dim=args.hidden_dim).to(device)

        encoder.load_state_dict(checkpoint["encoder"])
        coupling.load_state_dict(checkpoint["coupling"])
        value_head.load_state_dict(checkpoint["value_head"])

        if args.task == "dismantle":
            policy_head = DismantlePolicyHead(hidden_dim=args.hidden_dim).to(device)
            policy_head.load_state_dict(checkpoint["policy_head"])

            # 生成或加载测试图
            if args.eval_graph:
                G = nx.read_edgelist(args.eval_graph)
                G = nx.convert_node_labels_to_integers(G)
            else:
                G = nx.barabasi_albert_graph(args.n_nodes, 2)
                G = nx.convert_node_labels_to_integers(G)

            result = evaluate_dismantle_policy(encoder, coupling, policy_head, value_head, G, device)
            baseline = random_dismantle_baseline(G, max_steps=args.n_nodes)
            plot_evaluation(result, baseline, save_path=args.save_plot)

            policy_auc = compute_auc(result["lcc_curve"])
            baseline_auc = compute_auc(baseline)
            print(f"[Eval] Policy AUC: {policy_auc:.4f}")
            print(f"[Eval] Random AUC: {baseline_auc:.4f}")

        elif args.task == "construct":
            policy_head = ConstructPolicyHead(hidden_dim=args.hidden_dim).to(device)
            policy_head.load_state_dict(checkpoint["policy_head"])

            if args.eval_graph:
                G = nx.read_edgelist(args.eval_graph)
                G = nx.convert_node_labels_to_integers(G)
            else:
                G = nx.barabasi_albert_graph(args.n_nodes, 1)
                G = nx.convert_node_labels_to_integers(G)

            result = evaluate_construct_policy(encoder, coupling, policy_head, value_head, G, device)
            plot_evaluation(result, save_path=args.save_plot)
            print(f"[Eval] Success: {result['success']}, Steps: {result['steps']}")

        else:
            print("[Eval] Rewiring evaluation not yet fully implemented.")


if __name__ == "__main__":
    import os
    main()
