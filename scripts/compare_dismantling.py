"""
对比实验：自旋玻璃 PPO  dismantling 模型 vs network_dismantling 中所有算法。

在测试集上运行，计算 LCC 曲线、AUC、R值（dismantling cost）等指标。
"""

import argparse
import logging
import json
import time
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import networkx as nx
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.io_utils import get_next_test_dir
from models.encoder import TGNNEncoder
from models.coupling import AttentionCoupling
from models.policy import DismantlePolicyHead
from models.value import ValueHead
from envs.topology_env import DismantleEnv
from network_dismantling.unified_interface import dismantle, METHOD_REGISTRY

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_ppo_model(checkpoint_path: str, device: torch.device):
    """加载已训练的 PPO dismantling 模型。"""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hidden_dim = 64
    encoder = TGNNEncoder(in_channels=1, hidden_dim=hidden_dim).to(device)
    coupling = AttentionCoupling(hidden_dim=hidden_dim).to(device)
    policy_head = DismantlePolicyHead(hidden_dim=hidden_dim).to(device)
    value_head = ValueHead(hidden_dim=hidden_dim).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    coupling.load_state_dict(ckpt["coupling"])
    policy_head.load_state_dict(ckpt["policy_head"])
    value_head.load_state_dict(ckpt["value_head"])

    encoder.eval()
    coupling.eval()
    policy_head.eval()
    value_head.eval()

    return encoder, coupling, policy_head, value_head


def ppo_dismantle_sequence(
    G: nx.Graph,
    encoder: TGNNEncoder,
    coupling: AttentionCoupling,
    policy_head: DismantlePolicyHead,
    value_head: ValueHead,
    device: torch.device,
    stop_condition: float = 0.01,
) -> List[int]:
    """
    使用 PPO 策略生成 dismantling 序列。
    """
    env = DismantleEnv(sigma_threshold=stop_condition)
    obs = env.reset(G)
    done = False
    removed = []

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
        removed.append(node)
        obs = obs_next

        if len(removed) >= G.number_of_nodes():
            break

    return removed


def compute_lcc_curve(G: nx.Graph, removal_sequence: List[int]) -> List[float]:
    """计算随着节点移除的 LCC 比例曲线。"""
    n = G.number_of_nodes()
    G_tmp = G.copy()
    curve = [largest_connected_component_size(G_tmp) / n]

    for node in removal_sequence:
        if G_tmp.has_node(node):
            G_tmp.remove_node(node)
        if G_tmp.number_of_nodes() == 0:
            curve.append(0.0)
            break
        curve.append(largest_connected_component_size(G_tmp) / n)

    return curve


def largest_connected_component_size(G: nx.Graph) -> int:
    """返回最大连通分量大小。"""
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_edges() == 0:
        return 1
    return len(max(nx.connected_components(G), key=len))


def compute_auc(lcc_curve: List[float]) -> float:
    """计算 LCC 曲线下面积（梯形法则）。"""
    return float(np.trapz(lcc_curve, dx=1.0))


def compute_r_auc(lcc_curve: List[float], thresholds: List[float] = None) -> Dict[str, float]:
    """计算 R 值：使 LCC 下降到各阈值所需移除的节点比例。"""
    if thresholds is None:
        thresholds = [0.5, 0.2, 0.1, 0.05, 0.01]
    results = {}
    n = len(lcc_curve)
    for t in thresholds:
        idx = next((i for i, val in enumerate(lcc_curve) if val <= t), n - 1)
        results[f"R_{int(t*100)}"] = idx / (n - 1) if n > 1 else 0.0
    return results


# ---------------------------------------------------------------------------
# Timeout wrapper for baseline methods
# ---------------------------------------------------------------------------

def _run_baseline_worker(args_dict):
    """Worker function for multiprocessing."""
    G_dict = args_dict["G_dict"]
    method = args_dict["method"]
    stop_condition = args_dict["stop_condition"]
    kwargs = args_dict.get("kwargs", {})

    # Reconstruct graph from dict
    G = nx.Graph()
    G.add_nodes_from(G_dict["nodes"])
    G.add_edges_from(G_dict["edges"])

    start = time.time()
    try:
        seq = dismantle(G, method=method, stop_condition=stop_condition, **kwargs)
        elapsed = time.time() - start
        return {"seq": seq, "elapsed": elapsed, "error": None}
    except Exception as e:
        return {"seq": [], "elapsed": time.time() - start, "error": str(e)}


def run_baseline_with_timeout(
    G: nx.Graph,
    method: str,
    stop_condition: int = 1,
    timeout: float = 30.0,
    use_mp: bool = False,
    **kwargs
) -> Tuple[List[int], float, str]:
    """
    运行 baseline 算法，可选带超时控制。

    注意：multiprocessing spawn 模式开销很大，对于快速算法建议关闭。

    Returns:
        (removal_sequence, elapsed_time, error_message)
    """
    if not use_mp:
        # Direct execution (fast for small graphs)
        start = time.time()
        try:
            seq = dismantle(G, method=method, stop_condition=stop_condition, **kwargs)
            elapsed = time.time() - start
            return seq, elapsed, None
        except Exception as e:
            return [], time.time() - start, str(e)

    # Multiprocessing with timeout
    G_dict = {"nodes": list(G.nodes()), "edges": list(G.edges())}
    args = {"G_dict": G_dict, "method": method, "stop_condition": stop_condition, "kwargs": kwargs}

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(1) as pool:
        async_result = pool.apply_async(_run_baseline_worker, (args,))
        try:
            result = async_result.get(timeout=timeout)
            return result["seq"], result["elapsed"], result["error"]
        except multiprocessing.TimeoutError:
            pool.terminate()
            return [], timeout, "TIMEOUT"


def evaluate_on_graph(
    G: nx.Graph,
    method_name: str,
    removal_sequence: List[int],
    elapsed_time: float,
    error: str = None,
) -> Dict[str, Any]:
    """评估单个算法在单张图上的表现。"""
    if error or not removal_sequence:
        return {
            "method": method_name,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "removed_count": 0,
            "auc": np.nan,
            "time": elapsed_time,
            "error": error or "EMPTY_SEQUENCE",
        }

    lcc_curve = compute_lcc_curve(G, removal_sequence)
    auc = compute_auc(lcc_curve)
    r_values = compute_r_auc(lcc_curve)

    return {
        "method": method_name,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "removed_count": len(removal_sequence),
        "auc": auc,
        "time": elapsed_time,
        "lcc_curve": lcc_curve,
        **r_values,
    }


def plot_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: str = "comparison.png",
    normalized: bool = False,
):
    """绘制所有方法的 LCC 曲线对比图。

    Args:
        normalized: 若为 True，横坐标为移除节点比例（removed_nodes / n）；
                    若为 False，横坐标为移除节点绝对数量。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    valid_results = {k: v for k, v in results_dict.items() if "lcc_curve" in v}
    if not valid_results:
        logger.warning("No valid results to plot")
        plt.close()
        return

    colors = plt.cm.tab20(np.linspace(0, 1, len(valid_results)))

    n_nodes = None
    for i, (method, res) in enumerate(sorted(valid_results.items())):
        curve = res["lcc_curve"]
        if n_nodes is None:
            n_nodes = res.get("n_nodes", len(curve))
        x = np.arange(len(curve))
        if normalized:
            x = x / n_nodes if n_nodes > 0 else x
        axes[0].plot(x, curve, label=method, color=colors[i], marker="o", markersize=2, markevery=max(1, len(curve)//20))

    if normalized:
        axes[0].set_xlabel("The ratio of removed nodes")
    else:
        axes[0].set_xlabel("Removed nodes")
    axes[0].set_ylabel("LCC Ratio")
    axes[0].set_title("Dismantling Curves")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Bar plot: AUC comparison
    methods = sorted(valid_results.keys())
    aucs = [valid_results[m]["auc"] for m in methods]
    bars = axes[1].bar(range(len(methods)), aucs, color=colors[:len(methods)])
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("AUC")
    axes[1].set_title("AUC (lower is better)")
    axes[1].grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, aucs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    logger.info(f"Comparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Dismantling 对比实验")
    parser.add_argument("--test_dir", type=str, required=True, help="测试图目录（包含 .gml 文件）")
    parser.add_argument("--checkpoint", type=str, default=str(PROJECT_ROOT / "checkpoints" / "checkpoint_dismantle.pt"), help="PPO 模型 checkpoint")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help="要对比的算法列表，默认全部可用算法")
    parser.add_argument("--max_graphs", type=int, default=None, help="最大测试图数量")
    parser.add_argument("--stop_condition", type=float, default=0.01, help="LCC 停止阈值")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录。默认自动创建递增编号文件夹 results/tests/{编号}")
    parser.add_argument("--device", type=str, default="cuda", help="计算设备")
    parser.add_argument("--timeout", type=float, default=30.0, help="每个 baseline 方法的超时时间（秒）")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    device = torch.device(args.device)
    if args.output_dir is None:
        output_dir = get_next_test_dir(PROJECT_ROOT / "results" / "tests")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 PPO 模型
    logger.info(f"Loading PPO model from {args.checkpoint}")
    encoder, coupling, policy_head, value_head = load_ppo_model(args.checkpoint, device)

    # 确定对比算法
    if args.methods is None:
        methods = list(METHOD_REGISTRY.keys())
    else:
        methods = [m for m in args.methods if m in METHOD_REGISTRY]
    logger.info(f"Comparing methods: {methods}")

    # 加载测试图
    test_dir = Path(args.test_dir)
    graph_files = sorted(test_dir.glob("*.gml"))
    if args.max_graphs:
        graph_files = graph_files[:args.max_graphs]
    logger.info(f"Found {len(graph_files)} test graphs in {test_dir}")

    all_methods = ["PPO_SpinGlass"] + methods
    all_results = {method: [] for method in all_methods}

    for idx, gfile in enumerate(graph_files):
        logger.info(f"[{idx+1}/{len(graph_files)}] Processing {gfile.name}")
        try:
            G = nx.read_gml(gfile, label="id")
            G = nx.Graph(G)
            G.remove_edges_from(nx.selfloop_edges(G))
            if G.number_of_nodes() == 0:
                continue
            G = nx.convert_node_labels_to_integers(G)
        except Exception as e:
            logger.warning(f"Failed to load {gfile.name}: {e}")
            continue

        # PPO 策略
        start = time.time()
        seq_ppo = ppo_dismantle_sequence(G, encoder, coupling, policy_head, value_head, device, stop_condition=args.stop_condition)
        elapsed_ppo = time.time() - start
        res_ppo = evaluate_on_graph(G, "PPO_SpinGlass", seq_ppo, elapsed_ppo)
        all_results["PPO_SpinGlass"].append(res_ppo)
        logger.info(f"  PPO_SpinGlass: AUC={res_ppo['auc']:.3f}, time={elapsed_ppo:.3f}s")

        # Baseline 方法
        for method in methods:
            try:
                seq, elapsed, error = run_baseline_with_timeout(
                    G, method,
                    stop_condition=max(1, int(args.stop_condition * G.number_of_nodes())),
                    timeout=args.timeout,
                    use_mp=True,
                )
                res = evaluate_on_graph(G, method, seq, elapsed, error)
                all_results[method].append(res)
                status = f"AUC={res.get('auc', 'N/A')}" if not error else f"ERROR={error}"
                logger.info(f"  {method}: {status}, time={elapsed:.3f}s")
            except Exception as e:
                logger.warning(f"Method {method} failed on {gfile.name}: {e}")
                all_results[method].append({
                    "method": method,
                    "n_nodes": G.number_of_nodes(),
                    "n_edges": G.number_of_edges(),
                    "auc": np.nan,
                    "time": np.nan,
                    "error": str(e),
                })

    # 汇总统计
    summary = {}
    for method, results in all_results.items():
        if not results:
            continue
        valid = [r for r in results if not np.isnan(r.get("auc", np.nan)) and "error" not in r]
        if not valid:
            continue
        summary[method] = {
            "avg_auc": np.mean([r["auc"] for r in valid]),
            "std_auc": np.std([r["auc"] for r in valid]),
            "avg_time": np.mean([r["time"] for r in valid]),
            "std_time": np.std([r["time"] for r in valid]),
            "success_rate": len(valid) / len(results),
        }
        for key in ["R_50", "R_20", "R_10", "R_5", "R_1"]:
            vals = [r.get(key, np.nan) for r in valid]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                summary[method][f"avg_{key}"] = np.mean(vals)
                summary[method][f"std_{key}"] = np.std(vals)

    # 保存 JSON
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "detailed_results.json", "w") as f:
        serializable = {}
        for method, results in all_results.items():
            serializable[method] = [{k: v for k, v in r.items() if k != "lcc_curve"} for r in results]
        json.dump(serializable, f, indent=2)

    # 打印表格
    logger.info("\n" + "="*90)
    logger.info("Summary Results")
    logger.info("="*90)
    header = f"{'Method':<25} {'Avg AUC':<10} {'Std AUC':<10} {'Avg Time':<10} {'Succ%':<8} {'R_1':<10}"
    logger.info(header)
    logger.info("-"*90)
    for method in sorted(summary.keys()):
        s = summary[method]
        r1 = s.get("avg_R_1", np.nan)
        logger.info(f"{method:<25} {s['avg_auc']:<10.4f} {s['std_auc']:<10.4f} {s['avg_time']:<10.4f} {s['success_rate']:<8.2%} {r1:<10.4f}")

    # 为每张图绘制对比曲线（removed nodes 与 ratio 两个版本）
    for idx, gfile in enumerate(graph_files):
        graph_results = {}
        for method in all_results:
            if all_results[method] and idx < len(all_results[method]):
                graph_results[method] = all_results[method][idx]
        if graph_results:
            stem = Path(gfile).stem
            plot_comparison(graph_results, save_path=str(output_dir / f"{stem}_lcc.png"), normalized=False)
            plot_comparison(graph_results, save_path=str(output_dir / f"{stem}_lcc_ratio.png"), normalized=True)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
