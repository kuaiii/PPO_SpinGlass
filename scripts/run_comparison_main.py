#!/usr/bin/env python3
"""
合成网络拆解对比实验 (kanResilience 环境)。
拓扑: BA, WS, RA
规模: 50, 100, 200, 300, 400, 500
方法: PPO_SpinGlass + network_dismantling 中除 FINDER/GDM 外的可用方法。
结果保存至 results/kanResilience_results.json
"""
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any

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

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda")
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "checkpoint_dismantle.pt"
THRESHOLD = 0.1
SEED = 42
SIZES = [50, 100, 200, 300, 400, 500]
TOPOLOGIES = ["BA", "WS", "RA"]


def generate_graphs(sizes: List[int], seed: int = 42) -> Dict[str, Dict[int, nx.Graph]]:
    graphs = {"BA": {}, "WS": {}, "RA": {}}
    rng = np.random.default_rng(seed)
    for n in sizes:
        # BA: m=2
        graphs["BA"][n] = nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31)))
        # WS: k=4, p=0.3
        k = 4 if n > 4 else 2
        k = k if k % 2 == 0 else k - 1
        if k < 2:
            k = 2
        graphs["WS"][n] = nx.watts_strogatz_graph(n, k, 0.3, seed=int(rng.integers(0, 2**31)))
        # RA: Random Regular d=4
        d = 4
        if n <= d:
            d = max(2, n - 1)
            d = d if d % 2 == 0 else d - 1
            if d < 2:
                d = 2
        if n * d % 2 != 0:
            d = 2
        try:
            graphs["RA"][n] = nx.random_regular_graph(d, n, seed=int(rng.integers(0, 2**31)))
        except Exception:
            p = 4.0 / max(1, n - 1)
            graphs["RA"][n] = nx.erdos_renyi_graph(n, p, seed=int(rng.integers(0, 2**31)))
    return graphs


def load_ppo_model(checkpoint_path: str, device: torch.device):
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


def ppo_dismantle_sequence(G, encoder, coupling, policy_head, value_head, device, threshold=0.1):
    env = DismantleEnv(sigma_threshold=threshold)
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


def lcc_size(G):
    if G.number_of_nodes() == 0:
        return 0
    if G.number_of_edges() == 0:
        return 1
    return len(max(nx.connected_components(G), key=len))


def compute_lcc_curve(G, seq):
    n = G.number_of_nodes()
    G_tmp = G.copy()
    curve = [lcc_size(G_tmp) / n]
    for node in seq:
        if G_tmp.has_node(node):
            G_tmp.remove_node(node)
        if G_tmp.number_of_nodes() == 0:
            curve.append(0.0)
            break
        curve.append(lcc_size(G_tmp) / n)
    return curve


def compute_metrics(G, seq, threshold=0.1):
    n = G.number_of_nodes()
    curve = compute_lcc_curve(G, seq)
    auc = float(np.trapz(curve, dx=1.0))
    rem_num = next((i for i, val in enumerate(curve) if val <= threshold), len(curve) - 1)
    return {
        "n": n,
        "auc": auc,
        "rem_num": rem_num,
        "rem_ratio": rem_num / n if n > 0 else 0.0,
        "curve": curve,
    }


def run_baseline(G, method, threshold=0.1):
    stop_cond = max(1, int(np.ceil(threshold * G.number_of_nodes())))
    start = time.time()
    try:
        seq = dismantle(G, method=method, stop_condition=stop_cond)
        elapsed = time.time() - start
        metrics = compute_metrics(G, seq, threshold)
        metrics["time"] = elapsed
        metrics["error"] = None
        return metrics
    except Exception as e:
        elapsed = time.time() - start
        return {"n": G.number_of_nodes(), "auc": np.nan, "rem_num": np.nan,
                "rem_ratio": np.nan, "time": elapsed, "error": str(e), "curve": []}


def run_ppo(G, encoder, coupling, policy_head, value_head, threshold=0.1):
    start = time.time()
    seq = ppo_dismantle_sequence(G, encoder, coupling, policy_head, value_head, DEVICE, threshold)
    elapsed = time.time() - start
    metrics = compute_metrics(G, seq, threshold)
    metrics["time"] = elapsed
    metrics["error"] = None
    return metrics


def plot_lcc_curves(records, output_path, normalized=False):
    """为每个 (topology, n) 绘制所有方法的 LCC 曲线对比图。

    Parameters
    ----------
    records : list
        测试结果记录列表
    output_path : Path
        输出图片路径
    normalized : bool
        若为 True，横坐标为移除节点比例（removed_nodes / n）；
        若为 False，横坐标为移除节点绝对数量。
    """
    grouped = {}
    for r in records:
        key = (r["topology"], r["n"])
        grouped.setdefault(key, []).append(r)

    n_plots = len(grouped)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, (key, group) in zip(axes, grouped.items()):
        topo, n = key
        valid = [r for r in group if r.get("error") is None and r.get("curve")]
        valid.sort(key=lambda r: r.get("auc", float("inf")))

        for rec in valid:
            curve = rec["curve"]
            x = np.arange(len(curve))
            if normalized:
                x = x / n
            label = f"{rec['method']} (AUC={rec['auc']:.2f})"
            ax.plot(x, curve, label=label, linewidth=1.5)

        ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
        if normalized:
            ax.set_xlabel("The ratio of removed nodes")
        else:
            ax.set_xlabel("Removed nodes")
        ax.set_ylabel("LCC size / n")
        ax.set_title(f"{topo}-{n}")
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved LCC curves to {output_path}")


def main():
    output_dir = get_next_test_dir(PROJECT_ROOT / "results" / "tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Main] Using device: {DEVICE}")
    print(f"[Main] Loading PPO model from {CHECKPOINT}")
    print(f"[Main] Output dir: {output_dir}")
    encoder, coupling, policy_head, value_head = load_ppo_model(CHECKPOINT, DEVICE)

    graphs = generate_graphs(SIZES, seed=SEED)

    # 排除 FINDER(不在此环境) 和 GDM/GDM+R(模型缺失)
    skip_methods = {"GDM", "GDM+R"}
    methods = [m for m in METHOD_REGISTRY.keys() if m not in skip_methods]
    methods.sort()

    records = []

    for topo in TOPOLOGIES:
        for n in SIZES:
            G = graphs[topo][n]
            G = nx.convert_node_labels_to_integers(G)
            print(f"\n[{topo}] n={n}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

            # PPO
            res = run_ppo(G, encoder, coupling, policy_head, value_head, THRESHOLD)
            records.append({"topology": topo, "n": n, "method": "PPO_SpinGlass", **res})
            print(f"  PPO_SpinGlass: AUC={res['auc']:.3f}, rem_num={res['rem_num']}, time={res['time']:.3f}s")

            for method in methods:
                # 对已知极慢的方法在大图上跳过
                if method == "betweenness" and n >= 200:
                    print(f"  {method}: SKIPPED")
                    records.append({"topology": topo, "n": n, "method": method,
                                    "auc": np.nan, "rem_num": np.nan, "rem_ratio": np.nan,
                                    "time": np.nan, "error": "SKIPPED", "curve": []})
                    continue
                if method == "GND" and n >= 300:
                    print(f"  {method}: SKIPPED")
                    records.append({"topology": topo, "n": n, "method": method,
                                    "auc": np.nan, "rem_num": np.nan, "rem_ratio": np.nan,
                                    "time": np.nan, "error": "SKIPPED", "curve": []})
                    continue
                if method in ("entanglement_small", "entanglement_mid",
                              "entanglement_large", "vertex_entanglement") and n >= 400:
                    print(f"  {method}: SKIPPED")
                    records.append({"topology": topo, "n": n, "method": method,
                                    "auc": np.nan, "rem_num": np.nan, "rem_ratio": np.nan,
                                    "time": np.nan, "error": "SKIPPED", "curve": []})
                    continue
                if method == "brute_force" and n > 30:
                    print(f"  {method}: SKIPPED")
                    records.append({"topology": topo, "n": n, "method": method,
                                    "auc": np.nan, "rem_num": np.nan, "rem_ratio": np.nan,
                                    "time": np.nan, "error": "SKIPPED", "curve": []})
                    continue

                res = run_baseline(G, method, THRESHOLD)
                status = f"AUC={res['auc']:.3f}, rem_num={res['rem_num']}" if res["error"] is None else f"ERROR={res['error']}"
                print(f"  {method}: {status}, time={res['time']:.3f}s")
                records.append({"topology": topo, "n": n, "method": method, **res})

    json_records = [{k: v for k, v in r.items() if k != "curve"} for r in records]
    with open(output_dir / "kanResilience_results.json", "w") as f:
        json.dump(json_records, f, indent=2)

    # Plot - absolute removed nodes
    plot_lcc_curves(records, output_dir / "lcc_curves.png", normalized=False)

    # Plot - normalized removed nodes ratio
    plot_lcc_curves(records, output_dir / "lcc_curves_ratio.png", normalized=True)

    print(f"\n[Main] Results saved to {output_dir}")


if __name__ == "__main__":
    main()
