#!/usr/bin/env python3
"""
合成网络拆解对比实验 (finder_tf 环境)。
只运行 FINDER 方法。
拓扑: BA, WS, RA
规模: 50, 100, 200, 300, 400, 500
结果保存至 results/finder_results.json
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'network_dismantling' / 'FINDER_ND'))
from finder_pure import FINDER_Pure
from utils.io_utils import get_next_test_dir

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
SIZES = [50, 100, 200, 300, 400, 500]
TOPOLOGIES = ["BA", "WS", "RA"]


def generate_graphs(sizes, seed=42):
    graphs = {"BA": {}, "WS": {}, "RA": {}}
    rng = np.random.default_rng(seed)
    for n in sizes:
        graphs["BA"][n] = nx.barabasi_albert_graph(n, 2, seed=int(rng.integers(0, 2**31)))
        k = 4 if n > 4 else 2
        k = k if k % 2 == 0 else k - 1
        if k < 2:
            k = 2
        graphs["WS"][n] = nx.watts_strogatz_graph(n, k, 0.3, seed=int(rng.integers(0, 2**31)))
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


def main():
    output_dir = get_next_test_dir(PROJECT_ROOT / "results" / "tests")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FINDER] Output dir: {output_dir}")
    print("[FINDER] Loading model...")
    dqn = FINDER_Pure()
    model_path = 'network_dismantling/FINDER_ND/models/nrange_30_50_iter_78000.ckpt'
    dqn.LoadModel(model_path)
    print("[FINDER] Model loaded.")

    graphs = generate_graphs(SIZES, seed=SEED)
    records = []

    for topo in TOPOLOGIES:
        for n in SIZES:
            G = graphs[topo][n]
            G = nx.convert_node_labels_to_integers(G)
            print(f"\n[{topo}] n={n}, nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

            solution, time_cost = dqn.EvaluateRealData(G, stepRatio=0.01)
            metrics = compute_metrics(G, solution, threshold=0.1)
            metrics["time"] = time_cost
            metrics["error"] = None
            records.append({"topology": topo, "n": n, "method": "FINDER", **metrics})
            print(f"  FINDER: AUC={metrics['auc']:.3f}, rem_num={metrics['rem_num']}, time={time_cost:.3f}s")

    json_records = [{k: v for k, v in r.items() if k != "curve"} for r in records]
    with open(output_dir / "finder_results.json", "w") as f:
        json.dump(json_records, f, indent=2)

    print(f"\n[FINDER] Results saved to {output_dir / 'finder_results.json'}")


if __name__ == "__main__":
    main()
