#!/usr/bin/env python3
"""
生成合成训练图，覆盖多种规模和密度。

输出目录：
- dataset/synth_training/sparse_large/  : n=100~1000, 稀疏图 (ρ≤0.02)
- dataset/synth_training/dense_small/   : n=20~200, 各种密度 (ρ∈[0.001, 0.5])
"""

import argparse
import random
from pathlib import Path
import networkx as nx


def generate_sparse_large(output_dir: Path, num_graphs: int = 800, seed: int = 42):
    """生成稀疏大图：n=100~1000, 密度≤0.02"""
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in range(num_graphs):
        n = random.randint(100, 1000)
        gtype = random.choice(["ER", "BA", "WS"])
        if gtype == "ER":
            p = random.uniform(0.001, 0.02)
            G = nx.erdos_renyi_graph(n, p)
        elif gtype == "BA":
            m = random.randint(1, 5)
            G = nx.barabasi_albert_graph(n, m)
        else:  # WS
            k = random.randint(4, 10)
            p = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n, k, p)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        G = nx.convert_node_labels_to_integers(G)
        path = output_dir / f"sparse_{gtype}_{i:04d}_n{n}.gml"
        nx.write_gml(G, str(path))
        count += 1
    print(f"[Sparse Large] Generated {count} graphs in {output_dir}")


def generate_dense_small(output_dir: Path, num_graphs: int = 500, seed: int = 42):
    """生成中小图，覆盖各种密度：n=20~200, ρ∈[0.001, 0.5]"""
    random.seed(seed + 1)
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for i in range(num_graphs):
        n = random.randint(20, 200)
        # 对数均匀采样密度
        log_rho = random.uniform(-6.9, -0.69)  # ln(0.001) to ln(0.5)
        rho = min(max(2.0 / (n * (n - 1)), 10 ** log_rho), 0.5)

        gtype = random.choice(["ER", "BA", "WS", "complete"])
        if gtype == "ER":
            G = nx.erdos_renyi_graph(n, rho)
        elif gtype == "BA":
            m = max(1, min(n - 1, int(rho * (n - 1) / 2)))
            G = nx.barabasi_albert_graph(n, m)
        elif gtype == "WS":
            k = max(2, min(n - 1, int(rho * (n - 1))))
            k = k if k % 2 == 0 else k + 1
            p = random.uniform(0.1, 0.5)
            G = nx.watts_strogatz_graph(n, k, p)
        else:  # complete
            G = nx.complete_graph(n)

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue
        G = nx.convert_node_labels_to_integers(G)
        actual_rho = 2 * G.number_of_edges() / (n * (n - 1)) if n > 1 else 0
        path = output_dir / f"dense_{gtype}_{i:04d}_n{n}_r{actual_rho:.4f}.gml"
        nx.write_gml(G, str(path))
        count += 1
    print(f"[Dense Small] Generated {count} graphs in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dataset/synth_training")
    parser.add_argument("--sparse_num", type=int, default=800)
    parser.add_argument("--dense_num", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output_dir)
    generate_sparse_large(out / "sparse_large", args.sparse_num, args.seed)
    generate_dense_small(out / "dense_small", args.dense_num, args.seed)
    print("Done.")


if __name__ == "__main__":
    main()
