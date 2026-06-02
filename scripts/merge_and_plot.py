#!/usr/bin/env python3
"""
合并 kanResilience 和 finder_tf 的结果，生成汇总表格和对比图。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None,
                        help="结果目录路径。默认使用 results/ 下最新的数字编号文件夹。")
    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # 自动查找 results/tests/ 下最新的数字编号文件夹
        tests_dir = PROJECT_ROOT / "results" / "tests"
        if tests_dir.exists():
            existing = [d for d in tests_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if existing:
                results_dir = max(existing, key=lambda d: int(d.name))
            else:
                results_dir = PROJECT_ROOT / "results"
        else:
            results_dir = PROJECT_ROOT / "results"

    kan_path = results_dir / "kanResilience_results.json"
    finder_path = results_dir / "finder_results.json"

    if not kan_path.exists():
        print(f"[Error] {kan_path} not found.")
        return
    if not finder_path.exists():
        print(f"[Error] {finder_path} not found.")
        return

    with open(kan_path) as f:
        kan_data = json.load(f)
    with open(finder_path) as f:
        finder_data = json.load(f)

    all_records = kan_data + finder_data
    df = pd.DataFrame(all_records)

    # Save CSV
    df.to_csv(results_dir / "summary.csv", index=False)

    methods = sorted(df["method"].unique())
    topologies = ["BA", "WS", "RA"]
    sizes = [50, 100, 200, 300, 400, 500]

    colors = plt.cm.tab20(np.linspace(0, 1, len(methods)))
    method_color = {m: colors[i] for i, m in enumerate(methods)}

    # Plot AUC vs n
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, topo in enumerate(topologies):
        ax = axes[idx]
        sub = df[df["topology"] == topo]
        for method in methods:
            msub = sub[sub["method"] == method].sort_values("n")
            ax.plot(msub["n"], msub["auc"], marker="o", label=method, color=method_color[method])
        ax.set_xlabel("Network Size n")
        ax.set_ylabel("AUC (lower is better)")
        ax.set_title(f"{topo} Network")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(results_dir / "auc_comparison.png", dpi=200)
    plt.close()

    # Plot rem_num vs n
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, topo in enumerate(topologies):
        ax = axes[idx]
        sub = df[df["topology"] == topo]
        for method in methods:
            msub = sub[sub["method"] == method].sort_values("n")
            ax.plot(msub["n"], msub["rem_num"], marker="s", label=method, color=method_color[method])
        ax.set_xlabel("Network Size n")
        ax.set_ylabel("rem_num (lower is better)")
        ax.set_title(f"{topo} Network")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(results_dir / "remnum_comparison.png", dpi=200)
    plt.close()

    # Plot rem_ratio vs n
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, topo in enumerate(topologies):
        ax = axes[idx]
        sub = df[df["topology"] == topo]
        for method in methods:
            msub = sub[sub["method"] == method].sort_values("n")
            ax.plot(msub["n"], msub["rem_ratio"], marker="^", label=method, color=method_color[method])
        ax.set_xlabel("Network Size n")
        ax.set_ylabel("rem_ratio (lower is better)")
        ax.set_title(f"{topo} Network")
        ax.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(results_dir / "remratio_comparison.png", dpi=200)
    plt.close()

    # Summary table printed to stdout
    print("\n" + "="*110)
    print(f"{'Topology':<8} {'n':<6} {'Method':<20} {'AUC':<10} {'rem_num':<10} {'rem_ratio':<12} {'time(s)':<10}")
    print("="*110)
    for topo in topologies:
        for n in sizes:
            sub = df[(df["topology"] == topo) & (df["n"] == n)]
            for method in methods:
                row = sub[sub["method"] == method]
                if not row.empty:
                    r = row.iloc[0]
                    auc_str = f"{r['auc']:.3f}" if pd.notna(r.get('auc', np.nan)) else "N/A"
                    rem_str = f"{int(r['rem_num'])}" if pd.notna(r.get('rem_num', np.nan)) else "N/A"
                    ratio_str = f"{r['rem_ratio']:.3f}" if pd.notna(r.get('rem_ratio', np.nan)) else "N/A"
                    time_str = f"{r['time']:.3f}" if pd.notna(r.get('time', np.nan)) else "N/A"
                    print(f"{topo:<8} {n:<6} {method:<20} {auc_str:<10} {rem_str:<10} {ratio_str:<12} {time_str:<10}")

    print(f"\nAll results saved to {results_dir}")


if __name__ == "__main__":
    main()
