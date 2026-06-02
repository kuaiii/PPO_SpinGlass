"""将 .gt.xz 文件批量转换为 .gml 文件。"""
import sys
from pathlib import Path
import subprocess

MICROMAMBA = "/tmp/bin/micromamba"
MAMBA_ENV = "gt"


def convert_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files = sorted(input_path.glob("*.gt.xz"))
    print(f"Found {len(files)} files in {input_dir}")

    for f in files:
        out_file = output_path / (f.stem + ".gml")
        if out_file.exists():
            continue
        py_code = f"""
import graph_tool as gt
import networkx as nx
g = gt.load_graph('{f}')
nxg = nx.Graph()
for v in g.vertices():
    nxg.add_node(int(v))
for e in g.edges():
    nxg.add_edge(int(e.source()), int(e.target()))
nx.write_gml(nxg, '{out_file}')
print('OK', '{out_file}', nxg.number_of_nodes(), nxg.number_of_edges())
"""
        cmd = [MICROMAMBA, "run", "-n", MAMBA_ENV, "python", "-c", py_code]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR converting {f.name}: {result.stderr.strip()}")
        else:
            print(result.stdout.strip())


if __name__ == "__main__":
    convert_directory("dataset/test_review", "dataset/test_converted")
    convert_directory("dataset/test_synth", "dataset/test_synth_converted")
    convert_directory("dataset/test_lfr", "dataset/test_lfr_converted")
