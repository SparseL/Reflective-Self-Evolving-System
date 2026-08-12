from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ENRON_SCRIPT = ROOT / "code" / "analysis" / "robustness" / "make_enron_robustness_figure.py"
spec = importlib.util.spec_from_file_location("enron_robustness", ENRON_SCRIPT)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load shared robustness implementation: {ENRON_SCRIPT}")
shared = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = shared
spec.loader.exec_module(shared)


DEFAULT_GRAPH_DIR = ROOT / "dataset" / "synthetic" / "sbm" / "1000"
DEFAULT_ALGO_DIR = ROOT / "evolve_experiment" / "evolution" / "robustness_algorithm_sources" / "synthetic_sbm_1000_state_soft"
DEFAULT_OUT_DIR = ROOT / "evolve_experiment" / "figures" / "robustness" / "synthetic_sbm_1000"
DEFAULT_DATASET = "synthetic_sbm_1000"
LEVELS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
LABEL_LEVELS = {5, 10, 15, 20}
PERTURBATIONS = ("node_delete", "edge_delete", "edge_add")
LABELS = {
    "node_delete": "Random node deletion",
    "edge_delete": "Random edge deletion",
    "edge_add": "Random noise-edge addition",
}
COLORS = {"node_delete": "#B279A2", "edge_delete": "#4C78A8", "edge_add": "#E45756"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness perturbation experiment for evolved-best SBM heuristics.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--algo-dir", type=Path, default=DEFAULT_ALGO_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 3, 5, 10, 15, 20])
    parser.add_argument("--repeats", type=int, default=1)
    return parser.parse_args()


def load_graph(path: Path) -> nx.Graph:
    graph = nx.read_gml(path)
    return nx.convert_node_labels_to_integers(graph, ordering="default")


def load_latest_algorithm(algo_dir: Path) -> tuple[Path, str]:
    files = sorted(algo_dir.glob("population_generation_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    if not files:
        raise FileNotFoundError(f"No population_generation_*.json found in {algo_dir}")
    path = files[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    code = str(payload.get("code", "")).strip()
    if not code:
        raise ValueError(f"No code field found in {path}")
    return path, code


def perturb_graph(graph: nx.Graph, perturbation: str, level: float, seed: int) -> nx.Graph:
    if perturbation == "node_delete":
        return shared.perturb_delete_nodes(graph, level, seed)
    if perturbation == "edge_delete":
        return shared.perturb_delete_edges(graph, level, seed)
    if perturbation == "edge_add":
        return shared.perturb_add_noise_edges(graph, level, seed)
    raise ValueError(perturbation)


def add_unit_weights(graph: nx.Graph) -> nx.Graph:
    weighted = graph.copy()
    for u, v in weighted.edges():
        weighted[u][v]["weight"] = 1.0
    return weighted


def precompute_weighted_sbm_features(graph: nx.Graph) -> None:
    """Prepare weighted features while preserving the original evolved-best code."""
    strength = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph, weight="weight")
    core = nx.core_number(nx.Graph(graph))
    try:
        eigenvector = nx.eigenvector_centrality(graph, max_iter=5000, weight="weight")
    except nx.NetworkXException:
        eigenvector = nx.eigenvector_centrality_numpy(graph, weight="weight")
    nx.set_node_attributes(graph, strength, "degree")
    nx.set_node_attributes(graph, clustering, "clustering")
    nx.set_node_attributes(graph, core, "core_number")
    nx.set_node_attributes(graph, eigenvector, "eigenvector")
    strengths = np.asarray(list(strength.values()), dtype=float)
    graph.graph["precomputed"] = {
        "avg_degree": float(strengths.mean()) if strengths.size else 0.0,
        "degree_variance": float(strengths.var()) if strengths.size else 0.0,
        "max_core_k": int(max(core.values()) if core else 0),
    }


def evaluate(graph: nx.Graph, code: str, weighted: bool = False) -> dict[str, float]:
    if weighted:
        precompute_weighted_sbm_features(graph)
    else:
        shared.precompute_unweighted_features(graph)
    return shared.evaluate_graph(graph, code)


def run(args: argparse.Namespace) -> tuple[list[dict[str, object]], Path]:
    graph_paths = sorted(args.graph_dir.glob("g_*"), key=lambda p: int(p.name.split("_", 1)[1]))
    if not graph_paths:
        raise FileNotFoundError(f"No g_* SBM instances found in {args.graph_dir}")
    algo_path, code = load_latest_algorithm(args.algo_dir)
    rows: list[dict[str, object]] = []
    seed_base = 20260725
    for graph_index, graph_path in enumerate(graph_paths):
        clean_graph = load_graph(graph_path)
        clean = evaluate(clean_graph.copy(), code, weighted=False)
        rows.append({"dataset": args.dataset, "instance": graph_path.name, "perturbation": "clean", "evaluation_mode": "unweighted_original", "level_percent": 0, "replicate": 0, **clean, "delta_anc_percent": 0.0})
        for perturbation in PERTURBATIONS:
            for level_percent in args.levels:
                level = level_percent / 100.0
                for replicate in range(1, args.repeats + 1):
                    seed = seed_base + graph_index * 100000 + level_percent * 100 + replicate
                    metrics = evaluate(perturb_graph(clean_graph, perturbation, level, seed), code, weighted=False)
                    reference = clean
                    delta = float(metrics["anc"]) - float(reference["anc"])
                    rows.append({
                        "dataset": args.dataset,
                        "instance": graph_path.name,
                        "perturbation": perturbation,
                        "evaluation_mode": "unweighted_original",
                        "level_percent": level_percent,
                        "replicate": replicate,
                        **metrics,
                        "delta_anc_percent": delta / float(reference["anc"]) * 100.0 if float(reference["anc"]) else 0.0,
                    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sbm_robustness_algorithm_source.txt").write_text(str(algo_path) + "\n", encoding="utf-8")
    return rows, algo_path


def write_outputs(rows: list[dict[str, object]], out_dir: Path, algo_path: Path) -> None:
    source = out_dir / "figure_sbm_robustness_source_data.csv"
    fields = ["dataset", "instance", "perturbation", "evaluation_mode", "level_percent", "replicate", "anc", "delta_anc_percent", "anc_prefix_k", "fht_50", "fht_10", "lcc_at_k_frac", "time_select", "time_anc", "k"]
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    clean = [r for r in rows if r["perturbation"] == "clean"]
    lines = [
        f"# {clean[0]['dataset'] if clean else 'synthetic'} robustness summary",
        "",
        f"- Original evolved-best source: `{algo_path}`",
        f"- Number of SBM instances: `{len(clean)}`",
        "",
        "The original evolved-best code is preserved. Both perturbation tests use the original unweighted feature context and unweighted ANC.",
        "",
    ]
    for perturbation in PERTURBATIONS:
        lines += [f"## {LABELS[perturbation]}", ""]
        for level in sorted({int(r["level_percent"]) for r in rows if r["perturbation"] == perturbation}):
            subset = [r for r in rows if r["perturbation"] == perturbation and int(r["level_percent"]) == level]
            lines.append(f"- {level}%: ANC `{np.mean([float(r['anc']) for r in subset]):.4f}`; relative change vs instance clean `{np.mean([float(r['delta_anc_percent']) for r in subset]):+.2f}%`")
        lines.append("")
    (out_dir / "figure_sbm_robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")
    dataset_name = str(clean[0].get("dataset", "synthetic")) if clean else "synthetic"
    caption = (
        f"# Figure | Robustness of the {dataset_name} evolved-best heuristic under node and edge perturbations\n\n"
        f"**a–c,** ANC trajectories for random node deletion, random edge deletion and random noise-edge addition, respectively. "
        f"The x axis denotes perturbation level (1%, 3%, 5%, 10%, 15% and 20%), and the y axis denotes ANC; lower values indicate better dismantling performance. "
        f"Each open circle is the mean ANC across the 10 synthetic graph instances at that perturbation level, and each numeric callout reports the exact mean ANC to four decimal places. "
        f"Callouts are placed outside the trajectory and connected to their markers by thin leader lines; they are not relative changes from baseline. "
        f"The dashed line denotes the mean clean-graph ANC. Node deletion removes the specified fraction of nodes before evaluating the fixed evolved-best heuristic; edge deletion removes existing edges; noise-edge addition adds absent edges while retaining the node set. "
        f"One random replicate is used per instance, level and perturbation type (`n = 10` instances in the aggregate), so no within-condition error bars are shown. Source data are provided as a Source Data file."
    )
    (out_dir / "figure_synthetic_robustness_caption.md").write_text(caption, encoding="utf-8")


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 88 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.21, top=0.79, wspace=0.18)
    all_anc = [float(r["anc"]) for r in rows]
    y_min, y_max = min(all_anc), max(all_anc)
    y_pad = max((y_max - y_min) * 0.25, 0.005)
    for ax, perturbation in zip(axes, PERTURBATIONS):
        clean_anc = float(np.mean([float(r["anc"]) for r in rows if r["perturbation"] == "clean"]))
        levels = sorted({int(r["level_percent"]) for r in rows if r["perturbation"] == perturbation})
        x = np.array(levels, dtype=float)
        y = np.array([np.mean([float(r["anc"]) for r in rows if r["perturbation"] == perturbation and int(r["level_percent"]) == level]) for level in levels])
        ax.plot(x, y, color=COLORS[perturbation], linewidth=1.6, marker="o", markersize=4.5, markerfacecolor="white", markeredgewidth=1.0)
        ax.axhline(clean_anc, color="#444444", linewidth=1.0, linestyle=(0, (3, 2)))
        ax.set_title(LABELS[perturbation], fontsize=7.5)
        ax.set_xlim(0.6, max(levels) + 0.4)
        ax.set_xticks(levels)
        ax.set_xlabel("Perturbation level (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        for point_index, (level, value) in enumerate(zip(levels, y)):
            if level not in LABEL_LEVELS:
                continue
            ax.annotate(f"{value:.4f}", xy=(level, value), xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=5.8, color=COLORS[perturbation], bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88), annotation_clip=False)
    axes[0].set_ylabel("ANC")
    axes[0].set_ylim(y_min - y_pad * 0.25, y_max + y_pad * 2.2)
    dataset = str(next(row for row in rows if row["perturbation"] == "clean").get("dataset", "synthetic"))
    fig.text(0.08, 0.93, f"Figure | Robustness of the {dataset} evolved-best heuristic", fontsize=8, fontweight="bold", ha="left")
    basename = out_dir / "figure_sbm_robustness"
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = args.out_dir / "figure_sbm_robustness_source_data.csv"
    algo_path = sorted(args.algo_dir.glob("population_generation_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))[-1] if list(args.algo_dir.glob("population_generation_*.json")) else args.algo_dir / "population_generation_latest.json"
    if source.exists():
        rows = []
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                if raw["perturbation"] not in {"clean", *PERTURBATIONS}:
                    continue
                row: dict[str, object] = {
                    "dataset": raw.get("dataset", args.dataset),
                    "instance": raw["instance"],
                    "perturbation": raw["perturbation"],
                    "evaluation_mode": raw.get("evaluation_mode", "unweighted_original"),
                    "level_percent": int(float(raw["level_percent"])),
                    "replicate": int(float(raw["replicate"])),
                }
                for field in ("anc", "delta_anc_percent", "anc_prefix_k", "fht_50", "fht_10", "lcc_at_k_frac", "time_select", "time_anc"):
                    row[field] = float(raw[field])
                row["k"] = int(float(raw["k"]))
                rows.append(row)
        if not rows or not {int(x) for x in args.levels}.issubset({int(r["level_percent"]) for r in rows if r["perturbation"] != "clean"}) or not set(PERTURBATIONS).issubset({str(r["perturbation"]) for r in rows}):
            rows, algo_path = run(args)
    else:
        rows, algo_path = run(args)
    write_outputs(rows, args.out_dir, algo_path)
    plot(rows, args.out_dir)
    print(f"Completed SBM robustness experiment: {len(rows)} rows")
    print(f"Algorithm: {algo_path}")
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
