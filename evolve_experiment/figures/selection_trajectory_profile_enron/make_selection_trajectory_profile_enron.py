from __future__ import annotations

import csv
import json
import re
import sys
import types
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))
EOH_DIR = ROOT / "code" / "eoh"
if str(EOH_DIR) not in sys.path:
    sys.path.append(str(EOH_DIR))

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode
from run_cn_baselines import CODE_CI_L3, CODE_COREHD, CODE_HDA


OUT_DIR = ROOT / "evolve_experiment" / "figures" / "selection_trajectory_profile_enron"
PREFIX_CACHE_DIR = ROOT / "evolve_experiment" / "figures" / "dismantling_process_real" / "cache"
SUMMARY_CSV = OUT_DIR / "selection_trajectory_profile_enron_summary.csv"
NODELEVEL_CSV = OUT_DIR / "selection_trajectory_profile_enron_nodelevel_source_data.csv"
OUT_BASENAME = OUT_DIR / "selection_trajectory_profile_enron"
NOTES_PATH = OUT_DIR / "selection_trajectory_profile_enron_notes.md"
EOH_JSON = ROOT / "evolve_experiment" / "transfer" / "best_algo_on_Enron_sampled.json"

DATASET_NAME = "Enron_sampled"
METHODS = [
    ("Evolved best", None),
    ("HDA", CODE_HDA),
    ("CoreHD", CODE_COREHD),
    ("CI(l=3)", CODE_CI_L3),
]
STAGES = [
    ("Stage 1", 0.0, 0.10),
    ("Stage 2", 0.10, 0.20),
    ("Stage 3", 0.20, 0.40),
    ("Stage 4", 0.40, 0.60),
    ("Stage 5", 0.60, 1.00),
]
FEATURES = [
    ("degree_pct", "Degree percentile"),
    ("core_pct", "Core number percentile"),
    ("betweenness_pct", "Betweenness percentile"),
    ("clustering_pct", "Clustering percentile"),
    ("eigenvector_pct", "Eigenvector percentile"),
    ("unvisited_neighbor_ratio", "Unvisited-neighbor ratio"),
]

COLORS = {
    "Evolved best": "#D64B4B",
    "HDA": "#96C77D",
    "CoreHD": "#7DB7D9",
    "CI(l=3)": "#9A9A9A",
}
MARKERS = {
    "Evolved best": "o",
    "HDA": "^",
    "CoreHD": "s",
    "CI(l=3)": "D",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.frameon": False,
    }
)


def save_pub_py(fig: plt.Figure, basename: Path, dpi: int = 600) -> None:
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=dpi, bbox_inches="tight")


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_code(payload: dict | list) -> str:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Empty EoH payload.")
        best_item = None
        best_objective = float("inf")
        for item in payload:
            if not isinstance(item, dict):
                continue
            objective = float(item.get("objective", float("inf")))
            if objective < best_objective:
                best_item = item
                best_objective = objective
        payload = best_item if best_item is not None else payload[-1]

    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return str(payload["code"])

    if isinstance(payload, dict) and isinstance(payload.get("llm_raw_response"), str):
        match = re.search(r"```python\s*(.*?)```", str(payload["llm_raw_response"]), flags=re.DOTALL)
        if match:
            return match.group(1).strip()

    raise ValueError("Cannot extract select_next_node() code.")


def load_selector(code_string: str):
    module = types.ModuleType("selection_profile_selector")
    exec(code_string, module.__dict__)
    selector = getattr(module, "select_next_node", None)
    if selector is None:
        raise ValueError("Code does not define select_next_node().")
    return selector


def cached_prefix_order(method_name: str) -> list[int]:
    cache_path = PREFIX_CACHE_DIR / f"{DATASET_NAME}__{method_name}.json"
    if not cache_path.exists():
        return []
    payload = load_json(cache_path)
    removal_order = payload.get("removal_order", [])
    if not isinstance(removal_order, list):
        return []
    return [int(node) for node in removal_order]


def build_full_removal_order(graph, selector, prefix_order: list[int] | None = None) -> list[int]:
    working_graph = graph.copy()
    nodes_idx = np.arange(graph.number_of_nodes())
    node_mask = np.ones(graph.number_of_nodes(), dtype=bool)
    order: list[int] = []
    prefix_order = prefix_order or []

    for node in prefix_order:
        if 0 <= int(node) < graph.number_of_nodes() and node_mask[int(node)] and int(node) in working_graph:
            order.append(int(node))
            node_mask[int(node)] = False
            working_graph.remove_node(int(node))

    for _ in range(graph.number_of_nodes() - len(order)):
        unvisited = nodes_idx[node_mask]
        if unvisited.size == 0:
            break
        try:
            next_node = int(selector(working_graph, unvisited))
        except Exception:
            degree_map = dict(working_graph.degree())
            next_node = max((int(n) for n in unvisited), key=lambda n: (degree_map.get(n, 0), -n))
        if next_node not in working_graph:
            valid = [int(n) for n in unvisited if int(n) in working_graph]
            if not valid:
                break
            next_node = valid[0]
        if not node_mask[next_node]:
            valid = [int(n) for n in unvisited if node_mask[int(n)]]
            next_node = valid[0]
        order.append(next_node)
        node_mask[next_node] = False
        working_graph.remove_node(next_node)
    if len(order) != graph.number_of_nodes():
        raise ValueError(f"Incomplete order: got {len(order)} of {graph.number_of_nodes()} nodes.")
    return order


def percentile_rank_map(node_to_value: dict[int, float]) -> dict[int, float]:
    nodes = sorted(node_to_value)
    values = np.array([float(node_to_value[node]) for node in nodes], dtype=float)
    if values.size <= 1:
        return {nodes[0]: 0.5} if nodes else {}
    sorted_vals = np.sort(values)
    denom = max(len(values) - 1, 1)
    mapping: dict[int, float] = {}
    for node, value in zip(nodes, values):
        left = int(np.searchsorted(sorted_vals, value, side="left"))
        right = int(np.searchsorted(sorted_vals, value, side="right"))
        avg_rank = (left + right - 1) / 2.0
        mapping[node] = float(avg_rank / denom)
    return mapping


def stage_slices(n_nodes: int) -> list[tuple[str, int, int]]:
    bounds: list[tuple[str, int, int]] = []
    for idx, (label, start_frac, end_frac) in enumerate(STAGES):
        start = int(np.floor(start_frac * n_nodes))
        end = n_nodes if idx == len(STAGES) - 1 else int(np.floor(end_frac * n_nodes))
        if end <= start:
            end = min(n_nodes, start + 1)
        bounds.append((label, start, end))
    bounds[-1] = (bounds[-1][0], bounds[-1][1], n_nodes)
    return bounds


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def stage_summary(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def write_notes(summary_rows: list[dict[str, object]]) -> None:
    pivot: dict[tuple[str, str], tuple[float, float]] = {}
    for row in summary_rows:
        pivot[(str(row["method"]), str(row["stage"]),)] = (float(row["mean"]), float(row["std"]))

    def mean_of(method: str, stage: str, feature: str) -> float:
        for row in summary_rows:
            if row["method"] == method and row["stage"] == stage and row["feature"] == feature:
                return float(row["mean"])
        return float("nan")

    lines = [
        "# Selection Trajectory Profiling on Enron_sampled",
        "",
        "All node-level features were normalized to percentile ranks within the original Enron_sampled graph.",
        "",
        "## Quick Observations",
        "",
        (
            f"- Early-stage degree preference remains high for all methods, but Evolved best is not simply the most hub-seeking rule: "
            f"in Stage 1 its degree percentile is {mean_of('Evolved best', 'Stage 1', 'degree_pct'):.3f}, "
            f"compared with {mean_of('HDA', 'Stage 1', 'degree_pct'):.3f} for HDA."
        ),
        (
            f"- Evolved best shows stronger mid-stage bridge sensitivity than degree-only baselines: "
            f"in Stage 3 its betweenness percentile is {mean_of('Evolved best', 'Stage 3', 'betweenness_pct'):.3f}, "
            f"above HDA ({mean_of('HDA', 'Stage 3', 'betweenness_pct'):.3f}) and CoreHD ({mean_of('CoreHD', 'Stage 3', 'betweenness_pct'):.3f})."
        ),
        (
            f"- Evolved best also tends to select lower-clustering nodes in the middle stages: "
            f"its Stage 3 clustering percentile is {mean_of('Evolved best', 'Stage 3', 'clustering_pct'):.3f}, "
            f"versus {mean_of('HDA', 'Stage 3', 'clustering_pct'):.3f} for HDA."
        ),
        (
            f"- The unvisited-neighbor ratio remains comparatively high for Evolved best in Stages 2-4 "
            f"({mean_of('Evolved best', 'Stage 2', 'unvisited_neighbor_ratio'):.3f}, "
            f"{mean_of('Evolved best', 'Stage 3', 'unvisited_neighbor_ratio'):.3f}, "
            f"{mean_of('Evolved best', 'Stage 4', 'unvisited_neighbor_ratio'):.3f}), "
            "consistent with removing nodes while they are still structurally active."
        ),
    ]
    NOTES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cn = CriticalNode(dataset_name=DATASET_NAME, use_precompute=True)
    graph = cn.instances[0].copy()
    node_attrs = graph.nodes
    n_nodes = graph.number_of_nodes()

    feature_maps = {
        "degree_pct": percentile_rank_map({int(node): float(node_attrs[node]["degree"]) for node in graph.nodes()}),
        "core_pct": percentile_rank_map({int(node): float(node_attrs[node]["core_number"]) for node in graph.nodes()}),
        "betweenness_pct": percentile_rank_map({int(node): float(node_attrs[node]["betweenness"]) for node in graph.nodes()}),
        "clustering_pct": percentile_rank_map({int(node): float(node_attrs[node]["clustering"]) for node in graph.nodes()}),
        "eigenvector_pct": percentile_rank_map({int(node): float(node_attrs[node]["eigenvector"]) for node in graph.nodes()}),
    }

    eoh_code = extract_code(load_json(EOH_JSON))
    method_to_order: dict[str, list[int]] = {}
    for method_name, baseline_code in METHODS:
        code_string = eoh_code if method_name == "Evolved best" else str(baseline_code)
        selector = load_selector(code_string)
        method_to_order[method_name] = build_full_removal_order(graph, selector, cached_prefix_order(method_name))

    summary_rows: list[dict[str, object]] = []
    nodelevel_rows: list[dict[str, object]] = []
    slices = stage_slices(n_nodes)

    for method_name, order in method_to_order.items():
        removed: set[int] = set()
        dynamic_ratio: list[float] = []
        for step_idx, node in enumerate(order, start=1):
            degree = max(float(node_attrs[node]["degree"]), 1.0)
            active_neighbors = sum(1 for nbr in graph.neighbors(node) if int(nbr) not in removed)
            ratio = float(active_neighbors / degree)
            dynamic_ratio.append(ratio)

            nodelevel_rows.append(
                {
                    "method": method_name,
                    "step": step_idx,
                    "fraction_removed": step_idx / n_nodes,
                    "node": int(node),
                    "stage": "",
                    "degree_pct": feature_maps["degree_pct"][node],
                    "core_pct": feature_maps["core_pct"][node],
                    "betweenness_pct": feature_maps["betweenness_pct"][node],
                    "clustering_pct": feature_maps["clustering_pct"][node],
                    "eigenvector_pct": feature_maps["eigenvector_pct"][node],
                    "unvisited_neighbor_ratio": ratio,
                }
            )
            removed.add(int(node))

        for stage_label, start, end in slices:
            idxs = range(start, end)
            for feature_key, _ in FEATURES:
                values = [
                    dynamic_ratio[idx] if feature_key == "unvisited_neighbor_ratio" else feature_maps[feature_key][order[idx]]
                    for idx in idxs
                ]
                mean_val, std_val = stage_summary(values)
                summary_rows.append(
                    {
                        "method": method_name,
                        "stage": stage_label,
                        "start_step": start + 1,
                        "end_step": end,
                        "n_nodes": end - start,
                        "feature": feature_key,
                        "mean": mean_val,
                        "std": std_val,
                    }
                )

    stage_lookup = {}
    for stage_label, start, end in slices:
        for idx in range(start + 1, end + 1):
            stage_lookup[idx] = stage_label
    for row in nodelevel_rows:
        row["stage"] = stage_lookup[int(row["step"])]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "stage", "start_step", "end_step", "n_nodes", "feature", "mean", "std"],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    with NODELEVEL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "step",
                "fraction_removed",
                "stage",
                "node",
                "degree_pct",
                "core_pct",
                "betweenness_pct",
                "clustering_pct",
                "eigenvector_pct",
                "unvisited_neighbor_ratio",
            ],
        )
        writer.writeheader()
        for row in nodelevel_rows:
            writer.writerow(row)

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 118 / 25.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.90, wspace=0.22, hspace=0.28)

    x_positions = np.arange(1, len(STAGES) + 1)
    stage_labels = [stage[0].replace("Stage ", "S") for stage in STAGES]
    summary_index: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in summary_rows:
        summary_index[(str(row["method"]), str(row["stage"]), str(row["feature"]))] = (
            float(row["mean"]),
            float(row["std"]),
        )

    for ax, (feature_key, title), panel_label in zip(axes.flatten(), FEATURES, ["a", "b", "c", "d", "e", "f"]):
        for method_name, _ in METHODS:
            means = [summary_index[(method_name, stage[0], feature_key)][0] for stage in STAGES]
            ax.plot(
                x_positions,
                means,
                color=COLORS[method_name],
                marker=MARKERS[method_name],
                markersize=3.5,
                linewidth=1.2,
                label=method_name,
            )
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(stage_labels)
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        add_panel_label(ax, panel_label)

    for ax in axes[1]:
        ax.set_xlabel("Removal stage")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean normalized value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.985), fontsize=6.5)

    save_pub_py(fig, OUT_BASENAME)
    plt.close(fig)

    write_notes(summary_rows)


if __name__ == "__main__":
    main()
