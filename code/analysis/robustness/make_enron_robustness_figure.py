from __future__ import annotations

import csv
import os
import json
import math
import sys
import types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EOH_DIR = ROOT / "code" / "eoh"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(EOH_DIR) not in sys.path:
    sys.path.append(str(EOH_DIR))

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode


OUT_DIR = ROOT / "evolve_experiment" / "figures" / "robustness" / "enron_sampled"
OUT_BASENAME = OUT_DIR / "figure13_enron_robustness"
SOURCE_DATA_PATH = OUT_DIR / "figure13_enron_robustness_source_data.csv"
SUMMARY_PATH = OUT_DIR / "figure13_enron_robustness_summary.md"

GRAPH_PATH = ROOT / "dataset" / "real" / "Enron_sampled.txt"
ALGO_JSON = ROOT / "evolve_experiment" / "evolution" / "real" / "Enron_sampled_post_with_precompute_new" / "results" / "pops_best" / "Enron_sampled" / "population_generation_50.json"

LEVELS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
N_REPEATS = 1
BASE_SEED = 20260610

PERTURBATION_ORDER = ["clean", "node_delete", "edge_delete", "edge_add"]
PERTURBATION_LABELS = {
    "clean": "Clean",
    "node_delete": "Random node deletion",
    "edge_delete": "Random edge deletion",
    "edge_add": "Random noise-edge addition",
}
COLORS = {
    "clean": "#444444",
    "node_delete": "#B279A2",
    "edge_delete": "#4C78A8",
    "edge_add": "#E45756",
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
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def save_pub_py(fig: plt.Figure, basename: Path, dpi: int = 600) -> None:
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=dpi, bbox_inches="tight")


def load_graph(path: Path) -> nx.Graph:
    graph = nx.Graph()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            left, right = line.split()[:2]
            graph.add_edge(int(left), int(right))
    return graph


def load_code(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    code = str(payload.get("code", "")).strip()
    if not code:
        raise ValueError(f"No code found in {path}")
    return code


def load_clean_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    other_inf = payload.get("other_inf", {})
    return {
        "anc": float(payload.get("objective", 0.0)),
        "anc_prefix_k": float(other_inf.get("anc_prefix_k", 0.0)),
        "fht_50": float(other_inf.get("fht_50", 0.0)),
        "fht_10": float(other_inf.get("fht_10", 0.0)),
        "lcc_at_k_frac": float(other_inf.get("lcc_at_k_frac", 0.0)),
        "time_select": float(payload.get("time_select", 0.0)),
        "time_anc": float(payload.get("time_anc", 0.0)),
        "k": int(other_inf.get("k", 0)),
    }


def compute_fragility(graph: nx.Graph, avg_degree: float, degree_attr: str = "degree") -> dict[int, float]:
    fragility: dict[int, float] = {}
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            fragility[node] = 0.0
            continue
        low_degree = sum(1 for nbr in neighbors if float(graph.nodes[nbr].get(degree_attr, 0.0)) <= avg_degree)
        fragility[node] = low_degree / len(neighbors)
    return fragility


def precompute_unweighted_features(graph: nx.Graph) -> None:
    deg = dict(graph.degree())
    nx.set_node_attributes(graph, deg, "degree")
    clust = nx.clustering(graph)
    nx.set_node_attributes(graph, clust, "clustering")
    core = nx.core_number(graph)
    nx.set_node_attributes(graph, core, "core_number")

    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    deg_arr = np.array(list(deg.values()), dtype=float) if deg else np.array([0.0])
    graph.graph["precomputed"] = {
        "avg_degree": float(avg_degree),
        "avg_clustering": float(np.mean(list(clust.values())) if clust else 0.0),
        "degree_variance": float(deg_arr.var()) if deg_arr.size > 0 else 0.0,
        "max_core_k": int(max(core.values()) if core else 0),
    }
    nx.set_node_attributes(graph, compute_fragility(graph, avg_degree), "fragility")


def precompute_weighted_features(graph: nx.Graph) -> None:
    strength = dict(graph.degree(weight="weight"))
    nx.set_node_attributes(graph, strength, "degree")
    clust = nx.clustering(graph, weight="weight")
    nx.set_node_attributes(graph, clust, "clustering")
    core = nx.core_number(nx.Graph(graph))
    nx.set_node_attributes(graph, core, "core_number")

    n = graph.number_of_nodes()
    total_strength = float(sum(float(data.get("weight", 1.0)) for _, _, data in graph.edges(data=True)))
    avg_strength = (2.0 * total_strength / n) if n > 0 else 0.0
    strength_arr = np.array(list(strength.values()), dtype=float) if strength else np.array([0.0])
    graph.graph["precomputed"] = {
        "avg_degree": float(avg_strength),
        "avg_clustering": float(np.mean(list(clust.values())) if clust else 0.0),
        "degree_variance": float(strength_arr.var()) if strength_arr.size > 0 else 0.0,
        "max_core_k": int(max(core.values()) if core else 0),
    }
    nx.set_node_attributes(graph, compute_fragility(graph, avg_strength), "fragility")


def load_selector(code_string: str):
    module = types.ModuleType("heuristic_module")
    exec(code_string, module.__dict__)
    selector = getattr(module, "select_next_node", None)
    if selector is None:
        raise ValueError("select_next_node() is not defined in the supplied code.")
    return selector


def evaluate_graph(graph: nx.Graph, code_string: str) -> dict[str, float]:
    selector = load_selector(code_string)
    instance = graph.copy()
    node_order = np.array(list(instance.nodes()), dtype=int)
    node_mask = np.ones(len(node_order), dtype=bool)
    node_to_pos = {int(node): index for index, node in enumerate(node_order)}
    critical_nodes: list[int] = []

    time_select_start = __import__("time").time()
    while len(critical_nodes) < len(node_order):
        candidates = node_order[node_mask]
        if candidates.size == 0:
            break
        try:
            next_node = int(selector(instance, candidates))
        except Exception:
            degree_map = dict(instance.degree())
            next_node = max((int(node) for node in candidates), key=lambda node: (degree_map.get(node, 0), -node))
        if next_node not in node_to_pos or not node_mask[node_to_pos[next_node]]:
            remaining = [int(node) for node in candidates]
            degree_map = dict(instance.degree())
            next_node = max(remaining, key=lambda node: (degree_map.get(node, 0), -node))
        critical_nodes.append(next_node)
        node_mask[node_to_pos[next_node]] = False
    time_select = __import__("time").time() - time_select_start

    time_anc_start = __import__("time").time()
    sizes0 = np.array([len(component) for component in nx.connected_components(instance)], dtype=float)
    cp0 = float((sizes0 * (sizes0 - 1) / 2).sum()) if sizes0.size > 0 else 0.0
    instance_copy = instance.copy()
    cp_list: list[float] = []
    n_nodes = instance.number_of_nodes()
    k = max(1, int(0.15 * n_nodes))
    lcc_at_k = None
    hit50 = None
    hit10 = None
    for step, node in enumerate(critical_nodes, start=1):
        instance_copy.remove_node(node)
        comps = list(nx.connected_components(instance_copy))
        sizes = np.array([len(comp) for comp in comps], dtype=float)
        cp_j = float((sizes * (sizes - 1) / 2).sum()) if sizes.size > 0 else 0.0
        cp_list.append(cp_j)
        if lcc_at_k is None and step == k:
            lcc_at_k = float(sizes.max()) if sizes.size > 0 else 0.0
        ratio = (cp_j / cp0) if cp0 > 0 else 0.0
        if hit50 is None and ratio <= 0.5:
            hit50 = step
        if hit10 is None and ratio <= 0.1:
            hit10 = step
    anc = (sum(cp_list) / (len(critical_nodes) * cp0)) if cp0 > 0 and critical_nodes else 0.0
    anc_prefix_k = (sum(cp_list[:k]) / (k * cp0)) if cp0 > 0 and k > 0 else 0.0
    lcc_ratio = (lcc_at_k / n_nodes) if lcc_at_k is not None and n_nodes > 0 else 0.0
    time_anc = __import__("time").time() - time_anc_start
    return {
        "anc": float(anc),
        "anc_prefix_k": float(anc_prefix_k),
        "fht_50": float(hit50 if hit50 is not None else n_nodes),
        "fht_10": float(hit10 if hit10 is not None else n_nodes),
        "lcc_at_k_frac": float(lcc_ratio),
        "time_select": float(time_select),
        "time_anc": float(time_anc),
        "k": int(k),
        "n_nodes": int(n_nodes),
        "n_edges": int(instance.number_of_edges()),
    }


def perturb_delete_edges(graph: nx.Graph, fraction: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    perturbed = graph.copy()
    edges = list(perturbed.edges())
    n_remove = max(1, int(round(len(edges) * fraction)))
    chosen = rng.choice(len(edges), size=min(n_remove, len(edges)), replace=False)
    for index in chosen:
        perturbed.remove_edge(*edges[int(index)])
    return perturbed


def perturb_delete_nodes(graph: nx.Graph, fraction: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    perturbed = graph.copy()
    nodes = np.asarray(list(perturbed.nodes()))
    n_remove = max(1, int(round(len(nodes) * fraction)))
    chosen = rng.choice(nodes, size=min(n_remove, len(nodes)), replace=False)
    perturbed.remove_nodes_from([int(node) for node in chosen])
    return perturbed


def perturb_add_noise_edges(graph: nx.Graph, fraction: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    perturbed = graph.copy()
    n_add = max(1, int(round(graph.number_of_edges() * fraction)))
    nodes = np.array(list(perturbed.nodes()), dtype=int)
    existing = {tuple(sorted(edge)) for edge in perturbed.edges()}
    added = 0
    while added < n_add:
        left, right = rng.choice(nodes, size=2, replace=False)
        edge = tuple(sorted((int(left), int(right))))
        if edge in existing or edge[0] == edge[1]:
            continue
        perturbed.add_edge(*edge)
        existing.add(edge)
        added += 1
    return perturbed


def perturb_edge_weights(graph: nx.Graph, sigma: float, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    perturbed = graph.copy()
    for u, v in perturbed.edges():
        weight = float(max(0.05, 1.0 + rng.normal(0.0, sigma)))
        perturbed[u][v]["weight"] = weight
    return perturbed


def build_graph(perturbation: str, level: float, seed: int) -> nx.Graph:
    base_graph = load_graph(GRAPH_PATH)
    if perturbation == "clean":
        precompute_unweighted_features(base_graph)
        return base_graph
    if perturbation == "edge_delete":
        graph = perturb_delete_edges(base_graph, level, seed)
        precompute_unweighted_features(graph)
        return graph
    if perturbation == "node_delete":
        graph = perturb_delete_nodes(base_graph, level, seed)
        precompute_unweighted_features(graph)
        return graph
    if perturbation == "edge_add":
        graph = perturb_add_noise_edges(base_graph, level, seed)
        precompute_unweighted_features(graph)
        return graph
    raise ValueError(f"Unsupported perturbation: {perturbation}")


def evaluate_condition(task: tuple[str, float, int, int, str, float]) -> dict[str, object]:
    perturbation, level, level_percent, replicate, code, clean_anc = task
    seed = BASE_SEED + 1000 * (LEVELS.index(level) + 1) + 100 * (PERTURBATION_ORDER.index(perturbation) + 1) + (replicate - 1)
    metrics = evaluate_graph(build_graph(perturbation, level, seed), code)
    delta_anc = float(metrics["anc"]) - clean_anc
    return {
        "dataset": "Enron_sampled",
        "perturbation": perturbation,
        "perturbation_label": PERTURBATION_LABELS[perturbation],
        "level": level,
        "level_percent": level_percent,
        "replicate": replicate,
        **metrics,
        "delta_anc": delta_anc,
        "delta_anc_percent": (delta_anc / clean_anc * 100.0) if clean_anc > 0 else 0.0,
    }


def run_experiment() -> list[dict[str, object]]:
    code = load_code(ALGO_JSON)
    rows: list[dict[str, object]] = []
    clean_metrics = load_clean_metrics(ALGO_JSON)
    clean_graph = load_graph(GRAPH_PATH)
    rows.append(
        {
            "dataset": "Enron_sampled",
            "perturbation": "clean",
            "perturbation_label": PERTURBATION_LABELS["clean"],
            "level": 0.0,
            "level_percent": 0,
            "replicate": 0,
            **clean_metrics,
            "delta_anc": 0.0,
            "delta_anc_percent": 0.0,
            "n_nodes": int(clean_graph.number_of_nodes()),
            "n_edges": int(clean_graph.number_of_edges()),
        }
    )

    tasks: list[tuple[str, float, int, int, str, float]] = []
    for perturbation in ("node_delete", "edge_delete", "edge_add"):
        for level_index, level in enumerate(LEVELS):
            for replicate in range(N_REPEATS):
                tasks.append(
                    (
                        perturbation,
                        level,
                        int(round(level * 100)),
                        replicate + 1,
                        code,
                        float(clean_metrics["anc"]),
                    )
                )
    max_workers = min(len(tasks), max(1, min(6, (os.cpu_count() or 1))))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        rows.extend(executor.map(evaluate_condition, tasks))
    rows.sort(key=lambda row: (PERTURBATION_ORDER.index(str(row["perturbation"])), int(row["level_percent"]), int(row["replicate"])))
    return rows


def write_source_data(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "perturbation",
                "perturbation_label",
                "level",
                "level_percent",
                "replicate",
                "anc",
                "delta_anc",
                "delta_anc_percent",
                "anc_prefix_k",
                "fht_50",
                "fht_10",
                "lcc_at_k_frac",
                "time_select",
                "time_anc",
                "k",
                "n_nodes",
                "n_edges",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_source_data(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "dataset": str(row["dataset"]),
                    "perturbation": str(row["perturbation"]),
                    "perturbation_label": str(row["perturbation_label"]),
                    "level": float(row["level"]),
                    "level_percent": int(float(row["level_percent"])),
                    "replicate": int(float(row["replicate"])),
                    "anc": float(row["anc"]),
                    "delta_anc": float(row["delta_anc"]),
                    "delta_anc_percent": float(row["delta_anc_percent"]),
                    "anc_prefix_k": float(row["anc_prefix_k"]),
                    "fht_50": float(row["fht_50"]),
                    "fht_10": float(row["fht_10"]),
                    "lcc_at_k_frac": float(row["lcc_at_k_frac"]),
                    "time_select": float(row["time_select"]),
                    "time_anc": float(row["time_anc"]),
                    "k": int(float(row["k"])),
                    "n_nodes": int(float(row["n_nodes"])),
                    "n_edges": int(float(row["n_edges"])),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in source data: {path}")
    rows.sort(key=lambda row: (PERTURBATION_ORDER.index(str(row["perturbation"])), int(row["level_percent"]), int(row["replicate"])))
    return rows


def group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, int], list[dict[str, object]]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["perturbation"]), int(row["level_percent"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def write_summary(rows: list[dict[str, object]]) -> None:
    clean = next(row for row in rows if str(row["perturbation"]) == "clean")
    grouped = group_rows([row for row in rows if str(row["perturbation"]) != "clean"])
    lines = [
        "# Enron_sampled robustness summary",
        "",
        f"- Clean ANC: `{float(clean['anc']):.4f}`",
        f"- Clean ANC Prefix K: `{float(clean['anc_prefix_k']):.4f}`",
        f"- Clean FHT50: `{int(float(clean['fht_50']))}`",
        f"- Clean FHT10: `{int(float(clean['fht_10']))}`",
        f"- Clean LCC@K ratio: `{float(clean['lcc_at_k_frac']):.4f}`",
        "",
    ]
    for perturbation in ("node_delete", "edge_delete", "edge_add"):
        lines.append(f"## {PERTURBATION_LABELS[perturbation]}")
        lines.append("")
        for level_percent in [1, 3, 5, 10, 15, 20]:
            items = grouped[(perturbation, level_percent)]
            anc_mean = float(np.mean([float(item["anc"]) for item in items]))
            delta_mean = float(np.mean([float(item["delta_anc_percent"]) for item in items]))
            lines.append(f"- {level_percent}%: ANC `{anc_mean:.4f}`; relative change vs clean `{delta_mean:+.2f}%`")
        lines.append("")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def plot_figure(rows: list[dict[str, object]]) -> None:
    clean = next(row for row in rows if str(row["perturbation"]) == "clean")
    clean_anc = float(clean["anc"])
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 64 / 25.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.24, top=0.84, wspace=0.18)

    y_values = [float(row["anc"]) for row in rows]
    y_min = min(y_values)
    y_max = max(y_values)
    y_pad = max((y_max - y_min) * 0.25, 0.005)

    for ax, perturbation in zip(axes, ("node_delete", "edge_delete", "edge_add")):
        subset = [row for row in rows if str(row["perturbation"]) == perturbation]
        x = np.array([int(row["level_percent"]) for row in subset], dtype=float)
        y = np.array([float(row["anc"]) for row in subset], dtype=float)
        ax.plot(
            x,
            y,
            color=COLORS[perturbation],
            linewidth=1.6,
            marker="o",
            markersize=4.5,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markeredgecolor=COLORS[perturbation],
            zorder=3,
        )
        ax.axhline(clean_anc, color=COLORS["clean"], linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
        ax.set_title(PERTURBATION_LABELS[perturbation], fontsize=7.5)
        ax.set_xlim(0.6, 20.4)
        ax.set_xticks([1, 3, 5, 10, 15, 20])
        ax.set_xlabel("Perturbation level (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.25)
        for point_index, row in enumerate(subset):
            offset_x = -8 if point_index == len(subset) - 1 else (6 if point_index == 0 else 0)
            offset_y = 12 if point_index % 2 == 0 else -16
            ax.annotate(f"{float(row['anc']):.4f}", xy=(int(row["level_percent"]), float(row["anc"])), xytext=(offset_x, offset_y), textcoords="offset points", ha="right" if offset_x < 0 else ("left" if offset_x > 0 else "center"), va="bottom" if offset_y > 0 else "top", fontsize=5.8, color=COLORS[perturbation], bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.82), arrowprops=dict(arrowstyle="-", color=COLORS[perturbation], linewidth=0.55, shrinkA=2, shrinkB=3), annotation_clip=False)
    axes[0].set_ylabel("ANC")
    axes[0].set_ylim(y_min - y_pad * 0.25, y_max + y_pad * 0.55)

    fig.text(
        0.08,
        0.93,
        "Figure | Robustness of the Enron_sampled Evolved best heuristic under graph perturbations",
        fontsize=8,
        fontweight="bold",
        ha="left",
    )
    # fig.text(
    #     0.08,
    #     0.885,
    #     "The figure is rendered from precomputed robustness metrics; lower ANC indicates better dismantling quality.",
    #     fontsize=6.4,
    #     color="#4D4D4D",
    #     ha="left",
    # )
    save_pub_py(fig, OUT_BASENAME)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required_levels = {int(round(level * 100)) for level in LEVELS}
    existing_levels: set[int] = set()
    if SOURCE_DATA_PATH.exists():
        try:
            existing_levels = {
                int(row["level_percent"])
                for row in load_source_data(SOURCE_DATA_PATH)
                if str(row["perturbation"]) != "clean"
            }
        except (KeyError, ValueError):
            existing_levels = set()
    existing_perturbations = set()
    if SOURCE_DATA_PATH.exists():
        try:
            existing_perturbations = {str(row["perturbation"]) for row in load_source_data(SOURCE_DATA_PATH)}
        except (KeyError, ValueError):
            existing_perturbations = set()
    if not SOURCE_DATA_PATH.exists() or not required_levels.issubset(existing_levels) or not set(PERTURBATION_ORDER).issubset(existing_perturbations):
        rows = run_experiment()
        write_source_data(rows)
    else:
        rows = load_source_data(SOURCE_DATA_PATH)
        filtered_rows = [row for row in rows if str(row["perturbation"]) in set(PERTURBATION_ORDER)]
        if len(filtered_rows) != len(rows):
            rows = filtered_rows
            write_source_data(rows)
    write_summary(rows)
    plot_figure(rows)


if __name__ == "__main__":
    main()
