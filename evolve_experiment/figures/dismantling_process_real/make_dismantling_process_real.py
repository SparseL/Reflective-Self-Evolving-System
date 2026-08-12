from __future__ import annotations

import json
import re
import sys
import types
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
ANALYSIS_DIR = ROOT / "code" / "analysis"
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.append(str(ANALYSIS_DIR))
EOH_DIR = ROOT / "code" / "eoh"
if str(EOH_DIR) not in sys.path:
    sys.path.append(str(EOH_DIR))

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode
from run_cn_baselines import CODE_CI_L3, CODE_COREHD, CODE_HDA


OUT_DIR = ROOT / "evolve_experiment" / "figures" / "dismantling_process_real"
CACHE_DIR = OUT_DIR / "cache"
SOURCE_DATA_PATH = OUT_DIR / "dismantling_process_real_source_data.csv"
OUT_BASENAME_LMCC = OUT_DIR / "dismantling_process_real_panel"
OUT_BASENAME_CP = OUT_DIR / "dismantling_process_real_panel_cp_ratio"
OUT_BASENAME_CP_TIGHT = OUT_DIR / "dismantling_process_real_panel_cp_ratio_tight"
OUT_BASENAME_CP_TIGHT_FOCUS = OUT_DIR / "dismantling_process_real_panel_cp_ratio_tight_focus"

Y_FLOOR = 1e-4
STOP_LCC = 1e-2
STOP_CP_RATIO = 1e-4
MAX_REMOVAL_FRACTION = 0.7
CACHE_VERSION = 3
EARLY_FRACTION_MAX = 0.15
X_LIMITS = {
    "Enron_sampled": 0.8,
}
INSET_CONFIGS = {
    "Crime": {
        "loc": "upper right",
        "width": "38%",
        "height": "36%",
        "xrange": (0.10, 0.20),
        "xticks": [0.10, 0.15, 0.20],
        "bbox_to_anchor": (0.03, 0.05, 0.94, 0.90),
        "source_anchor": ("right", "bottom"),
        "inset_anchor": (0.0, 0.18),
    },
    "Enron_sampled": {
        "disabled": True,
    },
    "HI-II-14_sampled": {
        "disabled": True,
    },
}
FOCUS_INSET_CONFIGS = {
    "Crime": {
        "loc": "upper right",
        "width": "39%",
        "height": "39%",
        "xrange": (0.0, 0.14),
        "yrange": (0.20, 1.02),
        "bbox_to_anchor": (0.03, 0.03, 0.95, 0.93),
        "source_anchor": ("right", "bottom"),
        "inset_anchor": (0.0, 0.15),
    },
    "HI-II-14_sampled": {
        "loc": "center right",
        "width": "41%",
        "height": "39%",
        "xrange": (0.0, 0.125),
        "yrange": (0.12, 1.02),
        "bbox_to_anchor": (0.05, 0.12, 0.92, 0.82),
        "source_anchor": ("right", "top"),
        "inset_anchor": (0.0, 0.20),
    },
}

DATASETS = [
    {
        "label": "Crime",
        "dataset_name": "Crime",
        "eoh_json": ROOT / "evolve_experiment" / "transfer" / "best_algo_on_Crime.json",
    },
    {
        "label": "Enron sampled",
        "dataset_name": "Enron_sampled",
        "eoh_json": ROOT / "evolve_experiment" / "transfer" / "best_algo_on_Enron_sampled.json",
    },
    {
        "label": "HI-II-14 sampled",
        "dataset_name": "HI-II-14_sampled",
        "eoh_json": ROOT / "evolve_experiment" / "transfer" / "best_algo_on_HI-II-14_sampled.json",
    },
    {
        "label": "Youtube sampled",
        "dataset_name": "Youtube_sampled",
        "eoh_json": ROOT / "evolve_experiment" / "transfer" / "best_algo_on_youtube_sampled.json",
    },
]

METHODS = [
    ("CI(l=3)", CODE_CI_L3),
    ("CoreHD", CODE_COREHD),
    ("HDA", CODE_HDA),
    ("Evolved best", None),
]

COLORS = {
    "CI(l=3)": "#9A9A9A",
    "CoreHD": "#7DB7D9",
    "HDA": "#96C77D",
    "Evolved best": "#D64B4B",
}

MARKERS = {
    "CI(l=3)": "s",
    "CoreHD": "o",
    "HDA": "^",
    "Evolved best": "*",
}


def apply_publication_style() -> None:
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


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_code(payload) -> str:
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Empty EoH result list.")
        best_item = None
        best_objective = float("inf")
        for item in payload:
            if not isinstance(item, dict):
                continue
            objective = item.get("objective", float("inf"))
            if objective < best_objective:
                best_objective = objective
                best_item = item
        payload = best_item if best_item is not None else payload[-1]

    if isinstance(payload, dict) and isinstance(payload.get("code"), str):
        return payload["code"]

    if isinstance(payload, dict):
        raw = payload.get("llm_raw_response")
        if isinstance(raw, str):
            match = re.search(r"```python\s*(.*?)```", raw, flags=re.DOTALL)
            if match:
                return match.group(1).strip()

    raise ValueError("Cannot extract executable code from the EoH result payload.")


def cp_value(graph) -> float:
    sizes = np.array([len(comp) for comp in nx_connected_components(graph)], dtype=float)
    if sizes.size == 0:
        return 0.0
    return float((sizes * (sizes - 1) / 2).sum())


def nx_connected_components(graph):
    import networkx as nx

    return nx.connected_components(graph)


def largest_cc_fraction(graph, lcc0: int) -> float:
    comps = list(nx_connected_components(graph))
    if not comps or lcc0 <= 0:
        return 0.0
    return max(len(comp) for comp in comps) / lcc0


def cache_path(dataset_name: str, method_name: str) -> Path:
    if method_name == "Evolved best":
        method_name = "EoH-Best"
    safe_method = method_name.replace("/", "_")
    return CACHE_DIR / f"{dataset_name}__{safe_method}.json"


def try_load_cached_curve(dataset_name: str, method_name: str) -> dict | None:
    cpath = cache_path(dataset_name, method_name)
    if not cpath.exists():
        return None

    cached = load_json(cpath)
    cache_version = cached.get("cache_version")
    if cache_version not in (None, CACHE_VERSION):
        return None
    cached_method = cached.get("method")
    method_matches = cached_method == method_name or (
        method_name == "Evolved best" and cached_method == "EoH-Best"
    )
    if cached.get("dataset") != dataset_name or not method_matches:
        return None

    if cached.get("max_removal_fraction") != MAX_REMOVAL_FRACTION:
        return None

    required_keys = {"fraction_removed", "lcc_fraction", "cp_ratio", "removal_order", "n_nodes"}
    if not required_keys.issubset(cached):
        return None
    return cached


def load_selector(code_string: str):
    module = types.ModuleType("dismantling_selector")
    exec(code_string, module.__dict__)
    select_next_node = getattr(module, "select_next_node", None)
    if select_next_node is None:
        raise ValueError("The supplied code does not define select_next_node().")
    return select_next_node


def build_curve(cn: CriticalNode, dataset_name: str, method_name: str, code_string: str) -> dict:
    cached = try_load_cached_curve(dataset_name, method_name)
    if cached is not None:
        return cached

    cpath = cache_path(dataset_name, method_name)

    working_graph = cn.instances[0]
    graph = working_graph.copy()
    nodes_idx = np.arange(working_graph.number_of_nodes())
    node_mask = np.ones(working_graph.number_of_nodes(), dtype=bool)
    selector = load_selector(code_string)
    n0 = graph.number_of_nodes()
    lcc0 = max((len(comp) for comp in nx_connected_components(graph)), default=0)
    cp0 = cp_value(graph)

    fraction_removed = [0.0]
    lcc_fraction = [1.0]
    cp_ratio = [1.0]
    order: list[int] = []

    max_steps = min(n0, int(np.ceil(n0 * MAX_REMOVAL_FRACTION)))

    for step in range(1, max_steps + 1):
        unvisited_nodes = nodes_idx[node_mask]
        next_node = selector(working_graph, unvisited_nodes)
        node = int(next_node)
        if not node_mask[node]:
            raise ValueError(f"{method_name} selected node {node} repeatedly on {dataset_name}.")
        order.append(node)
        node_mask[node] = False
        graph.remove_node(node)
        fraction_removed.append(step / n0)
        lcc_fraction.append(largest_cc_fraction(graph, lcc0))
        cp_now = cp_value(graph)
        cp_ratio.append((cp_now / cp0) if cp0 > 0 else 0.0)

    curve = {
        "cache_version": CACHE_VERSION,
        "dataset": dataset_name,
        "method": method_name,
        "n_nodes": n0,
        "max_removal_fraction": MAX_REMOVAL_FRACTION,
        "removal_order": order,
        "fraction_removed": fraction_removed,
        "lcc_fraction": lcc_fraction,
        "cp_ratio": cp_ratio,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(curve, indent=2), encoding="utf-8")
    return curve


def collect_curves() -> dict[str, dict[str, dict]]:
    all_curves: dict[str, dict[str, dict]] = {}
    for dataset in DATASETS:
        dataset_name = dataset["dataset_name"]
        method_curves: dict[str, dict] = {}
        missing_methods: list[str] = []

        for method_name, baseline_code in METHODS:
            cached = try_load_cached_curve(dataset_name, method_name)
            if cached is not None:
                method_curves[method_name] = cached
            else:
                missing_methods.append(method_name)

        if missing_methods:
            eoh_code = extract_code(load_json(dataset["eoh_json"]))
            cn = CriticalNode(dataset_name=dataset_name, use_precompute=True)
            for method_name, baseline_code in METHODS:
                if method_name not in missing_methods:
                    continue
                code_string = eoh_code if method_name == "Evolved best" else baseline_code
                method_curves[method_name] = build_curve(cn, dataset_name, method_name, code_string)

        all_curves[dataset_name] = method_curves
    return all_curves


def write_source_data(curves: dict[str, dict[str, dict]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["dataset,method,step,fraction_removed,lcc_fraction,cp_ratio"]
    for dataset in DATASETS:
        dataset_name = dataset["dataset_name"]
        for method_name, _ in METHODS:
            curve = curves[dataset_name][method_name]
            for step, row in enumerate(
                zip(curve["fraction_removed"], curve["lcc_fraction"], curve["cp_ratio"])
            ):
                frac_removed, lcc_frac, cp_ratio = row
                lines.append(
                    f"{dataset_name},{method_name},{step},{frac_removed:.12f},{lcc_frac:.12f},{cp_ratio:.12f}"
                )
    SOURCE_DATA_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clip_for_log(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    return np.clip(arr, Y_FLOOR, 1.0)


def truncated_series(curve: dict, y_key: str, stop_value: float) -> tuple[list[float], np.ndarray]:
    x = curve["fraction_removed"]
    y_raw = curve[y_key]
    stop_idx = len(y_raw) - 1
    for idx, value in enumerate(y_raw):
        if value <= stop_value:
            stop_idx = idx
            break
    return x[: stop_idx + 1], clip_for_log(y_raw[: stop_idx + 1])


def early_window_limits(
    curves_for_dataset: dict[str, dict],
    y_key: str,
    stop_value: float,
    x_range: tuple[float, float],
) -> tuple[float, float] | None:
    x_min, x_max = x_range
    early_values: list[float] = []
    for method_name, _ in METHODS:
        x, y = truncated_series(curves_for_dataset[method_name], y_key, stop_value)
        for xv, yv in zip(x, y):
            if x_min <= xv <= x_max:
                early_values.append(float(yv))

    if not early_values:
        return None

    y_min = max(Y_FLOOR, min(early_values) * 0.85)
    y_max = min(1.1, max(early_values) * 1.03)
    return y_min, y_max


def panel_axis_limits(
    plotted_series: dict[str, tuple[list[float], np.ndarray]],
    dataset_name: str,
) -> tuple[float, float]:
    x_max_data = max((max(x) for x, _ in plotted_series.values() if x), default=MAX_REMOVAL_FRACTION)
    x_upper_cap = X_LIMITS.get(dataset_name, MAX_REMOVAL_FRACTION)
    x_upper = min(x_upper_cap, x_max_data * 1.08)
    x_upper = max(0.12, x_upper)

    y_min_data = min((float(np.min(y)) for _, y in plotted_series.values() if len(y) > 0), default=Y_FLOOR)
    y_upper = 1.08
    y_lower = max(Y_FLOOR, y_min_data * 0.82)
    return x_upper, y_lower if y_lower < y_upper else Y_FLOOR


def add_panel_legend(ax: plt.Axes, dataset_name: str) -> None:
    legend_kwargs = {
        "fontsize": 6.2,
        "handlelength": 1.4,
        "handletextpad": 0.55,
        "labelspacing": 0.55,
        "borderpad": 0.28,
    }
    if dataset_name in {"Crime", "HI-II-14_sampled"}:
        legend = ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.015, 0.035),
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            edgecolor="none",
            facecolor="white",
            **legend_kwargs,
        )
        legend.set_zorder(10)
        return

    ax.legend(loc="lower left", frameon=False, **legend_kwargs)


def draw_single_connector_inset(
    ax,
    inset,
    xrange: tuple[float, float],
    yrange: tuple[float, float],
    source_anchor: tuple[str, str],
    inset_anchor: tuple[float, float],
) -> None:
    x0, x1 = xrange
    y0, y1 = yrange
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#8A8A8A", linewidth=0.8)
    ax.add_patch(rect)

    x_anchor = x1 if source_anchor[0] == "right" else x0
    y_anchor = y1 if source_anchor[1] == "top" else y0
    connector = ConnectionPatch(
        xyA=(x_anchor, y_anchor),
        coordsA=ax.transData,
        xyB=inset_anchor,
        coordsB=inset.transAxes,
        axesA=ax,
        axesB=inset,
        color="#7A7A7A",
        linewidth=0.7,
        alpha=0.9,
    )
    ax.figure.add_artist(connector)


def plot_curves(
    curves: dict[str, dict[str, dict]],
    y_key: str,
    y_label: str,
    out_basename: Path,
    subtitle: str,
    stop_value: float,
    add_early_inset: bool = False,
    compact_axes: bool = False,
    inset_configs: dict[str, dict] | None = None,
) -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width_mm = 183
    height_mm = 142
    fig, axes = plt.subplots(2, 2, figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.90, wspace=0.22, hspace=0.34)

    panel_labels = ["a", "b", "c", "d"]

    for ax, dataset, panel_label in zip(axes.flat, DATASETS, panel_labels):
        dataset_name = dataset["dataset_name"]
        plotted_series: dict[str, tuple[list[float], np.ndarray]] = {}
        for method_name, _ in METHODS:
            curve = curves[dataset_name][method_name]
            x, y = truncated_series(curve, y_key, stop_value)
            plotted_series[method_name] = (x, y)
            ax.plot(
                x,
                y,
                color=COLORS[method_name],
                linewidth=1.2 if method_name == "Evolved best" else 1.0,
                marker=MARKERS[method_name],
                markersize=3.2 if method_name == "Evolved best" else 2.1,
                markevery=max(1, len(x) // 24),
                label=method_name,
                alpha=0.98,
            )

        ax.set_title(dataset["label"], fontsize=8)
        ax.set_yscale("log")
        if compact_axes:
            x_upper, y_lower = panel_axis_limits(plotted_series, dataset_name)
            ax.set_xlim(0.0, x_upper)
            ax.set_ylim(y_lower, 1.08)
        else:
            ax.set_xlim(0.0, X_LIMITS.get(dataset_name, MAX_REMOVAL_FRACTION))
            ax.set_ylim(Y_FLOOR, 1.15)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.6)
        ax.text(-0.13, 1.03, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_xlabel("Fraction of removed nodes")
        ax.set_ylabel(y_label)
        add_panel_legend(ax, dataset_name)

        if add_early_inset:
            configs = inset_configs if inset_configs is not None else INSET_CONFIGS
            inset_cfg = configs.get(
                dataset_name,
                None,
            )
            if inset_cfg is not None and not inset_cfg.get("disabled", False):
                x_range = inset_cfg.get("xrange", (0.0, inset_cfg.get("xmax", EARLY_FRACTION_MAX)))
                y_range = inset_cfg.get("yrange")
                if y_range is None:
                    limits = early_window_limits(curves[dataset_name], y_key, stop_value, x_range)
                else:
                    limits = y_range
                if limits is None:
                    continue
                inset_kwargs = {
                    "width": inset_cfg["width"],
                    "height": inset_cfg["height"],
                    "loc": inset_cfg["loc"],
                    "borderpad": 0.9,
                }
                if "bbox_to_anchor" in inset_cfg:
                    inset_kwargs["bbox_to_anchor"] = inset_cfg["bbox_to_anchor"]
                    inset_kwargs["bbox_transform"] = ax.transAxes
                    inset_kwargs["borderpad"] = 0.0
                inset = inset_axes(ax, **inset_kwargs)
                for method_name, _ in METHODS:
                    x, y = plotted_series[method_name]
                    inset.plot(
                        x,
                        y,
                        color=COLORS[method_name],
                        linewidth=1.0 if method_name == "Evolved best" else 0.9,
                        marker=MARKERS[method_name],
                        markersize=2.4 if method_name == "Evolved best" else 1.6,
                        markevery=max(1, len(x) // 28),
                        alpha=0.98,
                    )
                inset.set_xlim(*x_range)
                inset.set_ylim(*limits)
                inset.set_yscale("log")
                if "xticks" in inset_cfg:
                    inset.set_xticks(inset_cfg["xticks"])
                elif x_range[1] <= 0.16:
                    inset.set_xticks([0.0, 0.05, 0.10, 0.15])
                else:
                    inset.set_xticks([0.0, 0.05, 0.10] if x_range[1] <= 0.13 else [0.0, 0.1, 0.3, 0.5])
                inset.tick_params(axis="x", labelsize=5, length=2)
                inset.tick_params(axis="y", labelleft=False, length=0)
                inset.grid(axis="y", color="#E8E8E8", linewidth=0.45)
                inset.set_facecolor("#F5F5F5")
                for spine in inset.spines.values():
                    spine.set_linewidth(0.85)
                    spine.set_edgecolor("#8A8A8A")
                draw_single_connector_inset(
                    ax,
                    inset,
                    x_range,
                    limits,
                    inset_cfg.get("source_anchor", ("right", "bottom")),
                    inset_cfg.get("inset_anchor", (0.0, 0.15)),
                )

    fig.text(
        0.09,
        0.965,
        "Figure | Dismantling process on real networks",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    if subtitle:
        fig.text(
            0.09,
            0.935,
            subtitle,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#4D4D4D",
        )

    fig.savefig(f"{out_basename}.svg", bbox_inches="tight")
    fig.savefig(f"{out_basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_basename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    curves = collect_curves()
    write_source_data(curves)
    plot_curves(
        curves,
        y_key="lcc_fraction",
        y_label="Normalized LMCC of residual network",
        out_basename=OUT_BASENAME_LMCC,
        subtitle="Curves show the normalized LMCC of the residual network during sequential node removal.",
        stop_value=STOP_LCC,
    )
    plot_curves(
        curves,
        y_key="cp_ratio",
        y_label="Normalized pairwise connectivity",
        out_basename=OUT_BASENAME_CP,
        subtitle="",
        stop_value=STOP_CP_RATIO,
        add_early_inset=True,
    )
    plot_curves(
        curves,
        y_key="cp_ratio",
        y_label="Normalized pairwise connectivity",
        out_basename=OUT_BASENAME_CP_TIGHT,
        subtitle="",
        stop_value=STOP_CP_RATIO,
        add_early_inset=True,
        compact_axes=True,
    )
    plot_curves(
        curves,
        y_key="cp_ratio",
        y_label="Normalized pairwise connectivity",
        out_basename=OUT_BASENAME_CP_TIGHT_FOCUS,
        subtitle="",
        stop_value=STOP_CP_RATIO,
        add_early_inset=True,
        compact_axes=True,
        inset_configs=FOCUS_INSET_CONFIGS,
    )


if __name__ == "__main__":
    main()
