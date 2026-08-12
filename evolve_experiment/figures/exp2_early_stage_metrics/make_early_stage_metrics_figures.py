from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec


PACKAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
ROOT = PACKAGE_ROOT
OUT_DIR = PACKAGE_ROOT / "evolve_experiment" / "figures" / "exp2_early_stage_metrics"
SOURCE_DATA_FILE = OUT_DIR / "early_stage_metrics_source_data.csv"

REAL_DATASET_ORDER = [
    "Enron",
    "Enron_sample",
    "HI-II-14",
    "HI-II-14_sampled",
    "Youtube_sampled",
    "Youtube_sample4000",
    "Youtube_sample8000",
    "Crime",
]

METHODS = ["CI(l=3)", "CoreHD", "HDA", "Evolved best"]
METHOD_ORDER = ["CI(l=3)", "CoreHD", "HDA", "Evolved best"]

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

DISPLAY_NAMES = {
    "Crime": "Crime",
    "Enron": "Enron",
    "Enron_sample": "Enron\nsample",
    "HI-II-14": "HI-II-14",
    "HI-II-14_sampled": "HI-II-14\nsampled",
    "Youtube_sampled": "YouTube\nsampled",
    "Youtube_sample4000": "YouTube\n4k",
    "Youtube_sample8000": "YouTube\n8k",
}


def is_sample_dataset(dataset: str) -> bool:
    return "sample" in dataset.lower()


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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_data(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "fht_50": float(row["fht_50"]),
                    "fht_10": float(row["fht_10"]),
                    "anc_prefix_k": float(row["anc_prefix_k"]),
                }
            )
    return rows


def read_real_eoh() -> dict[str, dict]:
    data: dict[str, dict] = {}

    crime = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_Crime.json")
    data["Crime"] = {**crime["other_inf"], "method": "Evolved best"}

    enron_sample = load_json(
        ROOT
        / "evolve_experiment"
        / "evolution"
        / "real"
        / "Enron_sampled_post_with_precompute_new"
        / "results"
        / "pops_best"
        / "Enron_sampled"
        / "population_generation_50.json"
    )
    data["Enron_sample"] = {**enron_sample["other_inf"], "method": "Evolved best"}

    enron = load_json(
        ROOT
        / "evolve_experiment"
        / "transfer"
        / "best_algo_on_original_Enron_from_Enron_sampled_gen50.json"
    )
    data["Enron"] = {**enron["other_inf"], "method": "Evolved best"}

    hi_sample = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_HI-II-14_sampled.json")
    data["HI-II-14_sampled"] = {**hi_sample["other_inf"], "method": "Evolved best"}

    hi_orig = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_original_HI-II-14.json")
    data["HI-II-14"] = {**hi_orig["other_inf"], "method": "Evolved best"}

    yt_sample = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_youtube_sampled.json")
    data["Youtube_sampled"] = {**yt_sample["other_inf"], "method": "Evolved best"}

    yt4000 = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_youtube4000.json")
    data["Youtube_sample4000"] = {**yt4000["result"][1], "method": "Evolved best"}

    yt8000 = load_json(ROOT / "evolve_experiment" / "transfer" / "best_algo_on_youtube8000.json")
    data["Youtube_sample8000"] = {**yt8000["result"][1], "method": "Evolved best"}

    return data


def read_real_baselines() -> dict[str, dict[str, dict]]:
    root = ROOT / "baseline_results_real"
    result: dict[str, dict[str, dict]] = {}

    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue

        rows = {}
        for json_file in sorted(ds_dir.glob("*.json")):
            obj = load_json(json_file)
            metrics = obj["metrics"]
            rows[obj["algorithm"]] = metrics

        result[ds_dir.name] = rows

    return result


def get_metric(metrics: dict, key: str) -> float | None:
    val = metrics.get(key)
    if val is None:
        return None

    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def build_source_data(
    datasets: list[str],
    eoh: dict[str, dict],
    baselines: dict[str, dict[str, dict]],
) -> list[dict]:
    rows: list[dict] = []

    for ds in datasets:
        for method in METHOD_ORDER:
            if method == "Evolved best":
                metrics = eoh.get(ds, {})
            else:
                metrics = baselines.get(ds, {}).get(method, {})

            rows.append(
                {
                    "dataset": ds,
                    "method": method,
                    "fht_50": get_metric(metrics, "fht_50"),
                    "fht_10": get_metric(metrics, "fht_10"),
                    "anc_prefix_k": get_metric(metrics, "anc_prefix_k"),
                }
            )

    return rows


def write_source_data_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["dataset", "method", "fht_50", "fht_10", "anc_prefix_k"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def make_pointplot(
    ax: plt.Axes,
    x: np.ndarray,
    y_map: dict[str, np.ndarray],
    ylabel: str,
    sample_flags: np.ndarray,
    size_scale: float = 1.0,
) -> dict[str, float]:
    offsets = np.linspace(-0.24, 0.24, num=len(METHOD_ORDER))
    offset_map = {method: float(offsets[i]) for i, method in enumerate(METHOD_ORDER)}

    for i, method in enumerate(METHOD_ORDER):
        y = y_map[method]
        mask = np.isfinite(y)

        if not np.any(mask):
            continue

        original_mask = mask & (~sample_flags)
        sample_mask = mask & sample_flags

        if np.any(original_mask):
            ax.scatter(
                x[original_mask] + offsets[i],
                y[original_mask],
                s=(44 if method != "Evolved best" else 66) * size_scale,
                marker=MARKERS[method],
                facecolor=COLORS[method],
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

        if np.any(sample_mask):
            ax.scatter(
                x[sample_mask] + offsets[i],
                y[sample_mask],
                s=(34 if method != "Evolved best" else 52) * size_scale,
                marker=MARKERS[method],
                facecolors="none",
                edgecolors=COLORS[method],
                linewidth=1.1,
                zorder=4,
            )

    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=0)

    return offset_map


def format_delta(metric: str, delta: float) -> str:
    sign = "+" if delta > 0 else ""

    if metric in {"fht_50", "fht_10"}:
        return f"Δ={sign}{int(round(delta))}"

    return f"Δ={sign}{delta:.3f}"


def annotate_key_datasets(
    ax: plt.Axes,
    datasets: list[str],
    x: np.ndarray,
    offset_map: dict[str, float],
    y_map: dict[str, np.ndarray],
    metric: str,
    key_datasets: list[str],
) -> None:
    baseline_methods = [m for m in METHOD_ORDER if m != "Evolved best"]

    for ds in key_datasets:
        if ds not in datasets:
            continue

        idx = datasets.index(ds)
        eoh_val = float(y_map["Evolved best"][idx])

        if not np.isfinite(eoh_val):
            continue

        baseline_vals = [
            float(y_map[m][idx])
            for m in baseline_methods
            if np.isfinite(float(y_map[m][idx]))
        ]

        if not baseline_vals:
            continue

        best_baseline = min(baseline_vals)
        delta = best_baseline - eoh_val

        x0 = float(x[idx] + offset_map["Evolved best"])
        y0 = eoh_val

        ax.annotate(
            format_delta(metric, delta),
            xy=(x0, y0),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=7,
            color=COLORS["Evolved best"],
            ha="left",
            va="bottom",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["Evolved best"],
                "lw": 0.6,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )


def make_combined_figure(
    datasets: list[str],
    x: np.ndarray,
    xtick_labels: list[str],
    sample_flags: np.ndarray,
    y_fht50: dict[str, np.ndarray],
    y_fht10: dict[str, np.ndarray],
    y_prefix: dict[str, np.ndarray],
) -> None:
    fig = plt.figure(figsize=(8.8, 5.9), constrained_layout=False)

    gs = GridSpec(
        2,
        2,
        figure=fig,
        height_ratios=[1.0, 1.08],
        hspace=0.46,
        wspace=0.30,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    offsets_a = make_pointplot(
        ax_a,
        x,
        y_fht50,
        ylabel="Nodes removed (FHT50, lower is better)",
        sample_flags=sample_flags,
    )

    offsets_b = make_pointplot(
        ax_b,
        x,
        y_fht10,
        ylabel="Nodes removed (FHT10, lower is better; log scale)",
        sample_flags=sample_flags,
    )

    offsets_c = make_pointplot(
        ax_c,
        x,
        y_prefix,
        ylabel="ANC Prefix K (first 15% removals, lower is better)",
        sample_flags=sample_flags,
        size_scale=1.35,
    )

    ax_a.set_title("FHT50")
    ax_b.set_title("FHT10")
    ax_c.set_title("Early-stage dismantling quality (K = 15% nodes removed)")

    ax_b.set_yscale("log")
    ax_b.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax_b.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax_b.yaxis.set_minor_formatter(NullFormatter())
    ax_b.yaxis.set_major_formatter(ScalarFormatter())

    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
        ax.set_xlim(-0.6, len(datasets) - 0.4)

    add_panel_label(ax_a, "a")
    add_panel_label(ax_b, "b")
    add_panel_label(ax_c, "c")

    annotate_key_datasets(
        ax_a,
        datasets,
        x,
        offsets_a,
        y_fht50,
        metric="fht_50",
        key_datasets=["HI-II-14_sampled", "Youtube_sampled"],
    )

    annotate_key_datasets(
        ax_b,
        datasets,
        x,
        offsets_b,
        y_fht10,
        metric="fht_10",
        key_datasets=["HI-II-14_sampled", "Youtube_sampled"],
    )

    annotate_key_datasets(
        ax_c,
        datasets,
        x,
        offsets_c,
        y_prefix,
        metric="anc_prefix_k",
        key_datasets=["HI-II-14_sampled", "Youtube_sampled"],
    )

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS[m],
            color="none",
            markerfacecolor=COLORS[m],
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=7 if m != "Evolved best" else 9,
            linewidth=0,
            label=m,
        )
        for m in METHOD_ORDER
    ]

    network_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#333333",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            markersize=5.5,
            linewidth=0,
            label="Original network",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="#333333",
            markeredgewidth=1.0,
            markersize=5.5,
            linewidth=0,
            label="Sampled network",
        ),
    ]

    legend1 = fig.legend(
        handles=method_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.060),
        columnspacing=1.25,
        handletextpad=0.4,
    )

    legend2 = fig.legend(
        handles=network_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.018),
        columnspacing=1.5,
        handletextpad=0.45,
    )

    fig.add_artist(legend1)

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.925,
        bottom=0.205,
    )

    out = OUT_DIR / "figure2_3_combined"

    fig.savefig(f"{out}.svg", bbox_inches="tight")
    fig.savefig(f"{out}.pdf", bbox_inches="tight")
    fig.savefig(f"{out}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{out}.tiff", dpi=600, bbox_inches="tight")

    plt.close(fig)

    print("Wrote:", out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    datasets = REAL_DATASET_ORDER
    if SOURCE_DATA_FILE.exists():
        source_rows = load_source_data(SOURCE_DATA_FILE)
    else:
        eoh = read_real_eoh()
        baselines = read_real_baselines()
        source_rows = build_source_data(datasets, eoh, baselines)
        write_source_data_csv(source_rows, SOURCE_DATA_FILE)

    x = np.arange(len(datasets), dtype=float)
    xtick_labels = [DISPLAY_NAMES.get(ds, ds) for ds in datasets]
    sample_flags = np.array([is_sample_dataset(ds) for ds in datasets], dtype=bool)

    y_fht50 = {}
    y_fht10 = {}
    y_prefix = {}

    for method in METHOD_ORDER:
        vals_50 = []
        vals_10 = []
        vals_p = []

        for ds in datasets:
            row = next(
                (
                    r
                    for r in source_rows
                    if r["dataset"] == ds and r["method"] == method
                ),
                None,
            )

            if row is None:
                vals_50.append(np.nan)
                vals_10.append(np.nan)
                vals_p.append(np.nan)
            else:
                vals_50.append(row["fht_50"] if row["fht_50"] is not None else np.nan)
                vals_10.append(row["fht_10"] if row["fht_10"] is not None else np.nan)
                vals_p.append(row["anc_prefix_k"] if row["anc_prefix_k"] is not None else np.nan)

        y_fht50[method] = np.array(vals_50, dtype=float)
        y_fht10[method] = np.array(vals_10, dtype=float)
        y_prefix[method] = np.array(vals_p, dtype=float)

    make_combined_figure(
        datasets=datasets,
        x=x,
        xtick_labels=xtick_labels,
        sample_flags=sample_flags,
        y_fht50=y_fht50,
        y_fht10=y_fht10,
        y_prefix=y_prefix,
    )

    print("Source data:", SOURCE_DATA_FILE)


if __name__ == "__main__":
    main()
