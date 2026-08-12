from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
OUT_DIR = PACKAGE_ROOT / "evolve_experiment" / "figures" / "exp7_ws_scale_generalization"
OUT_BASENAME = OUT_DIR / "figure7_ws_cross_scale_generalization"
SOURCE_DATA_PATH = OUT_DIR / "figure7_ws_cross_scale_generalization_source_data.csv"
SOURCE_TABLE = SOURCE_DATA_PATH

METHOD_ORDER = ["EoH-Best-from-1000", "HDA", "CoreHD", "CI(l=3)"]
SCALE_ORDER = [
    "30-50",
    "50-100",
    "100-200",
    "200-300",
    "300-400",
    "400-500",
    "2000",
    "3000",
    "4000",
    "5000",
]
METHOD_STYLE = {
    "EoH-Best-from-1000": {"color": "#C55A6C", "marker": "o", "linewidth": 2.0, "markersize": 5.0, "zorder": 5},
    "HDA": {"color": "#2F5D50", "marker": "^", "linewidth": 1.4, "markersize": 4.0, "zorder": 4},
    "CoreHD": {"color": "#4E79A7", "marker": "s", "linewidth": 1.4, "markersize": 4.0, "zorder": 3},
    "CI(l=3)": {"color": "#8F8F8F", "marker": "D", "linewidth": 1.4, "markersize": 3.8, "zorder": 2},
}
DISPLAY_LABELS = {
    "EoH-Best-from-1000": "Evolved best",
    "HDA": "HDA",
    "CoreHD": "CoreHD",
    "CI(l=3)": "CI(l=3)",
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


def load_ws_scale_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with SOURCE_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("family") != "ws":
                continue
            scale = str(row.get("scale", "")).strip()
            algorithm = str(row.get("algorithm", "")).strip()
            if scale not in SCALE_ORDER or algorithm not in METHOD_ORDER:
                continue
            rows.append(
                {
                    "scale": scale,
                    "method": algorithm,
                    "anc": float(row.get("objective") or row["anc"]),
                    "n_est": float(row["n_est"]),
                    "n_instances": int(float(row["n_instances"])),
                }
            )
    if not rows:
        raise ValueError("No WS scale-comparison rows found in scale_comparison_long.csv.")
    return rows


def build_plot_table(rows: list[dict[str, object]]) -> dict[str, list[float]]:
    table: dict[str, list[float]] = {}
    for method in METHOD_ORDER:
        values: list[float] = []
        for scale in SCALE_ORDER:
            match = next(
                (
                    row
                    for row in rows
                    if str(row["method"]) == method and str(row["scale"]) == scale
                ),
                None,
            )
            if match is None:
                raise ValueError(f"Missing WS result for method={method}, scale={scale}.")
            values.append(float(match["anc"]))
        table[method] = values
    return table


def write_source_data(rows: list[dict[str, object]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["scale_index", "scale", "method", "anc", "n_est", "n_instances"],
        )
        writer.writeheader()
        for scale_index, scale in enumerate(SCALE_ORDER):
            scale_rows = sorted(
                (row for row in rows if str(row["scale"]) == scale),
                key=lambda item: METHOD_ORDER.index(str(item["method"])),
            )
            for row in scale_rows:
                writer.writerow(
                    {
                        "scale_index": scale_index,
                        "scale": row["scale"],
                        "method": row["method"],
                        "anc": f"{float(row['anc']):.12f}",
                        "n_est": f"{float(row['n_est']):.6f}",
                        "n_instances": int(row["n_instances"]),
                    }
                )


def spread_targets(values: list[float], lower: float, upper: float, min_gap: float) -> list[float]:
    if not values:
        return []
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    placed: list[tuple[int, float]] = []
    cursor = lower
    for index, value in ranked:
        target = max(value, cursor)
        placed.append((index, target))
        cursor = target + min_gap
    overflow = placed[-1][1] - upper
    if overflow > 0:
        placed = [(index, y_value - overflow) for index, y_value in placed]
    output = [0.0] * len(values)
    for index, y_value in placed:
        output[index] = min(max(y_value, lower), upper)
    return output


def add_direct_labels(ax: plt.Axes, series: dict[str, list[float]], x_values: np.ndarray) -> None:
    label_x = x_values[-1] + 0.38
    raw_targets = [series[method][-1] for method in METHOD_ORDER]
    y_min, y_max = ax.get_ylim()
    padded_targets = spread_targets(raw_targets, y_min + 0.008, y_max - 0.008, min_gap=0.0085)
    for method, y_target, y_anchor in zip(METHOD_ORDER, padded_targets, raw_targets):
        style = METHOD_STYLE[method]
        ax.plot(
            [x_values[-1], label_x - 0.03],
            [y_anchor, y_target],
            color=style["color"],
            linewidth=0.8,
            alpha=0.9,
            clip_on=False,
            zorder=style["zorder"],
        )
        ax.text(
            label_x,
            y_target,
            DISPLAY_LABELS[method],
            color=style["color"],
            fontsize=6.5,
            fontweight="bold" if method == "EoH-Best-from-1000" else "normal",
            ha="left",
            va="center",
            clip_on=False,
        )


def plot_figure(series: dict[str, list[float]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width_mm = 183
    height_mm = 82
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.08, right=0.84, bottom=0.23, top=0.80)

    x_values = np.arange(len(SCALE_ORDER), dtype=float)

    # Separate dense small-scale regime from large-scale transfer regime.
    ax.axvspan(-0.45, 5.45, color="#F7F7F7", zorder=0)
    ax.axvspan(5.55, 9.45, color="#FCFCFC", zorder=0)
    ax.axvline(5.5, color="#D9D9D9", linestyle=(0, (2, 2)), linewidth=0.9, zorder=1)

    y_values_all = [value for values in series.values() for value in values]
    y_min = min(y_values_all)
    y_max = max(y_values_all)
    y_pad = max((y_max - y_min) * 0.20, 0.012)

    for method in METHOD_ORDER:
        values = np.array(series[method], dtype=float)
        style = METHOD_STYLE[method]
        ax.plot(
            x_values,
            values,
            color=style["color"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white" if method != "EoH-Best-from-1000" else style["color"],
            markeredgewidth=1.0,
            markeredgecolor=style["color"],
            zorder=style["zorder"],
        )

    ax.set_xlim(-0.45, len(SCALE_ORDER) - 1 + 0.95)
    ax.set_ylim(y_min - y_pad * 0.35, y_max + y_pad * 0.65)
    ax.set_ylabel("ANC")
    ax.set_xlabel("WS scale")
    ax.set_xticks(x_values)
    ax.set_xticklabels(SCALE_ORDER, rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=6.2, pad=3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)

    ax.text(
        0.00,
        1.16,
        "Figure | WS cross-scale generalization curve",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
    ax.text(
        0.00,
        1.08,
        "Lower ANC is better. Evolved best is transferred from WS-1000 and evaluated on unseen scales.",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.4,
        color="#4D4D4D",
    )
    ax.text(
        0.21,
        1.01,
        "30-500",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#6A6A6A",
    )
    ax.text(
        0.72,
        1.01,
        "2000-5000",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#6A6A6A",
    )

    add_direct_labels(ax, series, x_values)
    save_pub_py(fig, OUT_BASENAME)
    plt.close(fig)


def main() -> None:
    rows = load_ws_scale_rows()
    write_source_data(rows)
    series = build_plot_table(rows)
    plot_figure(series)


if __name__ == "__main__":
    main()
