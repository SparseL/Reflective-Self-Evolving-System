from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
OUT_DIR = ROOT / "evolve_experiment" / "figures" / "exp5_real_evolution"
SOURCE_DATA_PATH = OUT_DIR / "figure5_source_data.csv"
OUT_BASENAME = OUT_DIR / "figure5_real_evolution_best_objective_2x2_colored"
X_AXIS_MAX = 50

DATASETS = [
    {"label": "Crime"},
    {"label": "Enron sampled"},
    {"label": "HI-II-14 sampled"},
    {"label": "Youtube sampled"},
]

LINE_COLOR = "#34495E"
JUMP_COLOR = "#D64B4B"
FINAL_COLOR = "#2C3E50"


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


def extract_generation(path: Path) -> int | None:
    match = re.search(r"population_generation_(-?\d+)\.json$", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_best_curve(dataset_label: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with SOURCE_DATA_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row or row.get("dataset") != dataset_label:
                continue
            rows.append({"generation": int(row["generation"]), "best_anc": float(row["best_anc"])})
    if not rows:
        raise ValueError(f"No generation rows found for {dataset_label} in {SOURCE_DATA_PATH}")
    return rows


def write_source_data(curves: list[dict[str, object]]) -> None:
    fieldnames = ["topology", "generation", "best_anc"]
    with SOURCE_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            dataset = str(curve["label"])
            for row in curve["rows"]:
                writer.writerow(
                    {
                        "topology": dataset,
                        "generation": int(row["generation"]),
                        "best_anc": float(row["best_anc"]),
                    }
                )


def compute_jump_mask(values: np.ndarray) -> np.ndarray:
    previous = np.r_[np.nan, values[:-1]]
    jump_mask = np.isnan(previous) | (values < previous - 1e-12)
    return jump_mask


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.03,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        color="black",
    )


def plot_curve(ax: plt.Axes, label: str, rows: list[dict[str, float]], panel_label: str) -> None:
    generations = np.array([int(row["generation"]) for row in rows], dtype=int)
    values = np.array([float(row["best_anc"]) for row in rows], dtype=float)
    jump_mask = compute_jump_mask(values)
    plot_generations = generations.copy()
    plot_values = values.copy()
    if plot_generations[-1] < X_AXIS_MAX:
        plot_generations = np.append(plot_generations, X_AXIS_MAX)
        plot_values = np.append(plot_values, plot_values[-1])

    ax.step(plot_generations, plot_values, where="post", color=LINE_COLOR, linewidth=1.4, zorder=2)
    ax.scatter(
        generations[jump_mask],
        values[jump_mask],
        s=28,
        facecolor=JUMP_COLOR,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.scatter(
        generations[-1],
        values[-1],
        s=52,
        marker="D",
        facecolor="white",
        edgecolor=FINAL_COLOR,
        linewidth=1.0,
        zorder=4,
    )

    y_min = float(values.min())
    y_max = float(values.max())
    y_span = y_max - y_min
    y_margin = max(y_span * 0.10, y_max * 0.02, 0.002)

    ax.set_xlim(1, X_AXIS_MAX)
    ax.set_xticks([1, 10, 20, 30, 40, 50])
    ax.set_ylim(max(0.0, y_min - y_margin), y_max + y_margin)
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best ANC")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.text(
        0.98,
        0.06,
        f"Final = {values[-1]:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6,
        color=FINAL_COLOR,
    )
    add_panel_label(ax, panel_label)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curves: list[dict[str, object]] = []
    for dataset in DATASETS:
        curves.append(
            {
                "label": dataset["label"],
                "rows": load_best_curve(str(dataset["label"])),
            }
        )

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 142 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.92, wspace=0.26, hspace=0.34)

    for ax, curve, panel_label in zip(axes.flatten(), curves, ["a", "b", "c", "d"]):
        plot_curve(ax, str(curve["label"]), list(curve["rows"]), panel_label)

    save_pub_py(fig, OUT_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
