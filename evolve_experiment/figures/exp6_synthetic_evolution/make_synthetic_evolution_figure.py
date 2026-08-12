from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
EVOLVE_ROOT = PACKAGE_ROOT / "evolve_experiment"
TABLE_DIR = EVOLVE_ROOT / "processed" / "analysis_tables_synthetic_runs" / "popbest"
OUT_DIR = EVOLVE_ROOT / "figures" / "exp6_synthetic_evolution"
SOURCE_DATA_PATH = OUT_DIR / "figure6_source_data.csv"
OUT_BASENAME = OUT_DIR / "figure6_synthetic_evolution_best_anc_over_generation"

DATASETS = [
    {
        "label": "ER",
        "source_csv": TABLE_DIR
        / "group1_sbm_er__gpt-4.1-mini__synthetic_er_1000_20260330_133530__synthetic_er_1000_pops_best.csv",
    },
    {
        "label": "SBM",
        "source_csv": TABLE_DIR
        / "group1_sbm_er__gpt-4.1-mini__synthetic_sbm_1000_20260330_100245__synthetic_sbm_1000_pops_best.csv",
    },
    {
        "label": "Uniform-cost",
        "source_csv": TABLE_DIR
        / "group2_ws_uniform__gpt-4.1-mini__synthetic_uniform_cost_1000_20260330_112744__synthetic_uniform_cost_1000_pops_best.csv",
    },
    {
        "label": "WS",
        "source_csv": TABLE_DIR
        / "group2_ws_uniform__gpt-4.1-mini__synthetic_ws_1000_20260330_100257__synthetic_ws_1000_pops_best.csv",
    },
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


def load_curve(source_csv: Path) -> list[dict[str, float]]:
    df = pd.read_csv(source_csv)
    df = df.loc[:, ["generation", "objective"]].dropna().copy()
    df["generation"] = df["generation"].astype(int)
    df["best_anc"] = df["objective"].astype(float)
    df = df[df["generation"] > 0].sort_values("generation").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No generation rows found in {source_csv}")
    return df.loc[:, ["generation", "best_anc"]].to_dict("records")


def write_source_data(curves: list[dict[str, object]]) -> None:
    fieldnames = ["topology", "generation", "best_anc"]
    with SOURCE_DATA_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            topology = str(curve["label"])
            for row in curve["rows"]:
                writer.writerow(
                    {
                        "topology": topology,
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

    ax.step(generations, values, where="post", color=LINE_COLOR, linewidth=1.4, zorder=2)
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

    ax.set_xlim(max(0, int(generations.min()) - 1), int(generations.max()) + 1)
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
                "rows": load_curve(dataset["source_csv"]),
            }
        )

    write_source_data(curves)

    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 142 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.11, top=0.92, wspace=0.26, hspace=0.34)

    for ax, curve, panel_label in zip(axes.flatten(), curves, ["a", "b", "c", "d"]):
        plot_curve(ax, str(curve["label"]), list(curve["rows"]), panel_label)

    save_pub_py(fig, OUT_BASENAME)
    plt.close(fig)


if __name__ == "__main__":
    main()
