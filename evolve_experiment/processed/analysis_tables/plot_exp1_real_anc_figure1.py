from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
OUT_DIR = PACKAGE_ROOT / "evolve_experiment" / "figures" / "exp1_real_anc"
SOURCE_DATA_PATH = OUT_DIR / "figure1_real_network_anc_source_data_v5.csv"
TABLE_PATH = SOURCE_DATA_PATH
OUT_BASENAME = OUT_DIR / "figure1_real_network_anc_comparison_v5"

METHODS = ["CI(l=3)", "CoreHD", "HDA", "Evolved best"]
FAMILY_SPECS = [
    {"label": "Crime", "variants": ["Crime"]},
    {"label": "Enron", "variants": ["Enron_sample", "Enron"]},
    {"label": "HI-II-14", "variants": ["HI-II-14_sampled", "HI-II-14"]},
    {"label": "Youtube", "variants": ["Youtube_sampled", "Youtube_sample4000", "Youtube_sample8000"]},
]
DATASET_ORDER = [dataset for family in FAMILY_SPECS for dataset in family["variants"]]
DISPLAY_NAMES = {
    "Crime": "Crime",
    "Enron": "Enron",
    "Enron_sample": "Enron\nsample",
    "HI-II-14": "HI-II-14",
    "HI-II-14_sampled": "HI-II-14\nsampled",
    "Youtube_sample4000": "Youtube\n4000",
    "Youtube_sample8000": "Youtube\n8000",
    "Youtube_sampled": "Youtube\nsampled",
}

# Distinct but restrained Nature-style palette: cool baselines, warm accent for Evolved best.
COLORS = {
    "CI(l=3)": "#8F8F8F",
    "CoreHD": "#4E79A7",
    "HDA": "#2F5D50",
    "Evolved best": "#C55A6C",
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


def clean_cell(text: str) -> str:
    return text.replace("**", "").strip()


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{int(max(0, min(1, c)) * 255):02X}" for c in rgb)


def lighten(hex_color: str, amount: float) -> str:
    base = np.array(hex_to_rgb01(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    return rgb01_to_hex(tuple(base + (white - base) * amount))


def variant_fill_color(method: str, dataset: str) -> str:
    if dataset.endswith("_sample") or dataset.endswith("_sampled"):
        return lighten(COLORS[method], 0.58)
    if "4000" in dataset:
        return lighten(COLORS[method], 0.30)
    if "8000" in dataset:
        return COLORS[method]
    return COLORS[method]


def variant_rank(dataset: str) -> int:
    if dataset.endswith("_sample") or dataset.endswith("_sampled"):
        return 0
    if "4000" in dataset:
        return 1
    return 2


def add_scale_gradient(fig: plt.Figure) -> None:
    gradient_ax = fig.add_axes([0.76, 0.855, 0.17, 0.028])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "scale_grad", ["#D9D9D9", "#6A6A6A"]
    )
    gradient_ax.imshow(grad, aspect="auto", cmap=cmap)
    gradient_ax.set_xticks([])
    gradient_ax.set_yticks([])
    for spine in gradient_ax.spines.values():
        spine.set_visible(False)

    gradient_ax.text(
        0.0,
        1.30,
        "sampled",
        transform=gradient_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    gradient_ax.text(
        1.0,
        1.30,
        "original",
        transform=gradient_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#4D4D4D",
    )
    gradient_ax.annotate(
        "",
        xy=(1.0, -0.55),
        xytext=(0.0, -0.55),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#5A5A5A"),
    )
    gradient_ax.text(
        0.5,
        -1.18,
        "Darker color indicates larger scale",
        transform=gradient_ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.0,
        color="#5A5A5A",
    )


def parse_real_table(table_path: Path) -> dict[str, dict[str, float | None]]:
    data: dict[str, dict[str, float | None]] = {}
    with table_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            data[dataset] = {
                method: None if row.get(method, "") in {"", "-"} else float(row[method])
                for method in METHODS
            }
    return data


def write_source_data(data: dict[str, dict[str, float | None]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE_DATA_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", *METHODS])
        for dataset in DATASET_ORDER:
            writer.writerow([dataset, *[data[dataset].get(method) for method in METHODS]])


def plot_grouped_bars(data: dict[str, dict[str, float | None]]) -> None:
    apply_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width_mm = 183
    height_mm = 92
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.14, top=0.78)

    family_gap = 1.36
    method_gap = 0.54
    overlay_offsets = {
        1: [0.0],
        2: [-0.022, 0.022],
        3: [-0.070, 0.0, 0.070],
    }
    overlay_widths = {
        1: 0.14,
        2: 0.125,
        3: 0.095,
    }
    method_offsets = np.linspace(-0.55, 0.55, len(METHODS))

    family_centers: list[float] = []
    ymax = 0.0

    for family_idx, family in enumerate(FAMILY_SPECS):
        family_center = family_idx * family_gap
        family_centers.append(family_center)
        variants = family["variants"]
        for method_offset, method in zip(method_offsets, METHODS):
            method_idx = METHODS.index(method)
            slot_center = family_center + method_offset * method_gap
            values = [data[dataset].get(method) for dataset in variants]
            numeric_values = [v for v in values if v is not None]
            if numeric_values:
                ymax = max(ymax, max(numeric_values))

            ordered_variants = sorted(variants, key=variant_rank)
            current_offsets = overlay_offsets[len(variants)]
            for draw_idx, (variant_offset, dataset) in enumerate(zip(current_offsets, ordered_variants)):
                value = data[dataset].get(method)
                if value is None:
                    continue
                xpos = slot_center + variant_offset
                base_zorder = 3
                if family["label"] == "Youtube":
                    # For Youtube only: later method groups sit above earlier groups
                    # so gray < blue < green < red across the small overlaps.
                    base_zorder = 3 + method_idx * 10
                ax.bar(
                    xpos,
                    value,
                    width=overlay_widths[len(variants)],
                    color=variant_fill_color(method, dataset),
                    edgecolor="#404040",
                    linewidth=0.4,
                    zorder=base_zorder + draw_idx,
                    label=method if family_idx == 0 and dataset == variants[0] else None,
                )

    # Label Evolved best only, using the same overlap positions to keep the panel readable.
    for family_idx, family in enumerate(FAMILY_SPECS):
        family_center = family_centers[family_idx]
        variants = family["variants"]
        eoh_center = family_center + method_offsets[-1] * method_gap
        ordered_variants = sorted(variants, key=variant_rank)
        current_offsets = overlay_offsets[len(variants)]
        for variant_offset, dataset in zip(current_offsets, ordered_variants):
            value = data[dataset]["Evolved best"]
            if value is None:
                continue
            y_text = value + 0.006
            x_text = eoh_center + variant_offset
            va = "bottom"
            bbox = None
            if family["label"] == "Youtube":
                # Place Youtube labels just inside the pink bars so they remain
                # readable but do not sit on the top edge line.
                y_text = value - 0.011
                x_text = eoh_center + variant_offset
                va = "top"
                bbox = {
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                }
            ax.text(
                x_text,
                y_text,
                f"{value:.3f}",
                ha="center",
                va=va,
                fontsize=5.7,
                color="#A94A5A" if family["label"] == "Youtube" else variant_fill_color("Evolved best", dataset),
                fontweight="bold",
                bbox=bbox,
                zorder=100,
            )

    ax.set_ylabel("ANC")
    ax.set_xticks(family_centers)
    ax.set_xticklabels([family["label"] for family in FAMILY_SPECS], rotation=0, ha="center")
    ax.set_ylim(0.0, max(0.29, ymax + 0.035))
    ax.set_yticks(np.arange(0.0, 0.301, 0.05))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.margins(x=0.04)

    # Add gentle separators between network families.
    for left_center, right_center in zip(family_centers[:-1], family_centers[1:]):
        ax.axvline((left_center + right_center) / 2, color="#E6E6E6", linewidth=0.8, zorder=1)

    method_legend = ax.legend(
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.985),
        columnspacing=1.2,
        handlelength=1.2,
    )
    for text, method in zip(method_legend.get_texts(), METHODS):
        if method == "Evolved best":
            text.set_fontweight("bold")
    ax.add_artist(method_legend)

    fig.text(
        0.08,
        0.965,
        "Figure 1 | ANC comparison on real networks",
        ha="left",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.922,
        "Lower ANC is better; method order in each family is CI(l=3), CoreHD, HDA, Evolved best.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4D4D4D",
    )
    fig.text(
        0.08,
        0.885,
        "Dark bars are drawn on top of light bars within each method slot.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4D4D4D",
    )
    add_scale_gradient(fig)

    fig.savefig(f"{OUT_BASENAME}.svg", bbox_inches="tight")
    fig.savefig(f"{OUT_BASENAME}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT_BASENAME}.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(f"{OUT_BASENAME}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = parse_real_table(TABLE_PATH)
    write_source_data(data)
    plot_grouped_bars(data)


if __name__ == "__main__":
    main()
