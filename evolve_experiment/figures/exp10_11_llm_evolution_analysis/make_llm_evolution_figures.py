from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PACKAGE_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
EVOLVE_ROOT = PACKAGE_ROOT / "evolve_experiment"
SUMMARY_CSV = (
    EVOLVE_ROOT
    / "processed"
    / "analysis_tables_llm_benchmarks"
    / "synthetic_ws_400-500"
    / "llm_bench_summary.csv"
)
TRAJ_CSV = (
    EVOLVE_ROOT
    / "processed"
    / "analysis_tables_llm_benchmarks"
    / "synthetic_ws_400-500"
    / "llm_bench_pops_best_by_generation.csv"
)
OUT_DIR = EVOLVE_ROOT / "figures" / "exp10_11_llm_evolution_analysis"

FIG10_BASENAME = OUT_DIR / "figure10_llm_evolution_dynamics"
FIG11_BASENAME = OUT_DIR / "figure11_llm_strategy_style_summary"
SUPP_BASENAME = OUT_DIR / "supp_llm_detailed_evolution_small_multiples"

FIG10_TRAJECTORY_SOURCE = OUT_DIR / "figure10_trajectory_source_data.csv"
FIG10_SUMMARY_SOURCE = OUT_DIR / "figure10_summary_source_data.csv"
FIG11_TAG_SOURCE = OUT_DIR / "figure11_tag_heatmap_source_data.csv"
FIG11_OPERATOR_SOURCE = OUT_DIR / "figure11_operator_share_source_data.csv"
SUPP_SOURCE = OUT_DIR / "supp_llm_detailed_evolution_source_data.csv"

MODEL_ORDER_RAW = [
    "claude-sonnet-4-6",
    "deepseek-r1",
    "gemini-2.5-flash",
    "gpt-5.1-codex-ca",
    "kimi-k2.5",
]
MODEL_DISPLAY = {
    "claude-sonnet-4-6": "Claude Sonnet",
    "deepseek-r1": "DeepSeek-R1",
    "gemini-2.5-flash": "Gemini",
    "gpt-5.1-codex-ca": "GPT-5.1-Codex",
    "kimi-k2.5": "Kimi",
}
MODEL_COLORS = {
    "claude-sonnet-4-6": "#C55A6C",
    "deepseek-r1": "#2F5D50",
    "gemini-2.5-flash": "#4E79A7",
    "gpt-5.1-codex-ca": "#8E6BAE",
    "kimi-k2.5": "#D48A32",
}
OPERATOR_ORDER = ["e1", "e2", "m1", "m2", "t2", "i1", "unknown"]
OPERATOR_DISPLAY = {
    "e1": "E1",
    "e2": "E2",
    "m1": "M1",
    "m2": "M2",
    "t2": "T2",
    "i1": "I1",
    "unknown": "Unknown",
}
OPERATOR_COLORS = {
    "e1": "#72B7B2",
    "e2": "#4C78A8",
    "m1": "#ECAE4E",
    "m2": "#E45756",
    "t2": "#B279A2",
    "i1": "#9D9D9D",
    "unknown": "#C7C7C7",
}
PHASES = {
    "Early": (1, 10),
    "Middle": (11, 20),
    "Late": (21, 30),
}
TAG_ORDER = [
    "Bridge / Boundary",
    "Centrality Mix",
    "Core / Degree",
    "Local Cohesion",
    "Spectral / Path",
    "Adaptive Weighting",
]
TAG_RULES = {
    "Bridge / Boundary": [
        r"bridge",
        r"bottleneck",
        r"conductance",
        r"resistance",
        r"boundary",
        r"cut",
        r"fragment",
        r"detour",
        r"flow",
        r"component",
        r"split",
        r"stress",
        r"high_bc",
    ],
    "Centrality Mix": [
        r"betweenness",
        r"eigenvector",
        r"centrality",
        r"harmonic mean",
        r"geometric mean",
        r"anchor",
        r"entropy",
    ],
    "Core / Degree": [
        r"core",
        r"coreness",
        r"degree",
        r"hub",
        r"high-core",
    ],
    "Local Cohesion": [
        r"clustering",
        r"neighbor",
        r"neighborhood",
        r"local",
        r"cohesion",
        r"redundancy",
        r"2-hop",
        r"union-find",
    ],
    "Spectral / Path": [
        r"laplacian",
        r"spectral",
        r"path",
        r"apl",
        r"diameter",
        r"decay",
        r"distance",
    ],
    "Adaptive Weighting": [
        r"dynamic",
        r"progress",
        r"phase",
        r"weight",
        r"target",
        r"residual",
        r"penalty",
        r"adaptive",
        r"modulat",
        r"calibrat",
    ],
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
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


def model_display(raw_name: str) -> str:
    return MODEL_DISPLAY.get(raw_name, raw_name)


def phase_for_generation(generation: int) -> str:
    for phase, (start, end) in PHASES.items():
        if start <= generation <= end:
            return phase
    return "Other"


def lighten(color: str, amount: float) -> str:
    base = np.array(mcolors.to_rgb(color))
    return mcolors.to_hex(base + (1.0 - base) * amount)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_summary_rows() -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = str(row["model"]).strip()
            if raw_name not in MODEL_ORDER_RAW:
                continue
            summary[raw_name] = {
                "dataset": str(row["dataset"]).strip(),
                "run_name": str(row["run_name"]).strip(),
                "results_root": Path(str(row["results_root"]).strip()),
                "best_generation": int(float(row["best_generation"])),
                "best_anc": float(row["objective_best_overall"]),
            }
    if len(summary) != len(MODEL_ORDER_RAW):
        missing = [name for name in MODEL_ORDER_RAW if name not in summary]
        raise ValueError(f"Missing summary rows for models: {missing}")
    return summary


def load_trajectory_rows() -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {name: [] for name in MODEL_ORDER_RAW}
    with TRAJ_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_name = str(row["model"]).strip()
            if raw_name not in grouped:
                continue
            grouped[raw_name].append(
                {
                    "generation": int(float(row["generation"])),
                    "objective": float(row["objective"]),
                    "operator": str(row["operator"]).strip().lower() or "unknown",
                    "file": str(row["file"]).strip(),
                    "time_select": float(row["time_select"]),
                    "time_anc": float(row["time_anc"]),
                }
            )
    for raw_name in MODEL_ORDER_RAW:
        grouped[raw_name].sort(key=lambda item: int(item["generation"]))
        if not grouped[raw_name]:
            raise ValueError(f"No trajectory rows found for {raw_name}")
    return grouped


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_text_snippet(text: str, limit: int = 170) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def classify_text(algorithm_text: str, code_text: str) -> dict[str, float]:
    algo_lowered = algorithm_text.lower()
    code_lowered = code_text.lower()
    scores: dict[str, float] = {}
    for tag, patterns in TAG_RULES.items():
        score = 0.0
        for pattern in patterns:
            if algo_lowered and re.search(pattern, algo_lowered):
                score += 2.0
            if code_lowered and re.search(pattern, code_lowered):
                score += 0.6 if algo_lowered else 1.0
        scores[tag] = score
    if sum(scores.values()) <= 0:
        scores["Centrality Mix"] = 1.0
    total = sum(scores.values())
    return {tag: scores[tag] / total for tag in TAG_ORDER}


def build_model_payloads() -> dict[str, dict[str, object]]:
    summary_rows = load_summary_rows()
    traj_rows = load_trajectory_rows()
    payloads: dict[str, dict[str, object]] = {}
    for raw_name in MODEL_ORDER_RAW:
        rows = traj_rows[raw_name]
        meta = summary_rows[raw_name]
        best_so_far = math.inf
        improvements: list[dict[str, object]] = []
        enriched_rows: list[dict[str, object]] = []
        for row in rows:
            objective = float(row["objective"])
            improved = objective < best_so_far - 1e-12
            delta = 0.0 if math.isinf(best_so_far) else max(0.0, best_so_far - objective)
            if improved:
                best_so_far = objective
                json_path = Path(meta["results_root"]) / "pops_best" / str(meta["dataset"]) / str(row["file"])
                json_payload = load_json(json_path)
                algo_text = str(json_payload.get("algorithm", "") or "").strip()
                code_text = str(json_payload.get("code", "") or "").strip()
                tag_weights = classify_text(algo_text, code_text)
                improvements.append(
                    {
                        "generation": int(row["generation"]),
                        "objective": objective,
                        "operator": str(row["operator"]),
                        "delta": delta,
                        "algorithm": algo_text,
                        "code": code_text,
                        "tag_weights": tag_weights,
                        "summary_text": extract_text_snippet(algo_text or code_text, limit=180),
                    }
                )
            enriched_rows.append(
                {
                    "generation": int(row["generation"]),
                    "objective": objective,
                    "best_so_far": best_so_far,
                    "operator": str(row["operator"]),
                    "is_improvement": improved,
                    "delta": delta,
                    "phase": phase_for_generation(int(row["generation"])),
                }
            )

        final_best = float(enriched_rows[-1]["best_so_far"])
        start_value = float(enriched_rows[0]["best_so_far"])
        threshold_95 = start_value - 0.95 * (start_value - final_best)
        first_95 = next(
            int(row["generation"])
            for row in enriched_rows
            if float(row["best_so_far"]) <= threshold_95 + 1e-12
        )
        first_best = next(
            int(row["generation"])
            for row in enriched_rows
            if abs(float(row["best_so_far"]) - final_best) <= 1e-12
        )
        plateau_generations = len(enriched_rows) - first_best + 1
        improvement_deltas = [float(item["delta"]) for item in improvements[1:]]
        improvement_count = len(improvements) - 1
        mean_improvement = float(np.mean(improvement_deltas)) if improvement_deltas else 0.0
        operator_counter = Counter(str(item["operator"]) for item in improvements)
        dominant_operator = operator_counter.most_common(1)[0][0] if operator_counter else "unknown"
        tag_means = {
            tag: float(np.mean([float(item["tag_weights"][tag]) for item in improvements]))
            for tag in TAG_ORDER
        }
        top_tags = sorted(TAG_ORDER, key=lambda tag: tag_means[tag], reverse=True)[:2]
        payloads[raw_name] = {
            "meta": meta,
            "rows": enriched_rows,
            "improvements": improvements,
            "summary": {
                "model": raw_name,
                "display_model": model_display(raw_name),
                "start_anc": start_value,
                "final_anc": final_best,
                "first_95_generation": first_95,
                "first_best_generation": first_best,
                "plateau_generations": plateau_generations,
                "plateau_ratio": plateau_generations / len(enriched_rows),
                "improvement_count": improvement_count,
                "mean_improvement": mean_improvement,
                "dominant_operator": dominant_operator,
                "top_tag_1": top_tags[0],
                "top_tag_2": top_tags[1],
            },
            "tag_means": tag_means,
        }
    return payloads


def build_source_tables(payloads: dict[str, dict[str, object]]) -> None:
    trajectory_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    tag_rows: list[dict[str, object]] = []
    operator_rows: list[dict[str, object]] = []
    supp_rows: list[dict[str, object]] = []

    for raw_name in MODEL_ORDER_RAW:
        payload = payloads[raw_name]
        summary = payload["summary"]
        for row in payload["rows"]:
            trajectory_rows.append(
                {
                    "model": model_display(raw_name),
                    "raw_model": raw_name,
                    "generation": int(row["generation"]),
                    "objective": f"{float(row['objective']):.12f}",
                    "best_so_far": f"{float(row['best_so_far']):.12f}",
                    "operator": row["operator"],
                    "phase": row["phase"],
                    "is_improvement": int(bool(row["is_improvement"])),
                    "delta_from_previous_best": f"{float(row['delta']):.12f}",
                }
            )
            supp_rows.append(
                {
                    "model": model_display(raw_name),
                    "generation": int(row["generation"]),
                    "best_so_far": f"{float(row['best_so_far']):.12f}",
                    "operator": row["operator"],
                    "is_improvement": int(bool(row["is_improvement"])),
                }
            )

        summary_rows.append(
            {
                "model": model_display(raw_name),
                "start_anc": f"{float(summary['start_anc']):.12f}",
                "final_anc": f"{float(summary['final_anc']):.12f}",
                "first_95_generation": int(summary["first_95_generation"]),
                "first_best_generation": int(summary["first_best_generation"]),
                "plateau_generations": int(summary["plateau_generations"]),
                "plateau_ratio": f"{float(summary['plateau_ratio']):.12f}",
                "improvement_count": int(summary["improvement_count"]),
                "mean_improvement": f"{float(summary['mean_improvement']):.12f}",
                "dominant_operator": summary["dominant_operator"],
                "top_tag_1": summary["top_tag_1"],
                "top_tag_2": summary["top_tag_2"],
            }
        )

        for tag in TAG_ORDER:
            tag_rows.append(
                {
                    "model": model_display(raw_name),
                    "tag": tag,
                    "mean_weight_over_improvements": f"{float(payload['tag_means'][tag]):.12f}",
                }
            )

        improvement_rows = payload["improvements"]
        phase_operator_counter: dict[tuple[str, str], int] = defaultdict(int)
        phase_total_counter: dict[str, int] = defaultdict(int)
        for item in improvement_rows:
            phase = phase_for_generation(int(item["generation"]))
            operator = str(item["operator"])
            phase_operator_counter[(phase, operator)] += 1
            phase_total_counter[phase] += 1
        for phase in PHASES:
            total = phase_total_counter.get(phase, 0)
            for operator in OPERATOR_ORDER:
                count = phase_operator_counter.get((phase, operator), 0)
                share = (count / total) if total else 0.0
                operator_rows.append(
                    {
                        "model": model_display(raw_name),
                        "phase": phase,
                        "operator": OPERATOR_DISPLAY.get(operator, operator.upper()),
                        "share_in_improvements": f"{share:.12f}",
                        "count": count,
                    }
                )

    write_csv(
        FIG10_TRAJECTORY_SOURCE,
        [
            "model",
            "raw_model",
            "generation",
            "objective",
            "best_so_far",
            "operator",
            "phase",
            "is_improvement",
            "delta_from_previous_best",
        ],
        trajectory_rows,
    )
    write_csv(
        FIG10_SUMMARY_SOURCE,
        [
            "model",
            "start_anc",
            "final_anc",
            "first_95_generation",
            "first_best_generation",
            "plateau_generations",
            "plateau_ratio",
            "improvement_count",
            "mean_improvement",
            "dominant_operator",
            "top_tag_1",
            "top_tag_2",
        ],
        summary_rows,
    )
    write_csv(
        FIG11_TAG_SOURCE,
        ["model", "tag", "mean_weight_over_improvements"],
        tag_rows,
    )
    write_csv(
        FIG11_OPERATOR_SOURCE,
        ["model", "phase", "operator", "share_in_improvements", "count"],
        operator_rows,
    )
    write_csv(
        SUPP_SOURCE,
        ["model", "generation", "best_so_far", "operator", "is_improvement"],
        supp_rows,
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.02,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color="black",
    )


def spread_targets(values: list[float], lower: float, upper: float, gap: float) -> list[float]:
    if not values:
        return []
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    placed: list[tuple[int, float]] = []
    cursor = lower
    for index, value in indexed:
        target = max(value, cursor)
        placed.append((index, target))
        cursor = target + gap
    overflow = placed[-1][1] - upper
    if overflow > 0:
        placed = [(index, y_value - overflow) for index, y_value in placed]
    output = [0.0] * len(values)
    for index, y_value in placed:
        output[index] = float(min(max(y_value, lower), upper))
    return output


def plot_figure10(payloads: dict[str, dict[str, object]]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 120 / 25.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1.0], hspace=0.52, wspace=0.46)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])
    fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.16)

    # a) Hero panel: best-so-far ANC trajectories
    all_y: list[float] = []
    final_y: list[float] = []
    final_x: list[float] = []
    for raw_name in MODEL_ORDER_RAW:
        rows = payloads[raw_name]["rows"]
        x_vals = np.array([int(row["generation"]) for row in rows], dtype=float)
        y_vals = np.array([float(row["best_so_far"]) for row in rows], dtype=float)
        all_y.extend(y_vals.tolist())
        final_y.append(float(y_vals[-1]))
        final_x.append(float(x_vals[-1]))
        ax_a.step(
            x_vals,
            y_vals,
            where="post",
            color=MODEL_COLORS[raw_name],
            linewidth=2.0 if raw_name == "claude-sonnet-4-6" else 1.45,
            zorder=4 if raw_name == "claude-sonnet-4-6" else 3,
        )
        improvement_rows = [row for row in rows if bool(row["is_improvement"])]
        ax_a.scatter(
            [int(row["generation"]) for row in improvement_rows],
            [float(row["best_so_far"]) for row in improvement_rows],
            s=24 if raw_name == "claude-sonnet-4-6" else 18,
            facecolor="white" if raw_name != "claude-sonnet-4-6" else MODEL_COLORS[raw_name],
            edgecolor=MODEL_COLORS[raw_name],
            linewidth=0.9,
            zorder=5,
        )

    y_min = min(all_y)
    y_max = max(all_y)
    ax_a.set_xlim(1, 33.4)
    ax_a.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax_a.set_ylim(y_min - 0.007, y_max + 0.017)
    ax_a.set_xlabel("Generation")
    ax_a.set_ylabel("Best ANC")
    ax_a.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax_a.axvspan(1, 10, color="#FAFAFA", zorder=0)
    ax_a.axvspan(10, 20, color="#F5F5F5", zorder=0)
    ax_a.axvspan(20, 30, color="#FAFAFA", zorder=0)
    ax_a.text(0.17, 1.02, "Early", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=6.2, color="#6A6A6A")
    ax_a.text(0.50, 1.02, "Middle", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=6.2, color="#6A6A6A")
    ax_a.text(0.83, 1.02, "Late", transform=ax_a.transAxes, ha="center", va="bottom", fontsize=6.2, color="#6A6A6A")
    direct_y = spread_targets(final_y, y_min - 0.001, y_max + 0.014, gap=0.0054)
    label_x = 31.7
    for raw_name, x_anchor, y_anchor, y_target in zip(MODEL_ORDER_RAW, final_x, final_y, direct_y):
        color = MODEL_COLORS[raw_name]
        ax_a.plot([x_anchor, label_x - 0.18], [y_anchor, y_target], color=color, linewidth=0.8, clip_on=False)
        ax_a.text(
            label_x,
            y_target,
            model_display(raw_name),
            color=color,
            ha="left",
            va="center",
            fontsize=6.4,
            fontweight="bold" if raw_name == "claude-sonnet-4-6" else "normal",
            clip_on=False,
        )
    add_panel_label(ax_a, "a")

    # b) Milestone generations
    y_positions = np.arange(len(MODEL_ORDER_RAW), dtype=float)
    for y_pos, raw_name in zip(y_positions, MODEL_ORDER_RAW):
        summary = payloads[raw_name]["summary"]
        gen95 = int(summary["first_95_generation"])
        genbest = int(summary["first_best_generation"])
        color = MODEL_COLORS[raw_name]
        ax_b.hlines(y_pos, gen95, genbest, color=color, linewidth=2.2, alpha=0.85)
        if gen95 == genbest:
            ax_b.scatter(genbest, y_pos, s=48, facecolor="none", edgecolor=color, linewidth=1.2, zorder=4)
            ax_b.scatter(genbest, y_pos, s=22, facecolor=color, edgecolor=color, linewidth=1.0, zorder=5)
        else:
            ax_b.scatter(gen95, y_pos, s=28, facecolor="white", edgecolor=color, linewidth=1.0, zorder=4)
            ax_b.scatter(genbest, y_pos, s=32, facecolor=color, edgecolor=color, linewidth=1.0, zorder=5)
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([model_display(name) for name in MODEL_ORDER_RAW])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.5, 30.5)
    ax_b.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax_b.set_xlabel("Generation")
    ax_b.set_title("Convergence Milestones", pad=6)
    ax_b.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax_b.text(
        0.98,
        1.15,
        "open: 95% of final\nfilled: final best\nring+dot: same generation",
        transform=ax_b.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#666666",
    )
    ax_b.tick_params(axis="y", labelsize=6.2, pad=3)
    add_panel_label(ax_b, "b")

    # c) Improvement signature
    x_sig = [int(payloads[name]["summary"]["improvement_count"]) for name in MODEL_ORDER_RAW]
    y_sig = [float(payloads[name]["summary"]["mean_improvement"]) * 1000.0 for name in MODEL_ORDER_RAW]
    label_y_sig = spread_targets(y_sig, 0.45, max(y_sig) * 1.10, gap=0.78)
    for raw_name, x_value, y_value, label_y in zip(MODEL_ORDER_RAW, x_sig, y_sig, label_y_sig):
        ax_c.scatter(
            x_value,
            y_value,
            s=52 if raw_name == "claude-sonnet-4-6" else 40,
            facecolor=MODEL_COLORS[raw_name] if raw_name == "claude-sonnet-4-6" else "white",
            edgecolor=MODEL_COLORS[raw_name],
            linewidth=1.1,
            zorder=3,
        )
        label_x = x_value + 0.22
        if abs(label_y - y_value) > 0.08:
            ax_c.plot(
                [x_value + 0.06, label_x - 0.04],
                [y_value, label_y],
                color=MODEL_COLORS[raw_name],
                linewidth=0.65,
                alpha=0.9,
                clip_on=False,
                zorder=2,
            )
        ax_c.text(
            label_x,
            label_y,
            model_display(raw_name),
            fontsize=6.0,
            color=MODEL_COLORS[raw_name],
            va="center",
            ha="left",
            clip_on=False,
        )
    ax_c.set_xlabel("Improvement Steps")
    ax_c.set_ylabel("Mean ANC Drop per Gain (x1e-3)")
    ax_c.set_title("Jump Signature", pad=6)
    ax_c.grid(color="#E0E0E0", linewidth=0.7)
    ax_c.set_xlim(0.3, max(x_sig) + 2.7)
    ax_c.set_ylim(0.0, max(y_sig) * 1.22)
    add_panel_label(ax_c, "c")

    # d) Plateau stability
    plateau_vals = [float(payloads[name]["summary"]["plateau_ratio"]) * 100.0 for name in MODEL_ORDER_RAW]
    bars = ax_d.bar(
        np.arange(len(MODEL_ORDER_RAW)),
        plateau_vals,
        color=[lighten(MODEL_COLORS[name], 0.32 if name != "claude-sonnet-4-6" else 0.12) for name in MODEL_ORDER_RAW],
        edgecolor=[MODEL_COLORS[name] for name in MODEL_ORDER_RAW],
        linewidth=0.8,
        width=0.62,
    )
    for bar, raw_name, value in zip(bars, MODEL_ORDER_RAW, plateau_vals):
        ax_d.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.4,
            f"{value:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6.0,
            color=MODEL_COLORS[raw_name],
            fontweight="bold" if raw_name == "claude-sonnet-4-6" else "normal",
        )
    ax_d.set_xticks(np.arange(len(MODEL_ORDER_RAW)))
    ax_d.set_xticklabels(
        ["Claude", "DeepSeek", "Gemini", "GPT-5.1", "Kimi"],
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )
    ax_d.tick_params(axis="x", labelsize=5.8, pad=2)
    ax_d.set_ylabel("Plateau Time (%)")
    ax_d.set_title("Late-Stage Stability", pad=6)
    ax_d.grid(axis="y", color="#E0E0E0", linewidth=0.7)
    ax_d.set_ylim(0, max(plateau_vals) + 10)
    add_panel_label(ax_d, "d")

    fig.text(0.08, 0.975, "Figure 10 | LLM evolution dynamics on WS-400-500", ha="left", va="top", fontsize=8.4, fontweight="bold")
    fig.text(
        0.08,
        0.948,
        "Claude Sonnet reaches a substantially lower ANC, while the remaining LLMs converge to similar but weaker plateaus with distinct search rhythms.",
        ha="left",
        va="top",
        fontsize=6.5,
        color="#4D4D4D",
    )
    save_pub_py(fig, FIG10_BASENAME)
    plt.close(fig)


def plot_figure11(payloads: dict[str, dict[str, object]]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.20, 1.00], height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.34)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    fig.subplots_adjust(left=0.12, right=0.98, top=0.89, bottom=0.10)

    # a) Tag heatmap
    heat = np.array([[float(payloads[name]["tag_means"][tag]) for tag in TAG_ORDER] for name in MODEL_ORDER_RAW], dtype=float)
    im = ax_a.imshow(heat, cmap="Blues", aspect="auto", vmin=0.0, vmax=max(0.34, float(heat.max()) * 1.02))
    ax_a.set_xticks(np.arange(len(TAG_ORDER)))
    ax_a.set_xticklabels(
        ["Bridge\nBoundary", "Centrality\nMix", "Core\nDegree", "Local\nCohesion", "Spectral\nPath", "Adaptive\nWeighting"],
        rotation=0,
    )
    ax_a.set_yticks(np.arange(len(MODEL_ORDER_RAW)))
    ax_a.set_yticklabels([model_display(name) for name in MODEL_ORDER_RAW])
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat[i, j]
            ax_a.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if value > 0.17 else "#14344A",
            )
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.028, pad=0.02)
    cbar.ax.tick_params(labelsize=5.8)
    cbar.set_label("Mean weight over improvement heuristics", fontsize=6.0)
    ax_a.set_title("Heuristic Style Tags", pad=8)
    add_panel_label(ax_a, "a")

    # b) Operator share on improvement steps, aggregated over phases
    y_positions = np.arange(len(MODEL_ORDER_RAW), dtype=float)
    left = np.zeros(len(MODEL_ORDER_RAW), dtype=float)
    for operator in OPERATOR_ORDER:
        shares = []
        for raw_name in MODEL_ORDER_RAW:
            improvements = payloads[raw_name]["improvements"]
            total = len(improvements)
            count = sum(1 for item in improvements if str(item["operator"]) == operator)
            shares.append(count / total if total else 0.0)
        shares_arr = np.array(shares, dtype=float)
        if not np.any(shares_arr > 0):
            continue
        ax_b.barh(
            y_positions,
            shares_arr,
            left=left,
            color=OPERATOR_COLORS.get(operator, "#CCCCCC"),
            edgecolor="white",
            linewidth=0.6,
            height=0.58,
            label=OPERATOR_DISPLAY.get(operator, operator.upper()),
        )
        left += shares_arr
    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([model_display(name) for name in MODEL_ORDER_RAW])
    ax_b.invert_yaxis()
    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_xticks(np.linspace(0.0, 1.0, 6))
    ax_b.set_xlabel("Share of Improvement Events")
    ax_b.set_title("Operator Usage on Gains", pad=8)
    ax_b.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax_b.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.02, 1.05), fontsize=5.7, columnspacing=0.8, handlelength=0.9)
    add_panel_label(ax_b, "b")

    # c) Representative summaries
    ax_c.axis("off")
    ax_c.set_title("Representative Evolution Tendencies", pad=10)
    y_cursor = 0.95
    line_gap = 0.175
    for raw_name in MODEL_ORDER_RAW:
        summary = payloads[raw_name]["summary"]
        best_gen = int(summary["first_best_generation"])
        dominant_operator = OPERATOR_DISPLAY.get(str(summary["dominant_operator"]), str(summary["dominant_operator"]).upper())
        top1 = str(summary["top_tag_1"])
        top2 = str(summary["top_tag_2"])
        text = (
            f"{model_display(raw_name)}: {top1} + {top2}; "
            f"best reached at G{best_gen}; dominant gain operator {dominant_operator}."
        )
        ax_c.text(
            0.0,
            y_cursor,
            text,
            transform=ax_c.transAxes,
            ha="left",
            va="top",
            fontsize=6.2,
            color=MODEL_COLORS[raw_name],
            fontweight="bold" if raw_name == "claude-sonnet-4-6" else "normal",
            wrap=True,
        )
        y_cursor -= line_gap
    ax_c.text(
        0.0,
        0.02,
        "Tags are inferred from the best heuristic text/code at generations that actually improved the best-so-far ANC.",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color="#666666",
        wrap=True,
    )
    add_panel_label(ax_c, "c")

    fig.text(0.12, 0.972, "Figure 11 | Strategy style and heuristic thought across LLMs", ha="left", va="top", fontsize=8.4, fontweight="bold")
    # fig.text(
    #     0.12,
    #     0.945,
    #     "Different LLMs do not only differ in final ANC; they also emphasize different structural motifs and operator pathways during heuristic evolution.",
    #     ha="left",
    #     va="top",
    #     fontsize=6.5,
    #     color="#4D4D4D",
    # )
    save_pub_py(fig, FIG11_BASENAME)
    plt.close(fig)


def plot_supplementary(payloads: dict[str, dict[str, object]]) -> None:
    fig, axes = plt.subplots(len(MODEL_ORDER_RAW), 1, figsize=(183 / 25.4, 170 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.94, top=0.95, bottom=0.08, hspace=0.34)

    all_y = [float(row["best_so_far"]) for name in MODEL_ORDER_RAW for row in payloads[name]["rows"]]
    y_min = min(all_y) - 0.005
    y_max = max(all_y) + 0.007

    for ax, raw_name, panel in zip(axes, MODEL_ORDER_RAW, ["a", "b", "c", "d", "e"]):
        rows = payloads[raw_name]["rows"]
        color = MODEL_COLORS[raw_name]
        x_vals = np.array([int(row["generation"]) for row in rows], dtype=float)
        y_vals = np.array([float(row["best_so_far"]) for row in rows], dtype=float)
        improvements = [row for row in rows if bool(row["is_improvement"])]
        ax.step(x_vals, y_vals, where="post", color=color, linewidth=1.6)
        for op in OPERATOR_ORDER:
            op_rows = [row for row in improvements if str(row["operator"]) == op]
            if not op_rows:
                continue
            ax.scatter(
                [int(row["generation"]) for row in op_rows],
                [float(row["best_so_far"]) for row in op_rows],
                s=20,
                facecolor=OPERATOR_COLORS.get(op, "#CCCCCC"),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(1, 30.8)
        ax.grid(axis="y", color="#E3E3E3", linewidth=0.65)
        ax.set_ylabel("Best ANC")
        ax.set_title(model_display(raw_name), loc="left", fontsize=7.3, fontweight="bold", color=color)
        summary = payloads[raw_name]["summary"]
        ax.text(
            0.99,
            0.10,
            f"best G{int(summary['first_best_generation'])} | plateau {int(summary['plateau_generations'])} gens | {summary['top_tag_1']}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.9,
            color="#5A5A5A",
        )
        add_panel_label(ax, panel)

    axes[-1].set_xlabel("Generation")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, markerfacecolor=OPERATOR_COLORS[op], markeredgecolor="white", markeredgewidth=0.6, label=OPERATOR_DISPLAY[op])
        for op in OPERATOR_ORDER
        if op in {"e1", "e2", "m1", "m2", "t2"}
    ]
    axes[0].legend(handles=handles, ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.08), fontsize=5.8, columnspacing=0.8, handletextpad=0.3)

    fig.text(0.10, 0.985, "Supplementary Figure | Detailed best-so-far evolution for each LLM", ha="left", va="bottom", fontsize=8.2, fontweight="bold")
    save_pub_py(fig, SUPP_BASENAME)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payloads = build_model_payloads()
    build_source_tables(payloads)
    plot_figure10(payloads)
    plot_figure11(payloads)
    plot_supplementary(payloads)


if __name__ == "__main__":
    main()
