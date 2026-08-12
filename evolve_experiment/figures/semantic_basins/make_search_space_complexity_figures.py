from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
EVOLUTION_ROOT = ROOT / "evolve_experiment" / "evolution"
OUT_DIR = ROOT / "evolve_experiment" / "figures" / "semantic_basins"

BASIN_DISTANCE_THRESHOLD = 0.42
VIEW_WEIGHTS = {
    "description": 1.0 / 3.0,
    "reflection": 1.0 / 3.0,
    "code": 1.0 / 3.0,
}
IMPROVEMENT_LEVELS = [0.25, 0.50, 0.75, 0.90, 1.00]

RUN_SPECS = [
    {
        "label": "Enron sampled",
        "run_key": "enron_sampled",
        "dataset_key": "Enron_sampled",
        "results_dir": EVOLUTION_ROOT / "real" / "Enron_sampled_post_with_precompute_new" / "results",
    },
    {
        "label": "Crime sampled",
        "run_key": "crime_sampled",
        "dataset_key": "Crime_sampled",
        "results_dir": EVOLUTION_ROOT / "real" / "Crime_sampled_post_with_precompute" / "results",
    },
    {
        "label": "YouTube sampled",
        "run_key": "youtube_sampled",
        "dataset_key": "Youtube_sampled",
        "results_dir": EVOLUTION_ROOT / "real" / "Youtube_sampled_post_with_precompute" / "results",
    },
    {
        "label": "Synthetic SBM-1000",
        "run_key": "synthetic_sbm_1000",
        "dataset_key": "synthetic_sbm_1000",
        "results_dir": EVOLUTION_ROOT / "adaptive" / "synthetic_sbm_1000__continuous" / "results",
    },
    {
        "label": "Synthetic SBM-1000 (State soft)",
        "run_key": "synthetic_sbm_1000_state_soft",
        "dataset_key": "synthetic_sbm_1000",
        "results_dir": EVOLUTION_ROOT / "adaptive" / "synthetic_sbm_1000__state_soft__rep1" / "results",
    },
    {
        "label": "Synthetic ER-1000",
        "run_key": "synthetic_er_1000",
        "dataset_key": "synthetic_er_1000",
        "results_dir": EVOLUTION_ROOT / "synthetic" / "group1_sbm_er" / "gpt-4.1-mini" / "synthetic_er_1000_20260330_133530" / "results",
    },
    {
        "label": "Synthetic WS-1000",
        "run_key": "synthetic_ws_1000",
        "dataset_key": "synthetic_ws_1000",
        "results_dir": EVOLUTION_ROOT / "synthetic" / "group2_ws_uniform" / "gpt-4.1-mini" / "synthetic_ws_1000_20260330_100257" / "results",
    },
    {
        "label": "HI-II-14 sampled",
        "run_key": "hii14_sampled",
        "dataset_key": "HI-II-14_sampled",
        "results_dir": EVOLUTION_ROOT / "real" / "HI-II-14_sampled_post_with_precompute" / "results",
    },
]

BEST_COLOR = "#2F5D7E"
DIVERSITY_COLOR = "#4C78A8"
NOVEL_COLOR = "#E45756"
NEFF_COLOR = "#54A24B"
BASIN_COLOR = "#B279A2"
SUMMARY_COLORS = ["#4C78A8", "#72B7B2", "#ECAE4E", "#E45756", "#B279A2"]


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 8,
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


@dataclass
class CandidateRecord:
    generation: int
    candidate_index: int
    evaluation_step: int
    unique_id: str
    objective: float
    operator: str
    description: str
    reflection: str
    code: str
    source_path: str


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}
        self.rank = {item: 0 for item in items}

    def find(self, item: str) -> str:
        root = self.parent[item]
        if root != item:
            self.parent[item] = self.find(root)
        return self.parent[item]

    def union(self, a: str, b: str) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
            return
        if rank_a > rank_b:
            self.parent[root_b] = root_a
            return
        self.parent[root_b] = root_a
        self.rank[root_a] += 1


def save_pub_py(fig: plt.Figure, basename: Path, dpi: int = 600) -> None:
    fig.savefig(f"{basename}.svg", bbox_inches="tight")
    fig.savefig(f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{basename}.tiff", dpi=dpi, bbox_inches="tight")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_generation(path: Path) -> int | None:
    match = re.search(r"population_generation_(-?\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"generation_(-?\d+)\.json$", path.name)
    if match:
        return int(match.group(1))
    return None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_llm_raw_response(raw_response: str) -> dict[str, str]:
    if not raw_response:
        return {"description": "", "reflection": "", "code": ""}
    description = ""
    reflection = ""
    code = ""

    reflection_match = re.search(
        r"Reflection:\s*(.*?)(?=\s*Algorithm Description:|\s*Code:|$)",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if reflection_match:
        reflection = normalize_text(reflection_match.group(1))

    description_match = re.search(
        r"Algorithm Description:\s*\{?(.*?)\}?(?=\s*Code:|$)",
        raw_response,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if description_match:
        description = normalize_text(description_match.group(1))

    code_match = re.search(r"Code:\s*```(?:python)?\s*(.*?)```", raw_response, flags=re.IGNORECASE | re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()

    return {"description": description, "reflection": reflection, "code": code}


def tokenize_document(text: str, *, code_view: bool) -> list[str]:
    text = text.lower()
    if not text:
        return []
    tokens = re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", text)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if "_" in token:
            expanded.extend(part for part in token.split("_") if part)
        if code_view and len(token) > 6:
            expanded.append(token[:6])
    return expanded


def build_tfidf_vectors(documents: list[str], *, code_view: bool) -> list[dict[str, float]]:
    tokenized_docs = [tokenize_document(document, code_view=code_view) for document in documents]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_docs:
        for token in set(tokens):
            document_frequency[token] += 1

    total_docs = max(len(documents), 1)
    vectors: list[dict[str, float]] = []
    for tokens in tokenized_docs:
        if not tokens:
            vectors.append({})
            continue
        counts = Counter(tokens)
        total_tokens = float(len(tokens))
        vector: dict[str, float] = {}
        for token, count in counts.items():
            idf = math.log((1.0 + total_docs) / (1.0 + document_frequency[token])) + 1.0
            vector[token] = (count / total_tokens) * idf
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm > 0:
            for token in list(vector):
                vector[token] /= norm
        vectors.append(vector)
    return vectors


def cosine_sparse(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return float(sum(value * right.get(token, 0.0) for token, value in left.items()))


def load_reflection_lookup(reflections_dir: Path) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    if not reflections_dir.exists():
        return lookup

    for path in sorted(reflections_dir.glob("generation_*.json"), key=lambda item: extract_generation(item) or -9999):
        payload = json.loads(path.read_text(encoding="utf-8"))
        entries = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            unique_id = str(entry.get("offspring_unique_id", "")).strip()
            if not unique_id:
                continue
            parsed = parse_llm_raw_response(str(entry.get("llm_raw_response", "") or ""))
            lookup[unique_id] = {
                "description": normalize_text(parsed["description"]),
                "reflection": normalize_text(str(entry.get("llm_reflection", "") or parsed["reflection"])),
                "code": parsed["code"],
            }
    return lookup


def choose_best_text(*values: str) -> str:
    best = ""
    for value in values:
        cleaned = normalize_text(value)
        if len(cleaned) > len(best):
            best = cleaned
    return best


def build_candidate_trace(run_spec: dict[str, object]) -> list[CandidateRecord]:
    dataset_key = str(run_spec["dataset_key"])
    results_dir = Path(run_spec["results_dir"])
    pops_dir = results_dir / "pops" / dataset_key
    reflections_dir = results_dir / "reflections" / dataset_key

    if not pops_dir.exists():
        raise FileNotFoundError(f"Missing pops directory: {pops_dir}")

    reflection_lookup = load_reflection_lookup(reflections_dir)
    generation_paths = sorted(pops_dir.glob("population_generation_*.json"), key=lambda item: extract_generation(item) or -9999)

    trace: list[CandidateRecord] = []
    evaluation_step = 0
    for path in generation_paths:
        generation = extract_generation(path)
        if generation is None or generation <= 0:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            continue
        for candidate_index, candidate in enumerate(payload, start=1):
            unique_id = str(candidate.get("unique_id", "")).strip()
            if not unique_id:
                unique_id = f"{dataset_key}_g{generation}_i{candidate_index}"
            parsed = parse_llm_raw_response(str(candidate.get("llm_raw_response", "") or ""))
            reflected = reflection_lookup.get(unique_id, {})
            description = choose_best_text(
                str(candidate.get("algorithm", "") or ""),
                reflected.get("description", ""),
                parsed["description"],
            )
            reflection = choose_best_text(
                str(candidate.get("llm_reflection", "") or ""),
                reflected.get("reflection", ""),
                parsed["reflection"],
            )
            code = str(candidate.get("code", "") or "").strip() or reflected.get("code", "") or parsed["code"]
            operator = str(candidate.get("other_inf", {}).get("operator", "unknown")).strip().lower() or "unknown"
            objective = candidate.get("objective")
            if objective is None:
                continue
            evaluation_step += 1
            trace.append(
                CandidateRecord(
                    generation=int(generation),
                    candidate_index=int(candidate_index),
                    evaluation_step=int(evaluation_step),
                    unique_id=unique_id,
                    objective=float(objective),
                    operator=operator,
                    description=description,
                    reflection=reflection,
                    code=code,
                    source_path=str(path),
                )
            )

    if not trace:
        raise ValueError(f"No candidate records found in {pops_dir}")
    return trace


def collapse_unique_candidates(trace: list[CandidateRecord]) -> dict[str, dict[str, str]]:
    unique_docs: dict[str, dict[str, str]] = {}
    for record in trace:
        current = unique_docs.get(record.unique_id)
        if current is None:
            unique_docs[record.unique_id] = {
                "description": record.description,
                "reflection": record.reflection,
                "code": record.code,
            }
            continue
        current["description"] = choose_best_text(current["description"], record.description)
        current["reflection"] = choose_best_text(current["reflection"], record.reflection)
        if len(record.code) > len(current["code"]):
            current["code"] = record.code
    return unique_docs


def assign_basins(trace: list[CandidateRecord]) -> tuple[dict[str, int], dict[tuple[str, str], float]]:
    unique_docs = collapse_unique_candidates(trace)
    unique_ids = sorted(unique_docs)

    description_docs = [unique_docs[unique_id]["description"] for unique_id in unique_ids]
    reflection_docs = [unique_docs[unique_id]["reflection"] for unique_id in unique_ids]
    code_docs = [unique_docs[unique_id]["code"] for unique_id in unique_ids]

    description_vectors = build_tfidf_vectors(description_docs, code_view=False)
    reflection_vectors = build_tfidf_vectors(reflection_docs, code_view=False)
    code_vectors = build_tfidf_vectors(code_docs, code_view=True)

    union_find = UnionFind(unique_ids)
    distance_lookup: dict[tuple[str, str], float] = {}

    for i, left_id in enumerate(unique_ids):
        distance_lookup[(left_id, left_id)] = 0.0
        for j in range(i + 1, len(unique_ids)):
            right_id = unique_ids[j]
            similarity = 0.0
            similarity += VIEW_WEIGHTS["description"] * cosine_sparse(description_vectors[i], description_vectors[j])
            similarity += VIEW_WEIGHTS["reflection"] * cosine_sparse(reflection_vectors[i], reflection_vectors[j])
            similarity += VIEW_WEIGHTS["code"] * cosine_sparse(code_vectors[i], code_vectors[j])
            distance = max(0.0, 1.0 - similarity)
            distance_lookup[(left_id, right_id)] = distance
            distance_lookup[(right_id, left_id)] = distance
            if distance <= BASIN_DISTANCE_THRESHOLD:
                union_find.union(left_id, right_id)

    root_to_basin: dict[str, int] = {}
    basin_map: dict[str, int] = {}
    next_basin = 1
    for unique_id in unique_ids:
        root = union_find.find(unique_id)
        if root not in root_to_basin:
            root_to_basin[root] = next_basin
            next_basin += 1
        basin_map[unique_id] = root_to_basin[root]
    return basin_map, distance_lookup


def average_pairwise_distance(candidate_ids: list[str], distance_lookup: dict[tuple[str, str], float]) -> float:
    if len(candidate_ids) <= 1:
        return 0.0
    total = 0.0
    count = 0
    for i, left in enumerate(candidate_ids):
        for right in candidate_ids[i + 1 :]:
            total += distance_lookup[(left, right)]
            count += 1
    return total / count if count else 0.0


def generation_summary(
    trace: list[CandidateRecord],
    basin_map: dict[str, int],
    distance_lookup: dict[tuple[str, str], float],
) -> list[dict[str, object]]:
    by_generation: dict[int, list[CandidateRecord]] = defaultdict(list)
    for record in trace:
        by_generation[record.generation].append(record)

    seen_basins: set[int] = set()
    best_so_far = math.inf
    rows: list[dict[str, object]] = []
    for generation in sorted(by_generation):
        records = by_generation[generation]
        candidate_ids = [record.unique_id for record in records]
        basin_ids = {basin_map[record.unique_id] for record in records}
        novel_count = len([basin_id for basin_id in basin_ids if basin_id not in seen_basins])
        seen_basins.update(basin_ids)
        generation_best = min(record.objective for record in records)
        best_so_far = min(best_so_far, generation_best)
        rows.append(
            {
                "generation": generation,
                "population_size": len(records),
                "generation_best_objective": generation_best,
                "best_so_far_objective": best_so_far,
                "div_pair": average_pairwise_distance(candidate_ids, distance_lookup),
                "novel_basins": novel_count,
                "generation_basin_count": len(basin_ids),
                "cumulative_unique_basins": len(seen_basins),
            }
        )
    return rows


def step_summary(trace: list[CandidateRecord], basin_map: dict[str, int]) -> list[dict[str, object]]:
    best_so_far = math.inf
    basin_counts: Counter[int] = Counter()
    seen_basins: set[int] = set()
    rows: list[dict[str, object]] = []

    for record in trace:
        basin_id = basin_map[record.unique_id]
        basin_counts[basin_id] += 1
        seen_basins.add(basin_id)
        best_so_far = min(best_so_far, record.objective)

        total = sum(basin_counts.values())
        entropy = 0.0
        for count in basin_counts.values():
            probability = count / total
            entropy -= probability * math.log(probability)
        effective_basins = math.exp(entropy) if basin_counts else 0.0

        rows.append(
            {
                "evaluation_step": record.evaluation_step,
                "generation": record.generation,
                "candidate_index": record.candidate_index,
                "unique_id": record.unique_id,
                "objective": record.objective,
                "best_so_far": best_so_far,
                "operator": record.operator,
                "basin_id": basin_id,
                "cumulative_unique_basins": len(seen_basins),
                "effective_basins": effective_basins,
            }
        )
    return rows


def search_space_complexity_rows(
    run_label: str,
    generation_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not generation_rows or not step_rows:
        return []

    start_best = float(generation_rows[0]["generation_best_objective"])
    final_best = min(float(row["best_so_far"]) for row in step_rows)
    total_improvement = max(0.0, start_best - final_best)

    rows: list[dict[str, object]] = []
    for level in IMPROVEMENT_LEVELS:
        target = start_best - level * total_improvement
        reached = next((row for row in step_rows if float(row["best_so_far"]) <= target + 1e-12), None)
        rows.append(
            {
                "run_label": run_label,
                "improvement_fraction": level,
                "target_objective": target,
                "start_best": start_best,
                "final_best": final_best,
                "c_tau": int(reached["cumulative_unique_basins"]) if reached else "",
                "evaluation_step": int(reached["evaluation_step"]) if reached else "",
                "generation": int(reached["generation"]) if reached else "",
            }
        )
    return rows


def draw_population_diversity_panels(
    axes: list[plt.Axes],
    run_label: str,
    generation_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
    *,
    annotate_row_label: bool,
) -> None:
    generations = np.array([int(row["generation"]) for row in generation_rows], dtype=int)
    div_pair = np.array([float(row["div_pair"]) for row in generation_rows], dtype=float)
    novel = np.array([int(row["novel_basins"]) for row in generation_rows], dtype=int)
    eval_steps = np.array([int(row["evaluation_step"]) for row in step_rows], dtype=int)
    effective_basins = np.array([float(row["effective_basins"]) for row in step_rows], dtype=float)
    cumulative_basins = np.array([int(row["cumulative_unique_basins"]) for row in step_rows], dtype=int)

    ax = axes[0]
    ax.plot(generations, div_pair, color=DIVERSITY_COLOR, linewidth=1.2)
    ax.set_title("Population diversity")
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\mathrm{Div}_{\mathrm{pair}}(g)$")
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    if annotate_row_label:
        ax.text(-0.20, 1.22, run_label, transform=ax.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

    ax = axes[1]
    ax.bar(generations, novel, width=0.75, color=NOVEL_COLOR, alpha=0.85)
    ax.set_title("Novel basin discovery")
    ax.set_xlabel("Generation")
    ax.set_ylabel(r"$\mathrm{Novel}_{\tau}(g)$")
    ax.grid(axis="y", linestyle=":", alpha=0.20)

    ax = axes[2]
    ax.plot(eval_steps, effective_basins, color=NEFF_COLOR, linewidth=1.1, label=r"$N_{\mathrm{eff}}(t)$")
    ax2 = ax.twinx()
    ax2.plot(eval_steps, cumulative_basins, color=BASIN_COLOR, linewidth=1.0, linestyle="--", label="Cumulative basins")
    ax.set_title("Cumulative exploration")
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel(r"$N_{\mathrm{eff}}(t)$")
    ax2.set_ylabel("Unique basins")
    ax.grid(axis="y", linestyle=":", alpha=0.20)
    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", fontsize=6)


def plot_population_diversity(
    run_label: str,
    run_key: str,
    generation_rows: list[dict[str, object]],
    step_rows: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 58 / 25.4))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.18, top=0.84, wspace=0.34)
    draw_population_diversity_panels(list(axes), run_label, generation_rows, step_rows, annotate_row_label=False)
    fig.suptitle(f"{run_label}: diversity and basin exploration", fontsize=9, y=0.98)
    save_pub_py(fig, OUT_DIR / f"figure12_{run_key}_population_diversity")
    plt.close(fig)


def plot_population_diversity_pair(
    top_label: str,
    top_generation_rows: list[dict[str, object]],
    top_step_rows: list[dict[str, object]],
    bottom_label: str,
    bottom_generation_rows: list[dict[str, object]],
    bottom_step_rows: list[dict[str, object]],
    basename: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 118 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.88, wspace=0.34, hspace=0.62)
    draw_population_diversity_panels(list(axes[0]), top_label, top_generation_rows, top_step_rows, annotate_row_label=True)
    draw_population_diversity_panels(list(axes[1]), bottom_label, bottom_generation_rows, bottom_step_rows, annotate_row_label=True)
    fig.suptitle(title, fontsize=9, y=0.98)

    save_pub_py(fig, OUT_DIR / basename)
    plt.close(fig)


def plot_population_diversity_grid(
    panels: list[tuple[str, list[dict[str, object]], list[dict[str, object]]]],
    basename: str,
    title: str,
) -> None:
    """Render several runs as a vertically stacked, shared-style 3-column figure."""
    n_rows = len(panels)
    if n_rows == 0:
        return
    fig, axes = plt.subplots(n_rows, 3, figsize=(183 / 25.4, (58 * n_rows) / 25.4), squeeze=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.08,
        top=0.93,
        wspace=0.34,
        hspace=0.72 if n_rows > 2 else 0.62,
    )
    for row_axes, (run_label, generation_rows, step_rows) in zip(axes, panels):
        draw_population_diversity_panels(
            list(row_axes),
            run_label,
            generation_rows,
            step_rows,
            annotate_row_label=True,
        )
    fig.suptitle(title, fontsize=9, y=0.985)
    save_pub_py(fig, OUT_DIR / basename)
    plt.close(fig)


def plot_complexity_summary(summary_rows: list[dict[str, object]]) -> None:
    if not summary_rows:
        return

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["run_label"])].append(row)

    labels = list(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.88, wspace=0.28)

    ax = axes[0]
    for idx, label in enumerate(labels):
        rows = sorted(grouped[label], key=lambda item: float(item["improvement_fraction"]))
        x = [100.0 * float(row["improvement_fraction"]) for row in rows]
        y = [float(row["c_tau"]) for row in rows if str(row["c_tau"]) != ""]
        x = x[: len(y)]
        ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.1, color=SUMMARY_COLORS[idx % len(SUMMARY_COLORS)], label=label)
    ax.set_title(r"Result-induced complexity, $\mathcal{C}_{\tau}(A)$")
    ax.set_xlabel("Improvement milestone (%)")
    ax.set_ylabel("Basins visited before target")
    ax.set_xticks([25, 50, 75, 90, 100])
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    ax.legend(loc="upper left", fontsize=6)

    ax = axes[1]
    ninety_rows = []
    for label in labels:
        match = next((row for row in grouped[label] if abs(float(row["improvement_fraction"]) - 0.90) < 1e-12), None)
        if match is not None and str(match["c_tau"]) != "":
            ninety_rows.append((label, float(match["c_tau"])))
    positions = np.arange(len(ninety_rows))
    ax.scatter(
        positions,
        [value for _, value in ninety_rows],
        s=24,
        color=[SUMMARY_COLORS[idx % len(SUMMARY_COLORS)] for idx in range(len(ninety_rows))],
        zorder=3,
    )
    ax.set_title(r"$\mathcal{C}_{\tau}(A)$ at 90% improvement")
    ax.set_ylabel("Basins visited")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for label, _ in ninety_rows], rotation=15, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.25)

    save_pub_py(fig, OUT_DIR / "figure13_search_space_complexity_summary")
    plt.close(fig)


def write_markdown_notes(
    summary_rows: list[dict[str, object]],
    generation_rows_by_run: dict[str, list[dict[str, object]]],
) -> None:
    notes_path = OUT_DIR / "search_space_complexity_notes.md"
    lines = [
        "# Search-space complexity notes",
        "",
        f"- Basin distance threshold: `{BASIN_DISTANCE_THRESHOLD}`",
        "- Similarity views: description / reflection / code with uniform weights.",
        "- Embedding backend: local TF-IDF over each view, then weighted cosine similarity.",
        "- Single-run setting: this analysis reports `C_tau(A)` but does not estimate the cross-run lower envelope.",
        "- Improvement milestones are defined relative to the first positive-generation best objective and the final best-so-far objective.",
        "",
        "## Run summaries",
        "",
    ]
    for run_label, rows in generation_rows_by_run.items():
        start_best = float(rows[0]["generation_best_objective"])
        final_best = min(float(row["best_so_far_objective"]) for row in rows)
        total_basins = int(rows[-1]["cumulative_unique_basins"])
        lines.append(f"### {run_label}")
        lines.append("")
        lines.append(f"- Start best ANC: `{start_best:.5f}`")
        lines.append(f"- Final best ANC: `{final_best:.5f}`")
        lines.append(f"- Total unique basins visited: `{total_basins}`")
        lines.append("")
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    complexity_rows_all: list[dict[str, object]] = []
    generation_rows_by_run: dict[str, list[dict[str, object]]] = {}
    step_rows_by_run: dict[str, list[dict[str, object]]] = {}

    for run_spec in RUN_SPECS:
        run_label = str(run_spec["label"])
        run_key = str(run_spec["run_key"])

        trace = build_candidate_trace(run_spec)
        basin_map, distance_lookup = assign_basins(trace)
        generation_rows = generation_summary(trace, basin_map, distance_lookup)
        step_rows = step_summary(trace, basin_map)
        complexity_rows = search_space_complexity_rows(run_label, generation_rows, step_rows)

        generation_rows_by_run[run_label] = generation_rows
        step_rows_by_run[run_label] = step_rows
        complexity_rows_all.extend(complexity_rows)

        basin_rows = [
            {
                "unique_id": record.unique_id,
                "generation": record.generation,
                "candidate_index": record.candidate_index,
                "evaluation_step": record.evaluation_step,
                "objective": record.objective,
                "operator": record.operator,
                "basin_id": basin_map[record.unique_id],
                "description": record.description,
                "reflection": record.reflection,
                "source_path": record.source_path,
            }
            for record in trace
        ]
        write_csv(
            OUT_DIR / f"{run_key}_candidate_trace.csv",
            [
                "unique_id",
                "generation",
                "candidate_index",
                "evaluation_step",
                "objective",
                "operator",
                "basin_id",
                "description",
                "reflection",
                "source_path",
            ],
            basin_rows,
        )
        write_csv(
            OUT_DIR / f"{run_key}_generation_diversity.csv",
            [
                "generation",
                "population_size",
                "generation_best_objective",
                "best_so_far_objective",
                "div_pair",
                "novel_basins",
                "generation_basin_count",
                "cumulative_unique_basins",
            ],
            generation_rows,
        )
        write_csv(
            OUT_DIR / f"{run_key}_step_diversity.csv",
            [
                "evaluation_step",
                "generation",
                "candidate_index",
                "unique_id",
                "objective",
                "best_so_far",
                "operator",
                "basin_id",
                "cumulative_unique_basins",
                "effective_basins",
            ],
            step_rows,
        )
        write_csv(
            OUT_DIR / f"{run_key}_search_space_complexity.csv",
            [
                "run_label",
                "improvement_fraction",
                "target_objective",
                "start_best",
                "final_best",
                "c_tau",
                "evaluation_step",
                "generation",
            ],
            complexity_rows,
        )
        plot_population_diversity(run_label, run_key, generation_rows, step_rows)

    plot_population_diversity_pair(
        top_label="Enron sampled",
        top_generation_rows=generation_rows_by_run["Enron sampled"],
        top_step_rows=step_rows_by_run["Enron sampled"],
        bottom_label="HI-II-14 sampled",
        bottom_generation_rows=generation_rows_by_run["HI-II-14 sampled"],
        bottom_step_rows=step_rows_by_run["HI-II-14 sampled"],
        basename="figure12_real_pair_population_diversity_2x3",
        title="Real sampled runs: diversity and basin exploration",
    )

    plot_population_diversity_pair(
        top_label="Synthetic SBM-1000",
        top_generation_rows=generation_rows_by_run["Synthetic SBM-1000"],
        top_step_rows=step_rows_by_run["Synthetic SBM-1000"],
        bottom_label="Synthetic SBM-1000 (State soft)",
        bottom_generation_rows=generation_rows_by_run["Synthetic SBM-1000 (State soft)"],
        bottom_step_rows=step_rows_by_run["Synthetic SBM-1000 (State soft)"],
        basename="figure12_sbm_pair_population_diversity_2x3",
        title="Synthetic SBM runs: diversity and basin exploration",
    )

    plot_population_diversity_grid(
        panels=[
            (
                "Synthetic ER-1000",
                generation_rows_by_run["Synthetic ER-1000"],
                step_rows_by_run["Synthetic ER-1000"],
            ),
            (
                "Synthetic SBM-1000",
                generation_rows_by_run["Synthetic SBM-1000"],
                step_rows_by_run["Synthetic SBM-1000"],
            ),
            (
                "Synthetic WS-1000",
                generation_rows_by_run["Synthetic WS-1000"],
                step_rows_by_run["Synthetic WS-1000"],
            ),
        ],
        basename="figure12_synthetic_population_diversity_3x3",
        title="Synthetic graph datasets: diversity and basin exploration",
    )

    plot_population_diversity_grid(
        panels=[
            (
                "Crime sampled",
                generation_rows_by_run["Crime sampled"],
                step_rows_by_run["Crime sampled"],
            ),
            (
                "YouTube sampled",
                generation_rows_by_run["YouTube sampled"],
                step_rows_by_run["YouTube sampled"],
            ),
        ],
        basename="figure12_real_sampled_population_diversity_2x3",
        title="Real sampled datasets: diversity and basin exploration",
    )

    write_csv(
        OUT_DIR / "figure13_search_space_complexity_summary.csv",
        [
            "run_label",
            "improvement_fraction",
            "target_objective",
            "start_best",
            "final_best",
            "c_tau",
            "evaluation_step",
            "generation",
        ],
        complexity_rows_all,
    )
    plot_complexity_summary(complexity_rows_all)
    write_markdown_notes(complexity_rows_all, generation_rows_by_run)


if __name__ == "__main__":
    main()
