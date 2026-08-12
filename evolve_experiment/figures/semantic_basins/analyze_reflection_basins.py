from __future__ import annotations

import csv
import ast
import importlib.util
import json
import math
from pathlib import Path
from statistics import median


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
EVOLUTION_ROOT = ROOT / "evolve_experiment" / "evolution"
ANALYSIS = ROOT / "evolve_experiment" / "figures" / "semantic_basins" / "make_search_space_complexity_figures.py"
OUT = ROOT / "evolve_experiment" / "figures" / "semantic_basins" / "reflection_basin_comparison.csv"


def load_analysis_module():
    # The repository's plotting script also imports matplotlib.  This analysis
    # only needs its data/metric functions, so load those definitions without
    # making the numerical comparison depend on plotting packages.
    source = ANALYSIS.read_text(encoding="utf-8")
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            keep.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module in {"__future__", "collections", "dataclasses", "pathlib"}:
            keep.append(node)
        elif isinstance(node, ast.Import) and all(alias.name not in {"matplotlib", "matplotlib.pyplot", "numpy"} for alias in node.names):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & {"BASIN_DISTANCE_THRESHOLD", "VIEW_WEIGHTS", "IMPROVEMENT_LEVELS"}:
                keep.append(node)
    module = type("MetricModule", (), {})()
    namespace = module.__dict__
    namespace.update({"math": math, "Path": Path})
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(ANALYSIS), "exec"), namespace)
    return module


RUNS = [
    ("Crime", "Reflection off", EVOLUTION_ROOT / "reflection_off" / "Crime_post_with_precompute_reflection_off_seeded" / "results", "Crime"),
    ("Crime", "Reflection on", EVOLUTION_ROOT / "real" / "Crime_sampled_post_with_precompute" / "results", "Crime_sampled"),
    ("HI-II-14", "Reflection off", EVOLUTION_ROOT / "reflection_off" / "HI-II-14_sampled_post_with_precompute_reflection_off_seeded" / "results", "HI-II-14_sampled"),
    ("HI-II-14", "Reflection on", EVOLUTION_ROOT / "real" / "HI-II-14_sampled_post_with_precompute" / "results", "HI-II-14_sampled"),
    ("Youtube", "Reflection off", EVOLUTION_ROOT / "reflection_off" / "Youtube_sampled_post_with_precompute_reflection_off_seeded" / "results", "Youtube_sampled"),
    ("Youtube", "Reflection on", EVOLUTION_ROOT / "real" / "Youtube_sampled_post_with_precompute" / "results", "Youtube_sampled"),
]


def quantile(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def summarize(mod, family, condition, results_dir, dataset_key):
    spec = {"dataset_key": dataset_key, "results_dir": results_dir}
    trace = mod.build_candidate_trace(spec)
    basin_map, distances = mod.assign_basins(trace)
    objectives = [float(x.objective) for x in trace]
    basin_counts = {}
    basin_best = {}
    for rec in trace:
        b = basin_map[rec.unique_id]
        basin_counts[b] = basin_counts.get(b, 0) + 1
        basin_best[b] = min(basin_best.get(b, float("inf")), rec.objective)
    best = min(objectives)
    top10_cut = quantile(objectives, 0.10)
    top10 = [x for x in objectives if x <= top10_cut]
    top10_basins = {basin_map[x.unique_id] for x in trace if x.objective <= top10_cut}
    high_quality_cut = quantile(objectives, 0.25)
    high_quality_basins = {basin_map[x.unique_id] for x in trace if x.objective <= high_quality_cut}
    last_generation = max(x.generation for x in trace)
    gen_rows = mod.generation_summary(trace, basin_map, distances)
    step_rows = mod.step_summary(trace, basin_map)
    # Budget-matched statistics use the first min(on, off) evaluated candidates.
    row = {
        "family": family,
        "condition": condition,
        "n_evaluations": len(trace),
        "n_unique_candidates": len(mod.collapse_unique_candidates(trace)),
        "n_generations": last_generation,
        "total_basins": len(set(basin_map.values())),
        "final_effective_basins": float(step_rows[-1]["effective_basins"]),
        "mean_generation_pair_distance": sum(float(x["div_pair"]) for x in gen_rows) / len(gen_rows),
        "best_objective": best,
        "median_objective": median(objectives),
        "q25_objective": quantile(objectives, 0.25),
        "top10_mean_objective": sum(top10) / len(top10),
        "top10_basin_count": len(top10_basins),
        "top25_basin_count": len(high_quality_basins),
        "top10_basin_fraction": len(top10_basins) / max(len(set(basin_map.values())), 1),
        "top25_basin_fraction": len(high_quality_basins) / max(len(set(basin_map.values())), 1),
        "largest_basin_fraction": max(basin_counts.values()) / len(trace),
        "basin_concentration_hhi": sum((count / len(trace)) ** 2 for count in basin_counts.values()),
        "c_tau_90pct_gain": "",
    }
    complexity = mod.search_space_complexity_rows(f"{family} {condition}", gen_rows, step_rows)
    target = next((x for x in complexity if abs(float(x["improvement_fraction"]) - 0.90) < 1e-9), None)
    if target and target["c_tau"] != "":
        row["c_tau_90pct_gain"] = int(target["c_tau"])
    return row


def summarize_all_pops(mod, family, condition, results_dir, dataset_key, budget=None):
    """Use all_pops.json so the comparison counts generated/evaluated candidates.

    The generation snapshots are survivor populations and have different sizes
    in the on/off runs. all_pops is the fairer resource-level trace; its file
    order is the recorded evaluation order, so generation is not used here.
    """
    path = results_dir / "pops" / dataset_key / "all_pops.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    reflection_lookup = mod.load_reflection_lookup(results_dir / "reflections" / dataset_key)
    trace = []
    for i, candidate in enumerate(payload, start=1):
        objective = candidate.get("objective")
        if objective is None:
            continue
        unique_id = str(candidate.get("unique_id") or f"{family}_{condition}_{i}")
        reflected = reflection_lookup.get(unique_id, {})
        trace.append(mod.CandidateRecord(
            generation=1,
            candidate_index=i,
            evaluation_step=i,
            unique_id=unique_id,
            objective=float(objective),
            operator=str((candidate.get("other_inf") or {}).get("operator", "unknown")),
            description=mod.choose_best_text(str(candidate.get("algorithm") or ""), reflected.get("description", "")),
            reflection=mod.choose_best_text(str(candidate.get("llm_reflection") or ""), reflected.get("reflection", "")),
            code=str(candidate.get("code") or "").strip() or reflected.get("code", ""),
            source_path=str(path),
        ))
    if budget is not None:
        trace = trace[:budget]
    basin_map, distances = mod.assign_basins(trace)
    objectives = [float(x.objective) for x in trace]
    basin_counts = {}
    for rec in trace:
        b = basin_map[rec.unique_id]
        basin_counts[b] = basin_counts.get(b, 0) + 1
    best = min(objectives)
    top10_cut = quantile(objectives, 0.10)
    top25_cut = quantile(objectives, 0.25)
    top10_basins = {basin_map[x.unique_id] for x in trace if x.objective <= top10_cut}
    top25_basins = {basin_map[x.unique_id] for x in trace if x.objective <= top25_cut}
    counts = [x for x in objectives if x <= top10_cut]
    row = {
        "family": family, "condition": condition, "budget": budget or "full", "n_evaluations": len(trace),
        "n_unique_candidates": len(mod.collapse_unique_candidates(trace)), "n_generations": "all_pops",
        "total_basins": len(set(basin_map.values())), "final_effective_basins": "",
        "mean_generation_pair_distance": "", "best_objective": best,
        "median_objective": median(objectives), "q25_objective": quantile(objectives, 0.25),
        "top10_mean_objective": sum(counts) / len(counts), "top10_basin_count": len(top10_basins),
        "top25_basin_count": len(top25_basins),
        "top10_basin_fraction": len(top10_basins) / max(len(set(basin_map.values())), 1),
        "top25_basin_fraction": len(top25_basins) / max(len(set(basin_map.values())), 1),
        "largest_basin_fraction": max(basin_counts.values()) / len(trace),
        "basin_concentration_hhi": sum((count / len(trace)) ** 2 for count in basin_counts.values()),
        "c_tau_90pct_gain": "",
    }
    return row


def main():
    mod = load_analysis_module()
    traces = {}
    for run in RUNS:
        family, condition, results_dir, dataset_key = run
        # Build the full trace once to establish a matched prefix budget.
        traces[(family, condition)] = summarize_all_pops(mod, family, condition, results_dir, dataset_key)
    rows = []
    for family in sorted({x[0] for x in RUNS}):
        off = next(x for x in RUNS if x[0] == family and x[1] == "Reflection off")
        on = next(x for x in RUNS if x[0] == family and x[1] == "Reflection on")
        off_n = int(traces[(family, "Reflection off")]["n_evaluations"])
        on_n = int(traces[(family, "Reflection on")]["n_evaluations"])
        budget = min(off_n, on_n)
        rows.extend([
            summarize_all_pops(mod, *off, budget=budget),
            summarize_all_pops(mod, *on, budget=budget),
        ])
    fields = list(rows[0])
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(";".join(fields))
    for row in rows:
        print(";".join(str(row[k]) for k in fields))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
