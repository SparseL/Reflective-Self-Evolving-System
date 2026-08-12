from __future__ import annotations

import json
import csv
from pathlib import Path


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
EVOLVE_ROOT = ROOT / "evolve_experiment"
TRANSFER_ROOT = EVOLVE_ROOT / "transfer"
OUT_PATH = EVOLVE_ROOT / "processed" / "analysis_tables" / "rebuilt_metric_tables.md"
DOCSS_OUT_PATH = EVOLVE_ROOT / "processed" / "analysis_tables" / "resultanalysis_rebuilt_tables.md"

REAL_DATASET_ORDER = [
    "Crime",
    "Enron",
    "Enron_sample",
    "HI-II-14",
    "HI-II-14_sampled",
    "Youtube_sampled",
    "Youtube_sample4000",
    "Youtube_sample8000",
]

SYNTHETIC_DATASET_ORDER = [
    "synthetic_er_1000",
    "synthetic_sbm_1000",
    "synthetic_uniform_cost_1000",
    "synthetic_ws_1000",
]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def wrap_best(text: str, is_best: bool) -> str:
    return f"**{text}**" if is_best else text


def avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def read_real_eoh() -> dict[str, dict]:
    data = {}

    crime = load_json(TRANSFER_ROOT / "best_algo_on_Crime.json")
    data["Crime"] = {
        "method": "EoH-Best",
        "objective": crime["objective"],
        "time_select": crime["time_select"],
        "time_anc": crime["time_anc"],
        "fht_50": crime["other_inf"]["fht_50"],
        "fht_10": crime["other_inf"]["fht_10"],
        "lcc_at_k_frac": crime["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": crime["other_inf"]["anc_prefix_k"],
    }

    enron_sample = load_json(
        EVOLVE_ROOT / "evolution" / "real" / "Enron_sampled_post_with_precompute_new" / "results" / "pops_best" / "Enron_sampled" / "population_generation_50.json"
    )
    data["Enron_sample"] = {
        "method": "EoH-Best",
        "objective": enron_sample["objective"],
        "time_select": enron_sample["time_select"],
        "time_anc": enron_sample["time_anc"],
        "fht_50": enron_sample["other_inf"]["fht_50"],
        "fht_10": enron_sample["other_inf"]["fht_10"],
        "lcc_at_k_frac": enron_sample["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": enron_sample["other_inf"]["anc_prefix_k"],
    }

    enron = load_json(TRANSFER_ROOT / "best_algo_on_original_Enron_from_Enron_sampled_gen50.json")
    data["Enron"] = {
        "method": "EoH-Best",
        "objective": enron["objective"],
        "time_select": enron["time_select"],
        "time_anc": enron["time_anc"],
        "fht_50": enron["other_inf"]["fht_50"],
        "fht_10": enron["other_inf"]["fht_10"],
        "lcc_at_k_frac": enron["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": enron["other_inf"]["anc_prefix_k"],
    }

    hi_sample = load_json(TRANSFER_ROOT / "best_algo_on_HI-II-14_sampled.json")
    data["HI-II-14_sampled"] = {
        "method": "EoH-Best",
        "objective": hi_sample["objective"],
        "time_select": hi_sample["time_select"],
        "time_anc": hi_sample["time_anc"],
        "fht_50": hi_sample["other_inf"]["fht_50"],
        "fht_10": hi_sample["other_inf"]["fht_10"],
        "lcc_at_k_frac": hi_sample["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": hi_sample["other_inf"]["anc_prefix_k"],
    }

    hi_orig = load_json(TRANSFER_ROOT / "best_algo_on_original_HI-II-14.json")
    data["HI-II-14"] = {
        "method": "EoH-Best",
        "objective": hi_orig["objective"],
        "time_select": hi_orig["time_select"],
        "time_anc": hi_orig["time_anc"],
        "fht_50": hi_orig["other_inf"]["fht_50"],
        "fht_10": hi_orig["other_inf"]["fht_10"],
        "lcc_at_k_frac": hi_orig["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": hi_orig["other_inf"]["anc_prefix_k"],
    }

    yt_sample = load_json(TRANSFER_ROOT / "best_algo_on_youtube_sampled.json")
    data["Youtube_sampled"] = {
        "method": "EoH-Best",
        "objective": yt_sample["objective"],
        "time_select": yt_sample["time_select"],
        "time_anc": yt_sample["time_anc"],
        "fht_50": yt_sample["other_inf"]["fht_50"],
        "fht_10": yt_sample["other_inf"]["fht_10"],
        "lcc_at_k_frac": yt_sample["other_inf"]["lcc_at_k_frac"],
        "anc_prefix_k": yt_sample["other_inf"]["anc_prefix_k"],
    }

    yt4000 = load_json(TRANSFER_ROOT / "best_algo_on_youtube4000.json")
    yt4000_metrics = yt4000["result"][1]
    data["Youtube_sample4000"] = {
        "method": "EoH-Best",
        "objective": yt4000["result"][0],
        "time_select": yt4000_metrics["time_select"],
        "time_anc": yt4000_metrics["time_anc"],
        "fht_50": yt4000_metrics["fht_50"],
        "fht_10": yt4000_metrics["fht_10"],
        "lcc_at_k_frac": yt4000_metrics["lcc_at_k_frac"],
        "anc_prefix_k": yt4000_metrics["anc_prefix_k"],
    }

    yt8000 = load_json(TRANSFER_ROOT / "best_algo_on_youtube8000.json")
    yt8000_metrics = yt8000["result"][1]
    data["Youtube_sample8000"] = {
        "method": "EoH-Best",
        "objective": yt8000["result"][0],
        "time_select": yt8000_metrics["time_select"],
        "time_anc": yt8000_metrics["time_anc"],
        "fht_50": yt8000_metrics["fht_50"],
        "fht_10": yt8000_metrics["fht_10"],
        "lcc_at_k_frac": yt8000_metrics["lcc_at_k_frac"],
        "anc_prefix_k": yt8000_metrics["anc_prefix_k"],
    }

    return data


def read_real_baselines() -> dict[str, list[dict]]:
    root = EVOLVE_ROOT / "baseline_results_real"
    result = {}
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        rows = []
        for json_file in sorted(ds_dir.glob("*.json")):
            obj = load_json(json_file)
            metrics = obj["metrics"]
            rows.append(
                {
                    "method": obj["algorithm"],
                    "objective": obj["objective"],
                    "time_select": metrics["time_select"],
                    "time_anc": metrics["time_anc"],
                    "fht_50": metrics["fht_50"],
                    "fht_10": metrics["fht_10"],
                    "lcc_at_k_frac": metrics["lcc_at_k_frac"],
                    "anc_prefix_k": metrics["anc_prefix_k"],
                }
            )
        result[ds_dir.name] = rows
    return result


def parse_synthetic_eoh_summary() -> dict[str, dict]:
    data = {}
    summary_path = EVOLVE_ROOT / "processed" / "analysis_tables_synthetic_runs" / "synthetic_runs_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset = row["dataset"]
            data[dataset] = {
                "method": "EoH-Best",
                "objective": float(row["objective_best_overall"]),
                "time_select": float(row["time_select_at_best"]),
                "time_anc": float(row["time_anc_at_best"]),
                "fht_50": float(row["fht_50_at_best"]),
                "fht_10": float(row["fht_10_at_best"]),
                "lcc_at_k_frac": float(row["lcc_at_k_frac_at_best"]),
                "anc_prefix_k": float(row["anc_prefix_k_at_best"]),
            }
    return data


def read_synthetic_baselines() -> dict[str, list[dict]]:
    dir_map = {
        "synthetic_er_1000": ROOT / "baseline_results_synthetic" / "synthetic" / "er_1000" / "1000_detailed",
        "synthetic_sbm_1000": ROOT / "baseline_results_synthetic" / "synthetic" / "sbm_1000" / "1000_detailed",
        "synthetic_uniform_cost_1000": ROOT / "baseline_results_synthetic" / "synthetic" / "uniform_cost_1000" / "1000_detailed",
        "synthetic_ws_1000": ROOT / "baseline_results_synthetic" / "synthetic" / "ws_1000" / "1000_detailed",
    }
    result = {}
    for dataset, ds_dir in dir_map.items():
        rows = []
        for json_file in sorted(ds_dir.glob("*.json")):
            obj = load_json(json_file)
            details = obj.get("detailed_results", [])
            rows.append(
                {
                    "method": obj["algorithm"],
                    "objective": obj.get("summary_average", {}).get("objective", obj.get("objective")),
                    "time_select": avg([d["metrics"]["time_select"] for d in details]),
                    "time_anc": avg([d["metrics"]["time_anc"] for d in details]),
                    "fht_50": avg([d["metrics"]["fht_50"] for d in details]),
                    "fht_10": avg([d["metrics"]["fht_10"] for d in details]),
                    "lcc_at_k_frac": avg([d["metrics"]["lcc_at_k_frac"] for d in details]),
                    "anc_prefix_k": avg([d["metrics"]["anc_prefix_k"] for d in details]),
                }
            )
        result[dataset] = rows
    return result


def build_rows(dataset_order: list[str], eoh_map: dict[str, dict], base_map: dict[str, list[dict]]) -> dict[str, list[dict]]:
    data = {}
    for dataset in dataset_order:
        rows = []
        if dataset in eoh_map:
            rows.append(eoh_map[dataset])
        rows.extend(sorted(base_map.get(dataset, []), key=lambda x: (x["objective"], x["method"])))
        data[dataset] = rows
    return data


def build_overall_table(title: str, data_by_dataset: dict[str, list[dict]]) -> str:
    lines = [f"# {title}", ""]
    lines.append("| Dataset | Method | ANC ↓ | Diff vs EoH | Improvement (%) | Time Select (s) | Time ANC (s) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for dataset, rows in data_by_dataset.items():
        if not rows:
            continue
        best_obj = min(r["objective"] for r in rows if r["objective"] is not None)
        eoh_obj = next((r["objective"] for r in rows if r["method"] == "EoH-Best"), None)
        first = True
        for row in rows:
            ds_cell = f"**{dataset}**" if first else ""
            first = False
            obj_text = fmt_num(row["objective"])
            obj_text = wrap_best(obj_text, row["objective"] == best_obj)
            method_text = f"**{row['method']}**" if row["method"] == "EoH-Best" else row["method"]
            if row["method"] == "EoH-Best" or eoh_obj is None:
                diff_text = "–"
                imp_text = "–"
            else:
                diff = eoh_obj - row["objective"]
                imp = diff / row["objective"] * 100 if row["objective"] else None
                diff_text = fmt_num(diff)
                imp_text = fmt_num(imp, 2)
            lines.append(
                f"| {ds_cell} | {method_text} | {obj_text} | {diff_text} | {imp_text} | {fmt_num(row['time_select'])} | {fmt_num(row['time_anc'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_early_table(title: str, data_by_dataset: dict[str, list[dict]]) -> str:
    lines = [f"# {title}", ""]
    lines.append("| Dataset | Method | FHT50 ↓ | FHT10 ↓ | LCC@K Ratio ↓ | ANC Prefix K ↓ |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for dataset, rows in data_by_dataset.items():
        if not rows:
            continue
        best_fht50 = min(r["fht_50"] for r in rows if r["fht_50"] is not None)
        best_fht10 = min(r["fht_10"] for r in rows if r["fht_10"] is not None)
        best_lcc = min(r["lcc_at_k_frac"] for r in rows if r["lcc_at_k_frac"] is not None)
        best_prefix = min(r["anc_prefix_k"] for r in rows if r["anc_prefix_k"] is not None)
        first = True
        for row in rows:
            ds_cell = f"**{dataset}**" if first else ""
            first = False
            method_text = f"**{row['method']}**" if row["method"] == "EoH-Best" else row["method"]
            fht50_text = wrap_best(fmt_int(row["fht_50"]), row["fht_50"] == best_fht50)
            fht10_text = wrap_best(fmt_int(row["fht_10"]), row["fht_10"] == best_fht10)
            lcc_text = wrap_best(fmt_num(row["lcc_at_k_frac"]), row["lcc_at_k_frac"] == best_lcc)
            prefix_text = wrap_best(fmt_num(row["anc_prefix_k"]), row["anc_prefix_k"] == best_prefix)
            lines.append(f"| {ds_cell} | {method_text} | {fht50_text} | {fht10_text} | {lcc_text} | {prefix_text} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    real_eoh = read_real_eoh()
    real_baselines = read_real_baselines()
    synthetic_eoh = parse_synthetic_eoh_summary()
    synthetic_baselines = read_synthetic_baselines()

    real_data = build_rows(REAL_DATASET_ORDER, real_eoh, real_baselines)
    synthetic_data = build_rows(SYNTHETIC_DATASET_ORDER, synthetic_eoh, synthetic_baselines)

    parts = [
        "# 重新整理的算法指标结果表",
        "",
        "- 参考 `docss/resultanalysis.md` 的汇总风格，拆分为总体性能表与早期破坏能力表。",
        "- `Diff vs EoH = EoH-Best ANC - baseline ANC`；负值表示 `EoH-Best` 优于对应基线。",
        "- 仅展示当前结果目录中已存在的算法结果；缺失算法不强行补空行。",
        "",
        build_overall_table("Table 1. Real-world overall dismantling performance", real_data),
        build_early_table("Table 2. Real-world early-stage dismantling performance", real_data),
        build_overall_table("Table 3. Synthetic overall dismantling performance", synthetic_data),
        build_early_table("Table 4. Synthetic early-stage dismantling performance", synthetic_data),
    ]

    content = "\n".join(parts)
    OUT_PATH.write_text(content, encoding="utf-8")
    DOCSS_OUT_PATH.write_text(content, encoding="utf-8")
    print(OUT_PATH)
    print(DOCSS_OUT_PATH)


if __name__ == "__main__":
    main()
