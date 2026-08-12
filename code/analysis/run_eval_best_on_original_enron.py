import argparse
import hashlib
import json
import os
import sys
from datetime import datetime


def add_eoh_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
    eoh_dir = os.path.join(repo_root, "code", "eoh")
    if eoh_dir not in sys.path:
        sys.path.append(eoh_dir)
    return repo_root


REPO_ROOT = add_eoh_to_path()
BUNDLE_ROOT = REPO_ROOT

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode


DEFAULT_SOURCE_JSON = (
    os.path.join(BUNDLE_ROOT, "evolve_experiment", "evolution", "real", "Enron_sampled_post_with_precompute_new",
                 "results", "pops_best", "Enron_sampled", "population_generation_50.json")
)
DEFAULT_OUTPUT_JSON = (
    os.path.join(BUNDLE_ROOT, "evolve_experiment", "transfer", "best_algo_on_original_Enron.json")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an EOH best algorithm on the original Enron dataset."
    )
    parser.add_argument(
        "--source-json",
        default=DEFAULT_SOURCE_JSON,
        help="Path to the sampled-dataset best algorithm JSON.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help="Path to save the evaluation result JSON.",
    )
    parser.add_argument(
        "--dataset",
        default="Enron",
        help="Target dataset name for CriticalNode.",
    )
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="Disable precomputed graph features for the target dataset.",
    )
    return parser.parse_args()


def load_algorithm_payload(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    code = payload.get("code", "")
    if not code:
        raise ValueError(f"No 'code' found in source JSON: {json_path}")
    return payload, code


def build_success_payload(source_payload, source_json, dataset_name, objective, metrics):
    code = source_payload["code"]
    time_select = float(metrics.get("time_select", 0.0))
    time_anc = float(metrics.get("time_anc", 0.0))
    unique_id = source_payload.get("unique_id") or hashlib.md5(code.encode("utf-8")).hexdigest()

    return {
        "algorithm": source_payload.get("algorithm", ""),
        "code": code,
        "objective": float(objective),
        "other_inf": metrics,
        "problem": f"CriticalNode-{dataset_name}",
        "time": time_select + time_anc,
        "time_select": time_select,
        "time_anc": time_anc,
        "unique_id": unique_id,
        "dataset": dataset_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "source_json": source_json,
    }


def build_failure_payload(source_payload, source_json, dataset_name, error):
    code = source_payload.get("code", "")
    unique_id = source_payload.get("unique_id") or (
        hashlib.md5(code.encode("utf-8")).hexdigest() if code else None
    )

    return {
        "algorithm": source_payload.get("algorithm", ""),
        "code": code,
        "problem": f"CriticalNode-{dataset_name}",
        "unique_id": unique_id,
        "dataset": dataset_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "source_json": source_json,
        "status": "failed",
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
    }


def main():
    args = parse_args()
    source_json = os.path.abspath(args.source_json)
    output_json = os.path.abspath(args.output_json)
    use_precompute = not args.no_precompute

    print(f"Loading source algorithm: {source_json}")
    source_payload, code = load_algorithm_payload(source_json)

    print(
        f"Evaluating on dataset='{args.dataset}' with use_precompute={use_precompute}..."
    )
    problem = CriticalNode(dataset_name=args.dataset, use_precompute=use_precompute)
    result = problem.evaluate(code)

    if isinstance(result, Exception):
        output_payload = build_failure_payload(
            source_payload, source_json, args.dataset, result
        )
        status_message = f"Evaluation failed: {result}"
    else:
        objective, metrics = result
        output_payload = build_success_payload(
            source_payload, source_json, args.dataset, objective, metrics
        )
        status_message = f"Evaluation succeeded. objective={objective}"

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print(status_message)
    print(f"Saved result to: {output_json}")

    if isinstance(result, Exception):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
