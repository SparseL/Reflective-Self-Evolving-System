import os
import sys
import json
from datetime import datetime
import argparse
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
EOH_DIR = os.path.join(ROOT_DIR, "code", "eoh")
if EOH_DIR not in sys.path:
    sys.path.append(EOH_DIR)

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode
from run_cn_baselines import (
    CODE_HBA,
    CODE_HCA,
    CODE_HPRA,
    CODE_RatioCut,
    CODE_GND,
    CODE_MinSum,
    CODE_BPD,
    DEFAULT_ALGOS,
)


CODE_COREHD = r"""
import networkx as nx

_corehd_cache = {}

def _corehd_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        try:
            H = nx.k_core(G, k=2)
        except nx.NetworkXError:
            H = G
        if H.number_of_nodes() == 0:
            degrees = dict(G.degree())
            if not degrees:
                break
            for node, _ in sorted(degrees.items(), key=lambda x: (-x[1], x[0])):
                order.append(node)
            break
        degrees = dict(H.degree())
        max_deg = max(degrees.values())
        candidates = [u for u, d in degrees.items() if d == max_deg]
        v = min(candidates)
        order.append(v)
        G.remove_node(v)
    return order

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _corehd_cache:
        _corehd_cache[key] = _corehd_order(graph)
    ranking = _corehd_cache[key]
    unv_set = set(unvisited_nodes)
    for v in ranking:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""


CODE_CI_L3 = r"""
import networkx as nx

_ci_cache = {}

def _ci_scores(G, l):
    scores = {}
    deg = dict(G.degree())
    for i in G.nodes():
        if deg[i] <= 1:
            scores[i] = 0.0
            continue
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=l)
        boundary = [u for u, dist in lengths.items() if dist == l]
        s = sum(max(deg[u] - 1, 0) for u in boundary)
        scores[i] = max(deg[i] - 1, 0) * s
    return scores

def _ci_order(graph, l=3):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        if G.number_of_nodes() == 1:
            order.extend(list(G.nodes()))
            break
        scores = _ci_scores(G, l)
        max_ci = max(scores.values())
        candidates = [u for u, v in scores.items() if v == max_ci]
        v = min(candidates)
        order.append(v)
        G.remove_node(v)
    return order

def select_next_node(graph, unvisited_nodes):
    key = (id(graph), 3)
    if key not in _ci_cache:
        _ci_cache[key] = _ci_order(graph, l=3)
    ranking = _ci_cache[key]
    unv_set = set(unvisited_nodes)
    for v in ranking:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""


CODE_HDA = r"""
import networkx as nx

_hda_cache = {}

def _hda_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        if not degrees:
            break
        max_deg = max(degrees.values())
        candidates = [node for node, d in degrees.items() if d == max_deg]
        v = min(candidates)
        order.append(v)
        G.remove_node(v)
    return order

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _hda_cache:
        _hda_cache[key] = _hda_order(graph)
    ranking = _hda_cache[key]
    unv_set = set(unvisited_nodes)
    for v in ranking:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""


def _summarize(detailed_results):
    objs = [r.get("objective") for r in detailed_results if isinstance(r, dict) and r.get("objective") is not None]
    ts = [r.get("metrics", {}).get("time_select") for r in detailed_results if isinstance(r, dict)]
    ta = [r.get("metrics", {}).get("time_anc") for r in detailed_results if isinstance(r, dict)]
    ts = [x for x in ts if x is not None]
    ta = [x for x in ta if x is not None]
    return {
        "objective": float(np.mean(objs)) if objs else None,
        "objective_std": float(np.std(objs)) if objs else None,
        "time_select_avg": float(np.mean(ts)) if ts else None,
        "time_anc_avg": float(np.mean(ta)) if ta else None,
    }


def run_one_dataset(dataset_name: str, num_instance: int, out_root: str, size: int):
    cn = CriticalNode(num_instance=num_instance, dataset_name=dataset_name, use_precompute=False)
    baselines = [
        ("CoreHD", CODE_COREHD),
        ("CI(l=3)", CODE_CI_L3),
        ("HDA", CODE_HDA),
        ("HBA", CODE_HBA),
        ("HCA", CODE_HCA),
        ("HPRA", CODE_HPRA),
        ("RatioCut", CODE_RatioCut),
        ("GND", CODE_GND),
        ("MinSum", CODE_MinSum),
        ("BPD", CODE_BPD),
    ]
    if getattr(run_one_dataset, "selected_algos", None):
        selected = set(run_one_dataset.selected_algos)
        baselines = [(name, code) for name, code in baselines if name in selected]

    for algo_name, code in baselines:
        obj, metrics = cn.evaluate(code)
        detailed = metrics.get("detailed_results") if isinstance(metrics, dict) else None
        if not isinstance(detailed, list) or len(detailed) == 0:
            detailed = []
        summary = _summarize(detailed)
        payload = {
            "algorithm": algo_name,
            "dataset": dataset_name,
            "detailed_results": detailed,
            "summary_average": summary,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        save_dir = os.path.join(out_root, dataset_name.replace("synthetic_", ""), f"{size}_detailed")
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"{algo_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--num_instance", type=int, default=5)
    parser.add_argument("--out_root", type=str, default=os.path.join(ROOT_DIR, "evolve_experiment", "processed", "baseline_results", "synthetic"))
    parser.add_argument("--datasets", nargs="*", default=["er", "ws", "sbm", "uniform_cost"])
    parser.add_argument("--algos", nargs="*", default=None)
    args = parser.parse_args()

    run_one_dataset.selected_algos = args.algos or DEFAULT_ALGOS

    for ds in args.datasets:
        dataset_name = f"synthetic_{ds}_{args.size}"
        run_one_dataset(dataset_name, args.num_instance, args.out_root, args.size)


if __name__ == "__main__":
    main()
