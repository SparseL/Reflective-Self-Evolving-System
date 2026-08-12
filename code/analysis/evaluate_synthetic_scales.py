import os
import sys
import json
import re
from datetime import datetime
import argparse
import networkx as nx
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
EOH_DIR = os.path.join(ROOT_DIR, "code", "eoh")
BUNDLE_DIR = ROOT_DIR
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
def _corehd_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        try:
            H = nx.k_core(G, k=2)
        except: H = G
        if H.number_of_nodes() == 0:
            degrees = dict(G.degree())
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
    if not hasattr(select_next_node, 'cache'):
        select_next_node.cache = _corehd_order(graph)
    unv_set = set(unvisited_nodes)
    for v in select_next_node.cache:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_CI_L3 = r"""
import networkx as nx
def _ci_scores(G, l):
    scores = {}
    deg = dict(G.degree())
    for i in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, i, cutoff=l)
        boundary = [u for u, dist in lengths.items() if dist == l]
        s = sum(max(deg[u] - 1, 0) for u in boundary)
        scores[i] = max(deg[i] - 1, 0) * s
    return scores
def _ci_order(graph, l=3):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        scores = _ci_scores(G, l)
        max_ci = max(scores.values())
        v = min([u for u, v in scores.items() if v == max_ci])
        order.append(v)
        G.remove_node(v)
    return order
def select_next_node(graph, unvisited_nodes):
    if not hasattr(select_next_node, 'cache'):
        select_next_node.cache = _ci_order(graph, l=3)
    unv_set = set(unvisited_nodes)
    for v in select_next_node.cache:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_HDA = r"""
import networkx as nx
def _hda_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        max_deg = max(degrees.values())
        v = min([node for node, d in degrees.items() if d == max_deg])
        order.append(v)
        G.remove_node(v)
    return order
def select_next_node(graph, unvisited_nodes):
    if not hasattr(select_next_node, 'cache'):
        select_next_node.cache = _hda_order(graph)
    unv_set = set(unvisited_nodes)
    for v in select_next_node.cache:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""

class SyntheticCriticalNode(CriticalNode):
    def __init__(self, dataset_path, use_precompute=True):
        self.use_precompute = use_precompute
        self.instances = []
        self.instance_names = []
        if os.path.isdir(dataset_path):
            def _g_index(name):
                m = re.match(r"^g_(\d+)", name)
                if not m:
                    return None
                return int(m.group(1))

            files = [f for f in os.listdir(dataset_path) if f.startswith("g_")]
            files = [f for f in files if _g_index(f) is not None]
            files.sort(key=lambda x: (_g_index(x), x))

            seen = set()
            deduped = []
            for f in files:
                idx = _g_index(f)
                if idx in seen:
                    continue
                seen.add(idx)
                deduped.append(f)
            files = deduped
            for f in files:
                fpath = os.path.join(dataset_path, f)
                G = nx.read_gml(fpath)
                mapping = {node: i for i, node in enumerate(G.nodes())}
                G = nx.relabel_nodes(G, mapping)
                self.instances.append(G)
                self.instance_names.append(f)
        if self.use_precompute:
            for instance in self.instances:
                self.precompute_features(instance)

def load_eoh_best_code(summary_csv_path, ds_key):
    if not os.path.exists(summary_csv_path):
        return None, {}
    code_path = None
    meta = {}
    with open(summary_csv_path, "r", encoding="utf-8-sig") as f:
        header = None
        for line in f:
            line = line.strip()
            if not line:
                continue
            cols = line.split(",")
            if header is None:
                header = cols
                continue
            row = dict(zip(header, cols))
            dataset = row.get("dataset")
            if dataset == ds_key:
                code_path = row.get("best_json")
                meta = {
                    "group": row.get("group"),
                    "model": row.get("model"),
                    "run_name": row.get("run_name"),
                    "best_generation": row.get("best_generation"),
                    "operator_at_best": row.get("operator_at_best"),
                }
                break
    if not code_path or not os.path.exists(code_path):
        return None, {}
    with open(code_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        data = data[0]
    code = data.get("code")
    return code, meta

def evaluate_on_instances(instances, names, code_str, scn):
    results = []
    for i in range(min(len(instances), len(names))):
        scn.instances = [instances[i]]
        obj, metrics = scn.evaluate(code_str)
        results.append({"instance": names[i], "objective": obj, "metrics": metrics})
    return results

def avg(vals):
    arr = [v for v in vals if isinstance(v, (int, float))]
    if not arr:
        return None
    return float(np.mean(arr))

def summarize(detailed):
    tsel = [r.get("metrics", {}).get("time_select") for r in detailed]
    tanc = [r.get("metrics", {}).get("time_anc") for r in detailed]
    obj = [r.get("objective") for r in detailed]
    fht50 = [r.get("metrics", {}).get("fht_50") for r in detailed]
    fht10 = [r.get("metrics", {}).get("fht_10") for r in detailed]
    lcc = [r.get("metrics", {}).get("lcc_at_k_frac") for r in detailed]
    anc_k = [r.get("metrics", {}).get("anc_prefix_k") for r in detailed]
    k = [r.get("metrics", {}).get("k") for r in detailed]
    return {
        "objective": avg(obj),
        "time_select_avg": avg(tsel),
        "time_anc_avg": avg(tanc),
        "fht_50_avg": avg(fht50),
        "fht_10_avg": avg(fht10),
        "lcc_at_k_frac_avg": avg(lcc),
        "anc_prefix_k_avg": avg(anc_k),
        "k_avg": avg(k),
        "n_instances": len(detailed),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--types", nargs="+", default=["sbm", "er", "ws", "uniform_cost"])
    p.add_argument("--scales", nargs="+", default=["400-500", "1000", "3000", "5000"])
    p.add_argument("--instances", type=int, default=5)
    p.add_argument("--summary_csv", type=str, default=os.path.join(BUNDLE_DIR, "evolve_experiment", "processed", "analysis_tables_synthetic_runs", "synthetic_runs_summary.csv"))
    p.add_argument("--dataset_root", type=str, default=os.path.join(BUNDLE_DIR, "dataset", "synthetic"))
    p.add_argument("--out_root", type=str, default=os.path.join(BUNDLE_DIR, "evolve_experiment", "processed", "baseline_results", "synthetic"))
    p.add_argument("--algos", nargs="+", default=None)
    args = p.parse_args()

    algo_list = [
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
    os.makedirs(args.out_root, exist_ok=True)

    for ds_type in args.types:
        eoh_key = f"synthetic_{ds_type}_1000"
        eoh_code, eoh_meta = load_eoh_best_code(args.summary_csv, eoh_key)
        if eoh_code:
            algo_list_with_eoh = algo_list + [("EoH-Best-from-1000", eoh_code)]
        else:
            algo_list_with_eoh = algo_list
        selected = set(args.algos or DEFAULT_ALGOS)
        algo_list_with_eoh = [(name, code) for name, code in algo_list_with_eoh if name in selected]
        for scale in args.scales:
            dataset_path = os.path.join(args.dataset_root, ds_type, str(scale))
            if not os.path.isdir(dataset_path):
                continue
            scn = SyntheticCriticalNode(dataset_path, use_precompute=True)
            if not scn.instances:
                continue
            pick_n = min(len(scn.instances), args.instances)
            instances = scn.instances[:pick_n]
            names = scn.instance_names[:pick_n]
            out_dir = os.path.join(args.out_root, f"{ds_type}_{scale}_detailed")
            os.makedirs(out_dir, exist_ok=True)
            for algo_name, code in algo_list_with_eoh:
                detailed = evaluate_on_instances(instances, names, code, scn)
                summary = summarize(detailed)
                payload = {
                    "algorithm": algo_name if algo_name != "EoH-Best-from-1000" else f"{algo_name} ({eoh_meta.get('model','')})",
                    "dataset": f"synthetic_{ds_type}_{scale}",
                    "detailed_results": detailed,
                    "summary_average": summary,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                if algo_name == "EoH-Best-from-1000":
                    payload["eoh_origin"] = {
                        "from_dataset": eoh_key,
                        "meta": eoh_meta,
                    }
                save_path = os.path.join(out_dir, f"{algo_name}.json")
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                print(save_path)

if __name__ == "__main__":
    main()
