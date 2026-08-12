# -*- coding: utf-8 -*-
# Run from the submission root: python code/analysis/run_cn_baselines.py

import os
import sys
import json
import pprint
from datetime import datetime
import argparse

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
EOH_DIR = os.path.join(PACKAGE_ROOT, "code", "eoh")
if EOH_DIR not in sys.path:
    sys.path.append(EOH_DIR)

from eoh.src.eoh.problems.optimization.cn.run import CriticalNode

pp = pprint.PrettyPrinter(indent=2, width=120)

# ---------- CoreHD ----------

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

# ---------- CI (ℓ = 3) ----------

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

# ---------- HDA（Highest Degree Attack）----------

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

# ---------- HCA（Highest Closeness Attack）----------

CODE_HCA = r"""
import math
import networkx as nx

_hca_cache = {}

def _pick_best_node(score_map):
    max_score = max(score_map.values())
    candidates = [node for node, score in score_map.items() if math.isclose(score, max_score, rel_tol=1e-12, abs_tol=1e-12)]
    return min(candidates)

def _hca_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        if G.number_of_nodes() == 1:
            order.extend(list(G.nodes()))
            break
        scores = nx.closeness_centrality(G)
        v = _pick_best_node(scores)
        order.append(v)
        G.remove_node(v)
    return order

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _hca_cache:
        _hca_cache[key] = _hca_order(graph)
    ranking = _hca_cache[key]
    unv_set = set(unvisited_nodes)
    for v in ranking:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""

# ---------- HPRA（Highest PageRank Attack）----------

CODE_HPRA = r"""
import math
import networkx as nx

_hpra_cache = {}

def _pick_best_node(score_map):
    max_score = max(score_map.values())
    candidates = [node for node, score in score_map.items() if math.isclose(score, max_score, rel_tol=1e-12, abs_tol=1e-12)]
    return min(candidates)

def _hpra_order(graph):
    G = graph.copy()
    order = []
    while G.number_of_nodes() > 0:
        if G.number_of_nodes() == 1:
            order.extend(list(G.nodes()))
            break
        scores = nx.pagerank(G)
        v = _pick_best_node(scores)
        order.append(v)
        G.remove_node(v)
    return order

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _hpra_cache:
        _hpra_cache[key] = _hpra_order(graph)
    ranking = _hpra_cache[key]
    unv_set = set(unvisited_nodes)
    for v in ranking:
        if v in unv_set:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_RatioCut = r"""
import math
import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover

_ratiocut_cache = {}

def _largest_cc_nodes(G):
    comps = list(nx.connected_components(G))
    if not comps:
        return []
    return list(max(comps, key=len))

def _partition_nodes(H):
    try:
        from networkx.algorithms.community import kernighan_lin_bisection
        A, B = kernighan_lin_bisection(H, max_iter=10, seed=0)
        return set(A), set(B)
    except Exception:
        nodes = list(H.nodes())
        if not nodes:
            return set(), set()
        nodes.sort(key=lambda n: (-H.degree(n), n))
        mid = len(nodes) // 2
        return set(nodes[:mid]), set(nodes[mid:])

def _cut_vertex_cover(H, A, B):
    cutG = nx.Graph()
    for u in A:
        for v in H.neighbors(u):
            if v in B:
                cutG.add_edge(u, v)
    if cutG.number_of_edges() == 0:
        return set()
    return set(min_weighted_vertex_cover(cutG))

def _dismantle_order(graph):
    G0 = graph
    N = G0.number_of_nodes()
    C = max(1, int(0.15 * N))
    G = G0.copy()
    order = []
    removed = set()
    while G.number_of_nodes() > 0:
        cc_nodes = _largest_cc_nodes(G)
        if len(cc_nodes) <= C:
            break
        H = G.subgraph(cc_nodes).copy()
        if H.number_of_nodes() <= 1:
            break
        A, B = _partition_nodes(H)
        cover = _cut_vertex_cover(H, A, B)
        if not cover:
            degrees = dict(H.degree())
            max_deg = max(degrees.values())
            candidates = [u for u, d in degrees.items() if d == max_deg]
            cover = {min(candidates)}
        for v in sorted(cover):
            if v in removed:
                continue
            removed.add(v)
            order.append(v)
        G.remove_nodes_from(cover)
    remaining = [n for n in G0.nodes() if n not in removed]
    remaining.sort(key=lambda n: (-G0.degree(n), n))
    return order + remaining

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _ratiocut_cache:
        _ratiocut_cache[key] = _dismantle_order(graph)
    ranking = _ratiocut_cache[key]
    unv = set(unvisited_nodes)
    for v in ranking:
        if v in unv:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_GND = r"""
import math
import networkx as nx
from networkx.algorithms.approximation import min_weighted_vertex_cover

_gnd_cache = {}

def _largest_cc_nodes(G):
    comps = list(nx.connected_components(G))
    if not comps:
        return []
    return list(max(comps, key=len))

def _partition_nodes(H):
    try:
        from networkx.algorithms.community import kernighan_lin_bisection
        A, B = kernighan_lin_bisection(H, max_iter=10, seed=0)
        return set(A), set(B)
    except Exception:
        nodes = list(H.nodes())
        if not nodes:
            return set(), set()
        nodes.sort(key=lambda n: (-H.degree(n), n))
        mid = len(nodes) // 2
        return set(nodes[:mid]), set(nodes[mid:])

def _cut_vertex_cover(H, A, B):
    cutG = nx.Graph()
    for u in A:
        for v in H.neighbors(u):
            if v in B:
                cutG.add_edge(u, v)
    if cutG.number_of_edges() == 0:
        return set()
    return set(min_weighted_vertex_cover(cutG))

def _max_cc_size(G):
    comps = list(nx.connected_components(G))
    return max((len(c) for c in comps), default=0)

def _reverse_greedy_component(G0, order, C):
    removed = set(order)
    H = G0.copy()
    H.remove_nodes_from(removed)
    for v in reversed(order):
        H.add_node(v)
        for u in G0.neighbors(v):
            if u in H:
                H.add_edge(v, u)
        if _max_cc_size(H) <= C:
            removed.remove(v)
        else:
            H.remove_node(v)
    return [v for v in order if v in removed]

def _dismantle_order(graph):
    G0 = graph
    N = G0.number_of_nodes()
    C = max(1, int(0.15 * N))
    G = G0.copy()
    order = []
    removed = set()
    while G.number_of_nodes() > 0:
        cc_nodes = _largest_cc_nodes(G)
        if len(cc_nodes) <= C:
            break
        H = G.subgraph(cc_nodes).copy()
        if H.number_of_nodes() <= 1:
            break
        A, B = _partition_nodes(H)
        cover = _cut_vertex_cover(H, A, B)
        if not cover:
            degrees = dict(H.degree())
            max_deg = max(degrees.values())
            candidates = [u for u, d in degrees.items() if d == max_deg]
            cover = {min(candidates)}
        for v in sorted(cover):
            if v in removed:
                continue
            removed.add(v)
            order.append(v)
        G.remove_nodes_from(cover)
    order = _reverse_greedy_component(G0, order, C)
    removed = set(order)
    remaining = [n for n in G0.nodes() if n not in removed]
    remaining.sort(key=lambda n: (-G0.degree(n), n))
    return order + remaining

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _gnd_cache:
        _gnd_cache[key] = _dismantle_order(graph)
    ranking = _gnd_cache[key]
    unv = set(unvisited_nodes)
    for v in ranking:
        if v in unv:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_MinSum = r"""
import networkx as nx

_minsum_cache = {}

def _is_forest(G):
    try:
        return nx.is_forest(G)
    except Exception:
        return len(nx.cycle_basis(G)) == 0

def _max_cc_size(G):
    comps = list(nx.connected_components(G))
    return max((len(c) for c in comps), default=0)

def _decycle_corehd(G0):
    G = G0.copy()
    removed = []
    while True:
        try:
            H = nx.k_core(G, k=2)
        except Exception:
            H = G
        if H.number_of_nodes() == 0:
            break
        degrees = dict(H.degree())
        max_deg = max(degrees.values())
        candidates = [u for u, d in degrees.items() if d == max_deg]
        v = min(candidates)
        removed.append(v)
        G.remove_node(v)
    return removed

def _reverse_greedy_forest(G0, removed_order):
    removed = set(removed_order)
    H = G0.copy()
    H.remove_nodes_from(removed)
    for v in reversed(removed_order):
        H.add_node(v)
        for u in G0.neighbors(v):
            if u in H:
                H.add_edge(v, u)
        if _is_forest(H):
            removed.remove(v)
        else:
            H.remove_node(v)
    return [v for v in removed_order if v in removed]

def _tree_break(G0, removed_set, C):
    G = G0.copy()
    G.remove_nodes_from(removed_set)
    extra = []
    while _max_cc_size(G) > C:
        comps = list(nx.connected_components(G))
        comp = max(comps, key=len)
        H = G.subgraph(comp)
        degrees = dict(H.degree())
        max_deg = max(degrees.values())
        candidates = [u for u, d in degrees.items() if d == max_deg]
        v = min(candidates)
        extra.append(v)
        G.remove_node(v)
    return extra

def _order(graph):
    G0 = graph
    N = G0.number_of_nodes()
    C = max(1, int(0.15 * N))
    decycle = _decycle_corehd(G0)
    decycle = _reverse_greedy_forest(G0, decycle)
    removed = set(decycle)
    extra = _tree_break(G0, removed, C)
    removed_order = decycle + extra
    removed_set = set(removed_order)
    remaining = [n for n in G0.nodes() if n not in removed_set]
    remaining.sort(key=lambda n: (-G0.degree(n), n))
    return removed_order + remaining

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _minsum_cache:
        _minsum_cache[key] = _order(graph)
    ranking = _minsum_cache[key]
    unv = set(unvisited_nodes)
    for v in ranking:
        if v in unv:
            return v
    return list(unvisited_nodes)[0]
"""

CODE_BPD = r"""
import networkx as nx

_bpd_cache = {}

def _is_forest(G):
    try:
        return nx.is_forest(G)
    except Exception:
        return len(nx.cycle_basis(G)) == 0

def _max_cc_size(G):
    comps = list(nx.connected_components(G))
    return max((len(c) for c in comps), default=0)

def _decycle_betweenness(G0):
    G = G0.copy()
    removed = []
    while True:
        try:
            H = nx.k_core(G, k=2)
        except Exception:
            H = G
        if H.number_of_nodes() == 0:
            break
        if H.number_of_nodes() == 1:
            removed.extend(list(H.nodes()))
            break
        scores = nx.betweenness_centrality(H, normalized=False)
        max_score = max(scores.values())
        candidates = [u for u, s in scores.items() if s == max_score]
        v = min(candidates)
        removed.append(v)
        G.remove_node(v)
    return removed

def _reverse_greedy_forest(G0, removed_order):
    removed = set(removed_order)
    H = G0.copy()
    H.remove_nodes_from(removed)
    for v in reversed(removed_order):
        H.add_node(v)
        for u in G0.neighbors(v):
            if u in H:
                H.add_edge(v, u)
        if _is_forest(H):
            removed.remove(v)
        else:
            H.remove_node(v)
    return [v for v in removed_order if v in removed]

def _tree_break(G0, removed_set, C):
    G = G0.copy()
    G.remove_nodes_from(removed_set)
    extra = []
    while _max_cc_size(G) > C:
        comps = list(nx.connected_components(G))
        comp = max(comps, key=len)
        H = G.subgraph(comp)
        degrees = dict(H.degree())
        max_deg = max(degrees.values())
        candidates = [u for u, d in degrees.items() if d == max_deg]
        v = min(candidates)
        extra.append(v)
        G.remove_node(v)
    return extra

def _order(graph):
    G0 = graph
    N = G0.number_of_nodes()
    C = max(1, int(0.15 * N))
    decycle = _decycle_betweenness(G0)
    decycle = _reverse_greedy_forest(G0, decycle)
    removed = set(decycle)
    extra = _tree_break(G0, removed, C)
    removed_order = decycle + extra
    removed_set = set(removed_order)
    remaining = [n for n in G0.nodes() if n not in removed_set]
    remaining.sort(key=lambda n: (-G0.degree(n), n))
    return removed_order + remaining

def select_next_node(graph, unvisited_nodes):
    key = id(graph)
    if key not in _bpd_cache:
        _bpd_cache[key] = _order(graph)
    ranking = _bpd_cache[key]
    unv = set(unvisited_nodes)
    for v in ranking:
        if v in unv:
            return v
    return list(unvisited_nodes)[0]
"""

BASELINE_CODE_MAP = {
    "CoreHD": CODE_COREHD,
    "CI(l=3)": CODE_CI_L3,
    "HDA": CODE_HDA,
    "HBA": CODE_HBA,
    "HCA": CODE_HCA,
    "HPRA": CODE_HPRA,
    "RatioCut": CODE_RatioCut,
    "GND": CODE_GND,
    "MinSum": CODE_MinSum,
    "BPD": CODE_BPD,
}

DEFAULT_ALGOS = ["CoreHD", "CI(l=3)", "HDA", "HBA", "HCA", "HPRA"]

# ---------- 统一运行入口 ----------

def run_baseline(dataset_name, code_string, name, out_dir):
    print(f"\n=== Running baseline: {name} on dataset {dataset_name} ===")
    cn = CriticalNode(dataset_name=dataset_name, use_precompute=False)
    obj, metrics = cn.evaluate(code_string)
    print(f"objective = {obj}")
    pp.pprint(metrics)

    result = {
        "algorithm": name,
        "dataset": dataset_name,
        "objective": obj,
        "metrics": metrics,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to: {out_path}")

    return obj, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CN baselines")
    parser.add_argument(
        "--dataset",
        type=str,
        default="HI-II-14_sampled",
        help="Dataset name, e.g. HI-II-14_sampled / Enron_sample",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(PACKAGE_ROOT, "evolve_experiment", "processed", "baseline_results"),
        help="Directory to save JSON result files",
    )
    parser.add_argument(
        "--algos",
        nargs="+",
        default=None,
        help="Algorithms to run, e.g. CoreHD HDA HBA HCA HPRA 'CI(l=3)'",
    )
    args = parser.parse_args()

    dataset = args.dataset
    out_dir = args.out_dir

    if args.algos:
        baselines = [(name, BASELINE_CODE_MAP[name]) for name in args.algos if name in BASELINE_CODE_MAP]
    else:
        baselines = [(name, BASELINE_CODE_MAP[name]) for name in DEFAULT_ALGOS]

    for name, code in baselines:
        run_baseline(dataset, code, name, out_dir)
