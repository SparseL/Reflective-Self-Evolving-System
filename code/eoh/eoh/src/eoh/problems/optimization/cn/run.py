import pickle
from abc import abstractmethod, ABC

import numpy as np
import importlib
# from .get_instance import GetData
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.abspath(os.path.join(current_dir, "../../../../../../"))
sys.path.append(new_path)

from eoh.src.eoh.problems.optimization.cn.prompts import GetPrompts
import types
import warnings
import sys
import networkx as nx
import re
import hashlib


def get_anc(graph, critical_nodes):
    def calc_anc(graph):
        connected_components_count = np.array([len(x) for x in list(nx.connected_components(graph))])
        return (connected_components_count * (connected_components_count - 1) / 2).sum()

    anc = 0
    instance_copy = graph.copy()
    for j in range(len(critical_nodes)):
        # 从instance中移除critical_nodes[j]
        instance_copy.remove_node(critical_nodes[j])
        # 获取剩余的联通分量
        anc += calc_anc(instance_copy)

    anc /= len(critical_nodes) * calc_anc(graph)
    return anc


class CriticalNode:
    PRECOMPUTE_CACHE_VERSION = 1

    def precompute_features(self, graph):
        deg = dict(graph.degree())
        nx.set_node_attributes(graph, deg, 'degree')
        clust = nx.clustering(graph)
        nx.set_node_attributes(graph, clust, 'clustering')
        core = nx.core_number(graph)
        nx.set_node_attributes(graph, core, 'core_number')
        bc = nx.betweenness_centrality(graph, normalized=False)
        nx.set_node_attributes(graph, bc, 'betweenness')
        eig = nx.eigenvector_centrality(graph, max_iter=1000, tol=1e-06)
        nx.set_node_attributes(graph, eig, 'eigenvector')
        n = graph.number_of_nodes()
        m = graph.number_of_edges()
        avg_deg = (2.0 * m / n) if n > 0 else 0.0
        deg_arr = np.array(list(deg.values())) if len(deg) > 0 else np.array([0.0])
        deg_var = float(deg_arr.var()) if deg_arr.size > 0 else 0.0
        comps = list(nx.connected_components(graph))
        lcc = max(comps, key=len) if len(comps) > 0 else set()
        H = graph.subgraph(lcc).copy() if len(lcc) > 0 else graph.copy()
        try:
            apl = nx.average_shortest_path_length(H) if H.number_of_nodes() > 1 else 0.0
        except Exception:
            apl = 0.0
        try:
            diam = nx.diameter(H) if H.number_of_nodes() > 1 else 0
        except Exception:
            diam = 0
        max_core_k = max(core.values()) if len(core) > 0 else 0
        bc_vals = np.array(list(bc.values())) if len(bc) > 0 else np.array([0.0])
        thr = np.quantile(bc_vals, 0.9) if bc_vals.size > 0 else 0.0
        bridge_ratio = float((bc_vals > thr).sum()) / float(n) if n > 0 else 0.0
        graph.graph['precomputed'] = {
            'avg_degree': avg_deg,
            'degree_variance': float(deg_var),
            'apl_lcc': float(apl),
            'diameter_lcc': int(diam),
            'max_core_k': int(max_core_k),
            'bridge_ratio_high_bc': float(bridge_ratio)
        }

    def _dataset_base_dir(self):
        current = os.path.dirname(os.path.abspath(__file__))
        for _ in range(12):
            if os.path.isdir(os.path.join(current, "dataset", "real")):
                return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _cache_root(self):
        return os.path.join(self._dataset_base_dir(), "dataset", "cache", "precomputed")

    def _source_signature(self, source_path, graph):
        stat = os.stat(source_path) if source_path and os.path.exists(source_path) else None
        return {
            "cache_version": self.PRECOMPUTE_CACHE_VERSION,
            "source_path": os.path.abspath(source_path) if source_path else None,
            "source_size": int(stat.st_size) if stat else None,
            "source_mtime_ns": int(stat.st_mtime_ns) if stat else None,
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        }

    def _precompute_cache_path(self, dataset_name, instance_name, source_path):
        key_src = os.path.abspath(source_path) if source_path else f"{dataset_name}:{instance_name}"
        digest = hashlib.md5(key_src.encode("utf-8")).hexdigest()[:12]
        safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name))
        safe_instance = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(instance_name))
        return os.path.join(
            self._cache_root(),
            safe_dataset,
            f"{safe_instance}__{digest}.pkl",
        )

    def _load_precompute_cache(self, graph, dataset_name, instance_name, source_path):
        cache_path = self._precompute_cache_path(dataset_name, instance_name, source_path)
        if not os.path.isfile(cache_path):
            return False
        expected = self._source_signature(source_path, graph)
        try:
            with open(cache_path, "rb") as f:
                payload = pickle.load(f)
        except Exception as exc:
            print(f"Precompute cache unreadable, recomputing: {cache_path} ({exc})")
            return False

        if payload.get("signature") != expected:
            print(f"Precompute cache stale, recomputing: {cache_path}")
            return False

        node_attrs = payload.get("node_attrs", {})
        for attr_name, values in node_attrs.items():
            nx.set_node_attributes(graph, values, attr_name)
        graph.graph["precomputed"] = payload.get("graph_precomputed", {})
        print(f"Loaded precompute cache: {cache_path}")
        return True

    def _save_precompute_cache(self, graph, dataset_name, instance_name, source_path):
        cache_path = self._precompute_cache_path(dataset_name, instance_name, source_path)
        payload = {
            "signature": self._source_signature(source_path, graph),
            "node_attrs": {
                "degree": nx.get_node_attributes(graph, "degree"),
                "clustering": nx.get_node_attributes(graph, "clustering"),
                "core_number": nx.get_node_attributes(graph, "core_number"),
                "betweenness": nx.get_node_attributes(graph, "betweenness"),
                "eigenvector": nx.get_node_attributes(graph, "eigenvector"),
            },
            "graph_precomputed": graph.graph.get("precomputed", {}),
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
        print(f"Saved precompute cache: {cache_path}")

    def precompute_features_cached(self, graph, dataset_name, instance_name, source_path):
        if self._load_precompute_cache(graph, dataset_name, instance_name, source_path):
            return
        self.precompute_features(graph)
        self._save_precompute_cache(graph, dataset_name, instance_name, source_path)

    def read_crime(self, dataset_name):
        G = nx.Graph()
        path = os.path.join(self._dataset_base_dir(), f'dataset/real/{dataset_name}.txt')
        # print(path)
        with open(path, "r") as f:
            for line in f:
                # 将每行的数据分割为节点1、节点2以及可能的属性
                parts = line.strip().split(' ')

                # 提取节点1和节点2（忽略属性部分 '{}', 因为它是空的）
                node1 = int(parts[0])  # 假设节点是整数类型
                node2 = int(parts[1])

                # 将边添加到图中
                G.add_edge(node1, node2)
        print(f"{dataset_name}: ", len(G.nodes), len(G.edges))

        return G

    def read_synthetic(self, dataset_name, num_instance):
        parts = dataset_name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Invalid synthetic dataset_name: {dataset_name}")
        synth_type = "_".join(parts[1:-1])
        if synth_type == "uniform":
            synth_type = "uniform_cost"
        size_tag = parts[-1]
        if re.fullmatch(r"\d+", size_tag):
            size_dir = str(int(size_tag))
        elif re.fullmatch(r"\d+-\d+", size_tag):
            size_dir = size_tag
        else:
            raise ValueError(f"Invalid synthetic dataset size/range in dataset_name: {dataset_name}")

        base_dir = os.path.join(
            self._dataset_base_dir(),
            f"dataset/synthetic/{synth_type}/{size_dir}",
        )
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(f"Synthetic dataset directory not found: {base_dir}")

        files = [f for f in os.listdir(base_dir) if f.startswith("g_")]

        def _key(x):
            try:
                return int(x.split("_")[1])
            except Exception:
                return 10**9

        files.sort(key=_key)
        if num_instance is not None and int(num_instance) > 0:
            files = files[: int(num_instance)]

        instances = []
        names = []
        for f in files:
            fpath = os.path.join(base_dir, f)
            G = nx.read_gml(fpath)
            mapping = {node: i for i, node in enumerate(G.nodes())}
            G = nx.relabel_nodes(G, mapping)
            instances.append(G)
            names.append(f)
            self.instance_sources.append(fpath)

        self.instance_names = names
        if instances:
            print(f"{dataset_name}: ", len(instances), "instances | nodes:", instances[0].number_of_nodes(), "edges:", instances[0].number_of_edges())
        else:
            print(f"{dataset_name}: 0 instances")
        return instances
    
    def __init__(self, num_instance=1, dataset_name='Crime', use_precompute=True, init_prompt_profile='standard', aware_prompt_profile='none'):
        self.use_precompute = use_precompute
        self.prompts = GetPrompts(use_precompute, init_prompt_profile=init_prompt_profile, aware_prompt_profile=aware_prompt_profile)
        self.instance_names = []
        self.instance_sources = []

        if isinstance(dataset_name, str) and dataset_name.startswith("synthetic_"):
            self.instances = self.read_synthetic(dataset_name, num_instance)
        else:
            source_path = os.path.join(self._dataset_base_dir(), f"dataset/real/{dataset_name}.txt")
            self.instances = [self.read_crime(dataset_name)]
            self.instance_names = [dataset_name]
            self.instance_sources = [source_path]
        if self.use_precompute:
            print("Precomputing features...")
            for idx, instance in enumerate(self.instances):
                instance_name = self.instance_names[idx] if idx < len(self.instance_names) else f"instance_{idx}"
                source_path = self.instance_sources[idx] if idx < len(self.instance_sources) else None
                self.precompute_features_cached(instance, dataset_name, instance_name, source_path)
            print("Precomputing finished.")
        # if -1 == num_instance:
        #   n, m = 50, 4
        #   self.instances = [nx.barabasi_albert_graph(n, m) for _ in range(50)]
        #   # pickle.dump(self.instances, open("./train_instances.pkl", "wb"))
        #
        # self.instances = pickle.load(open("./train_instances.pkl", "rb"))
        # self.instances = self.instances[:num_instance]

    def run(self, heuristic_module, details=False):
        anc = np.zeros(len(self.instances))
        critical_nodes_list = []
        time_select = 0.0
        time_anc = 0.0
        prefix_vals = []
        fht50_list = []
        fht10_list = []
        lcck_list = []
        detailed_results = []
        for i, instance in enumerate(self.instances):
            # if self.use_precompute:
            #     self.precompute_features(instance)
            t_sel = __import__('time').time()
            critical_nodes = []
            nodes_idx = np.arange(instance.number_of_nodes())
            node_mask = np.ones(instance.number_of_nodes(), dtype=bool)
            while len(critical_nodes) < instance.number_of_nodes():
                try:
                    next_node = heuristic_module.select_next_node(instance, nodes_idx[node_mask])
                except Exception:
                    deg_map = dict(instance.degree())
                    candidates = list(nodes_idx[node_mask])
                    if len(candidates) == 0:
                        break
                    next_node = max(candidates, key=lambda n: (deg_map.get(int(n), 0), -int(n)))
                critical_nodes.append(next_node)
                node_mask[next_node] = False
            t_sel_i = __import__('time').time() - t_sel
            time_select += t_sel_i

            t_anc = __import__('time').time()
            sizes0 = np.array([len(x) for x in list(nx.connected_components(instance))])
            cp0 = (sizes0 * (sizes0 - 1) / 2).sum() if sizes0.size > 0 else 0.0

            instance_copy = instance.copy()
            cp_list = []
            N = instance.number_of_nodes()
            k = max(1, int(0.15 * N))
            lcc_at_k = None
            hit50 = None
            hit10 = None
            for j, node in enumerate(critical_nodes):
                instance_copy.remove_node(node)
                comps = list(nx.connected_components(instance_copy))
                sizes = np.array([len(x) for x in comps])
                cp_j = (sizes * (sizes - 1) / 2).sum() if sizes.size > 0 else 0.0
                cp_list.append(cp_j)
                if lcc_at_k is None and (j + 1) == k:
                    lcc_at_k = (sizes.max() if sizes.size > 0 else 0)
                ratio = (cp_j / cp0) if cp0 > 0 else 0.0
                if hit50 is None and ratio <= 0.5:
                    hit50 = j + 1
                if hit10 is None and ratio <= 0.1:
                    hit10 = j + 1

            anc[i] = (np.sum(cp_list) / (len(critical_nodes) * cp0)) if cp0 > 0 else 0.0
            anc_prefix_k_i = (np.sum(cp_list[:k]) / (k * cp0)) if (cp0 > 0 and k > 0) else 0.0
            fht50_i = hit50 if hit50 is not None else N
            fht10_i = hit10 if hit10 is not None else N
            lcck_i = (lcc_at_k / N) if (lcc_at_k is not None and N > 0) else 0.0
            t_anc_i = __import__('time').time() - t_anc
            time_anc += t_anc_i

            prefix_vals.append(anc_prefix_k_i)
            fht50_list.append(fht50_i)
            fht10_list.append(fht10_i)
            lcck_list.append(lcck_i)

            detailed_results.append({
                "instance": self.instance_names[i] if i < len(self.instance_names) else f"g_{i}",
                "objective": float(anc[i]),
                "metrics": {
                    "time_select": float(t_sel_i),
                    "time_anc": float(t_anc_i),
                    "anc_prefix_k": float(anc_prefix_k_i),
                    "fht_50": float(fht50_i),
                    "fht_10": float(fht10_i),
                    "lcc_at_k_frac": float(lcck_i),
                    "k": int(k),
                }
            })

            if details:
                critical_nodes_list.append(critical_nodes)
        if details:
            return anc, critical_nodes_list
        return anc.mean(), {"time_select": time_select, "time_anc": time_anc, "anc_prefix_k": float(np.mean(prefix_vals) if len(prefix_vals) > 0 else 0.0), "fht_50": float(np.mean(fht50_list) if len(fht50_list) > 0 else 0.0), "fht_10": float(np.mean(fht10_list) if len(fht10_list) > 0 else 0.0), "lcc_at_k_frac": float(np.mean(lcck_list) if len(lcck_list) > 0 else 0.0), "k": k, "detailed_results": detailed_results}

    def evaluate(self, code_string, details=False):

        try:
            # Suppress warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # Create a new module object
                heuristic_module = types.ModuleType("heuristic_module")

                # Execute the code string in the new module's namespace
                exec(code_string, heuristic_module.__dict__)

                # Add the module to sys.modules so it can be imported
                sys.modules[heuristic_module.__name__] = heuristic_module

                fitness = self.run(heuristic_module, details=details)
                return fitness
        except Exception as e:
            return e


"""paste your code here"""
code_snaps = """
def select_next_node(graph, unvisited_nodes):
  max_unvisited_neighbors = -1
  next_node = None
  for node in unvisited_nodes:
    unvisited_neighbors = [n for n in graph.neighbors(node) if n in unvisited_nodes]
    if len(unvisited_neighbors) > max_unvisited_neighbors:
      max_unvisited_neighbors = len(unvisited_neighbors)
      next_node = node
  return next_node
"""

if __name__ == "__main__":
    critical_node = CriticalNode(num_instance=8, dataset_name='Crime') # Crime  Digg  Enron  Epinions  Facebook  Flickr  Cnutella31  HI-II-14  Youtube
    print(critical_node.evaluate(code_snaps, details=True))
