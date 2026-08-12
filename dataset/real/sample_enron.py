import argparse
import os
import random
import numpy as np
import networkx as nx
from collections import deque
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_graph(path):
    """加载图数据"""
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u = int(parts[0])
                v = int(parts[1])
                G.add_edge(u, v)
    return G

def save_edges_txt(H, out_path):
    """保存图到文件"""
    with open(out_path, 'w') as f:
        for u, v in H.edges():
            f.write(f"{u} {v}\n")

def relabel_to_integers(H):
    """重新标记节点为连续整数"""
    mapping = {old: i for i, old in enumerate(sorted(H.nodes()))}
    return nx.relabel_nodes(H, mapping, copy=True)

def ensure_connectivity(G, sampled_nodes):
    """确保采样后的图是连通的"""
    if not sampled_nodes:
        return set()
    
    # 获取最大连通分量
    subgraph = G.subgraph(sampled_nodes)
    components = list(nx.connected_components(subgraph))
    if len(components) == 1:
        return sampled_nodes
    
    # 选择最大的连通分量
    largest_comp = max(components, key=len)
    
    # 如果最大的连通分量太小，添加桥接节点
    if len(largest_comp) < len(sampled_nodes) // 2:
        # 找到连接不同分量的最短路径
        remaining_nodes = sampled_nodes - largest_comp
        for node in remaining_nodes:
            # 尝试找到连接路径
            for target in largest_comp:
                try:
                    path = nx.shortest_path(G, source=node, target=target)
                    largest_comp.update(path)
                    break
                except:
                    continue
    
    return largest_comp

# ==================== 改进的采样方法 ====================

def sample_degree_based(G, k, start_node=None):
    """
    度优先采样：优先选择高度节点
    优点：保留枢纽节点，保持度分布形状
    """
    if start_node is None or start_node not in G:
        # 选择度数最高的节点作为起点
        start_node = max(G.nodes(), key=lambda x: G.degree(x))
    
    sampled_nodes = set([start_node])
    frontier = deque([start_node])
    
    # 按度数排序的候选节点
    all_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
    node_index = 1  # 从第二个节点开始（第一个是start_node）
    
    while len(sampled_nodes) < k and node_index < len(all_nodes):
        # 从高度节点中采样
        while node_index < len(all_nodes) and len(sampled_nodes) < k:
            node = all_nodes[node_index]
            if node not in sampled_nodes:
                sampled_nodes.add(node)
                # 添加一些邻居以保持连通性
                neighbors = list(G.neighbors(node))
                if neighbors:
                    for neighbor in neighbors[:2]:  # 添加前2个邻居
                        if len(sampled_nodes) < k and neighbor not in sampled_nodes:
                            sampled_nodes.add(neighbor)
            node_index += 1
    
    # 如果还不够，添加随机节点
    if len(sampled_nodes) < k:
        remaining = list(set(G.nodes()) - sampled_nodes)
        random.shuffle(remaining)
        sampled_nodes.update(remaining[:k - len(sampled_nodes)])
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

def sample_k_core_based(G, k, start_node=None):
    """
    K-Core采样：从网络核心开始采样
    优点：保持网络的核心-边缘结构
    """
    # 计算核心数
    core_numbers = nx.core_number(G)
    
    # 按核心数分组
    core_levels = {}
    for node, core in core_numbers.items():
        if core not in core_levels:
            core_levels[core] = []
        core_levels[core].append(node)
    
    # 从高核心级别开始采样
    sampled_nodes = set()
    sorted_cores = sorted(core_levels.keys(), reverse=True)
    
    for core_level in sorted_cores:
        nodes_at_level = core_levels[core_level]
        random.shuffle(nodes_at_level)
        
        for node in nodes_at_level:
            if len(sampled_nodes) >= k:
                break
            sampled_nodes.add(node)
    
    # 如果不够，补充低核心节点
    if len(sampled_nodes) < k:
        remaining = list(set(G.nodes()) - sampled_nodes)
        random.shuffle(remaining)
        sampled_nodes.update(remaining[:k - len(sampled_nodes)])
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

def sample_forest_fire(G, k, burn_prob=0.7, start_node=None):
    """
    森林火灾采样：更可能跟随高度数节点
    优点：自然偏好重要节点，保持局部拓扑
    """
    if start_node is None or start_node not in G:
        start_node = random.choice(list(G.nodes()))
    
    sampled_nodes = set([start_node])
    queue = deque([start_node])
    
    while len(sampled_nodes) < k and queue:
        current = queue.popleft()
        
        # 获取邻居并按度数排序（高度优先）
        neighbors = list(G.neighbors(current))
        neighbors.sort(key=lambda x: G.degree(x), reverse=True)
        
        for neighbor in neighbors:
            if neighbor not in sampled_nodes and len(sampled_nodes) < k:
                # 燃烧概率与邻居度数正相关
                degree_weight = min(0.9, 0.3 + 0.6 * G.degree(neighbor) / max(G.degree(n) for n in G.nodes()))
                if random.random() < burn_prob * degree_weight:
                    sampled_nodes.add(neighbor)
                    queue.append(neighbor)
            
            if len(sampled_nodes) >= k:
                break
        
        # 如果队列空了但还没采够，从已采样节点中重启
        if not queue and len(sampled_nodes) < k:
            # 选择度数最高的已采样节点重启
            if sampled_nodes:
                highest_degree = max(sampled_nodes, key=lambda x: G.degree(x))
                queue.append(highest_degree)
            else:
                remaining = list(set(G.nodes()) - sampled_nodes)
                if remaining:
                    queue.append(random.choice(remaining))
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

def sample_hybrid(G, k, start_node=None):
    """
    混合策略采样：组合多种方法
    优点：平衡各种特性，综合性能好
    """
    # 策略1：确保包含核心节点（前5%的高度节点）
    degree_threshold = sorted([G.degree(n) for n in G.nodes()], reverse=True)[int(len(G) * 0.05)]
    core_nodes = [n for n in G.nodes() if G.degree(n) >= degree_threshold]
    
    sampled_nodes = set()
    if core_nodes:
        # 采样核心节点
        core_sample_size = min(len(core_nodes), k // 3)
        sampled_nodes.update(random.sample(core_nodes, core_sample_size))
    
    # 策略2：随机游走探索
    if start_node is None or start_node not in G:
        current = random.choice(list(G.nodes()))
    else:
        current = start_node
    
    visited = set([current])
    queue = deque([current])
    
    while len(visited) < k // 3 and queue:
        current = queue.popleft()
        neighbors = list(G.neighbors(current))
        random.shuffle(neighbors)
        
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                if len(visited) >= k // 3:
                    break
    
    sampled_nodes.update(visited)
    
    # 策略3：度优先补充
    remaining = k - len(sampled_nodes)
    if remaining > 0:
        all_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)
        for node in all_nodes:
            if node not in sampled_nodes and remaining > 0:
                sampled_nodes.add(node)
                remaining -= 1
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

def sample_community_aware(G, k, start_node=None):
    """
    社区感知采样：按社区比例采样
    优点：保持社区结构，避免采样偏差
    """
    try:
        # 使用Louvain算法检测社区
        import community as community_louvain
        partition = community_louvain.best_partition(G)
    except:
        # 如果社区检测失败，使用连通分量
        communities = list(nx.connected_components(G))
        partition = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        if not partition:
            # 如果还是失败，回退到混合采样
            return sample_hybrid(G, k, start_node)
    
    # 统计社区大小
    community_sizes = {}
    for node, comm_id in partition.items():
        if comm_id not in community_sizes:
            community_sizes[comm_id] = 0
        community_sizes[comm_id] += 1
    
    # 按比例分配采样名额
    total_nodes = len(G)
    sampled_nodes = set()
    
    for comm_id, size in community_sizes.items():
        # 该社区应采样的节点数
        comm_sample_size = max(1, int(k * size / total_nodes))
        
        # 获取该社区的所有节点
        comm_nodes = [n for n in partition if partition[n] == comm_id]
        
        if comm_sample_size <= len(comm_nodes):
            # 在该社区内按度数采样
            comm_nodes_sorted = sorted(comm_nodes, key=lambda x: G.degree(x), reverse=True)
            sampled_in_comm = comm_nodes_sorted[:comm_sample_size]
            sampled_nodes.update(sampled_in_comm)
        else:
            sampled_nodes.update(comm_nodes)
    
    # 如果采样数不够，补充随机节点
    if len(sampled_nodes) < k:
        remaining = list(set(G.nodes()) - sampled_nodes)
        random.shuffle(remaining)
        sampled_nodes.update(remaining[:k - len(sampled_nodes)])
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

def sample_snowball(G, k, start_node=None, depth=2):
    """
    滚雪球采样：从起点开始的多层扩展
    优点：保持局部密集结构，适合社区发现
    """
    if start_node is None or start_node not in G:
        start_node = random.choice(list(G.nodes()))
    
    sampled_nodes = set([start_node])
    current_layer = set([start_node])
    
    for d in range(depth):
        next_layer = set()
        for node in current_layer:
            neighbors = set(G.neighbors(node))
            new_neighbors = neighbors - sampled_nodes
            # 采样部分邻居
            if new_neighbors:
                sample_size = min(len(new_neighbors), (k - len(sampled_nodes)) // (depth - d))
                if sample_size > 0:
                    sampled_new = random.sample(list(new_neighbors), sample_size)
                    sampled_nodes.update(sampled_new)
                    next_layer.update(sampled_new)
        
        current_layer = next_layer
        if len(sampled_nodes) >= k or not current_layer:
            break
    
    # 如果还不够，添加随机节点
    if len(sampled_nodes) < k:
        remaining = list(set(G.nodes()) - sampled_nodes)
        random.shuffle(remaining)
        sampled_nodes.update(remaining[:k - len(sampled_nodes)])
    
    sampled_nodes = ensure_connectivity(G, sampled_nodes)
    return G.subgraph(sampled_nodes).copy()

# ==================== 原始方法的保留（兼容性） ====================

def sample_connected_subgraph(G, k, start_node=None):
    """原始方法：连通子图采样"""
    if start_node is None:
        start_node = random.choice(list(G.nodes()))
    
    visited = set([start_node])
    queue = deque([start_node])
    
    while queue and len(visited) < k:
        x = queue.popleft()
        for n in G.neighbors(x):
            if n not in visited:
                visited.add(n)
                queue.append(n)
            if len(visited) >= k:
                break
    
    if len(visited) < k:
        candidates = list(set(G.nodes) - visited)
        random.shuffle(candidates)
        for n in candidates:
            visited.add(n)
            if len(visited) >= k:
                break
    
    return G.subgraph(visited).copy()

def sample_fixed_node_subgraph(G, k, start_node=None):
    """原始方法：固定节点子图采样"""
    if k >= G.number_of_nodes():
        return G.copy()
    
    comps = list(nx.connected_components(G))
    largest = max(comps, key=len)
    if len(largest) < k:
        nodes = list(G.nodes)[:k]
        return G.subgraph(nodes).copy()
    
    H = G.subgraph(largest).copy()
    if start_node is None or start_node not in H:
        start_node = random.choice(list(H.nodes))
    
    visited = [start_node]
    seen = set(visited)
    queue = deque([start_node])
    
    while queue and len(visited) < k:
        x = queue.popleft()
        for n in H.neighbors(x):
            if n not in seen:
                seen.add(n)
                visited.append(n)
                queue.append(n)
            if len(visited) >= k:
                break
    
    return H.subgraph(visited).copy()

# ==================== 主函数 ====================

def evaluate_sampling_quality(original, sampled):
    """评估采样质量"""
    metrics = {}
    
    # 基本统计
    metrics['original_nodes'] = original.number_of_nodes()
    metrics['sampled_nodes'] = sampled.number_of_nodes()
    metrics['original_edges'] = original.number_of_edges()
    metrics['sampled_edges'] = sampled.number_of_edges()
    
    # 度分布统计
    orig_degrees = [d for _, d in original.degree()]
    samp_degrees = [d for _, d in sampled.degree()]
    
    metrics['avg_degree_orig'] = np.mean(orig_degrees)
    metrics['avg_degree_samp'] = np.mean(samp_degrees)
    metrics['max_degree_orig'] = np.max(orig_degrees)
    metrics['max_degree_samp'] = np.max(samp_degrees)
    
    # 聚类系数
    metrics['clustering_orig'] = nx.average_clustering(original)
    metrics['clustering_samp'] = nx.average_clustering(sampled)
    
    # 连通性
    metrics['is_connected_orig'] = nx.is_connected(original)
    metrics['is_connected_samp'] = nx.is_connected(sampled)
    
    # 如果连通，计算平均路径长度
    if metrics['is_connected_orig'] and metrics['is_connected_samp']:
        metrics['apl_orig'] = nx.average_shortest_path_length(original)
        metrics['apl_samp'] = nx.average_shortest_path_length(sampled)
    
    return metrics

def run_sampling(in_path, out_path, args):
    """执行单个文件的采样流程"""
    # 加载原始图
    print(f"----------------------------------------")
    print(f"处理文件: {os.path.basename(in_path)}")
    print(f"加载图: {in_path}")
    try:
        G = load_graph(in_path)
    except Exception as e:
        print(f"错误: 无法加载图 {in_path}: {e}")
        return

    original_size = G.number_of_nodes()
    if original_size == 0:
        print(f"警告: 图为空，跳过")
        return

    print(f"原始图: {original_size} 节点, {G.number_of_edges()} 边")
    
    # 确定采样大小
    k = min(args.n, original_size)
    print(f"采样目标: {k} 节点 (原始图的 {k/original_size*100:.1f}%)")
    
    # 选择采样方法
    sampler_map = {
        'connected': sample_connected_subgraph,
        'fixed': sample_fixed_node_subgraph,
        'degree': sample_degree_based,
        'kcore': sample_k_core_based,
        'forestfire': sample_forest_fire,
        'hybrid': sample_hybrid,
        'community': sample_community_aware,
        'snowball': sample_snowball,
    }
    
    sampler = sampler_map[args.method]
    print(f"使用采样方法: {args.method}")
    
    # 执行采样
    try:
        H = sampler(G, k, args.start_node)
    except Exception as e:
        print(f"错误: 采样失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 重新标记节点
    H = relabel_to_integers(H)
    
    # 保存结果
    save_edges_txt(H, out_path)
    
    # 输出统计信息
    print(f"\n采样结果:")
    print(f"  节点数: {H.number_of_nodes()}")
    print(f"  边数: {H.number_of_edges()}")
    print(f"  平均度: {sum(dict(H.degree()).values()) / max(1, H.number_of_nodes()):.2f}")
    print(f"  是否连通: {nx.is_connected(H)}")
    print(f"  输出文件: {out_path}")
    
    # 评估采样质量
    if args.evaluate and H.number_of_nodes() > 0:
        print(f"\n采样质量评估:")
        try:
            metrics = evaluate_sampling_quality(G, H)
            
            print(f"  度统计:")
            print(f"    平均度: {metrics['avg_degree_orig']:.2f} → {metrics['avg_degree_samp']:.2f}")
            print(f"    最大度: {metrics['max_degree_orig']} → {metrics['max_degree_samp']}")
            
            print(f"  聚类系数: {metrics['clustering_orig']:.4f} → {metrics['clustering_samp']:.4f}")
            
            if 'apl_orig' in metrics and 'apl_samp' in metrics:
                print(f"  平均路径长度: {metrics['apl_orig']:.2f} → {metrics['apl_samp']:.2f}")
            
            print(f"  连通性: {metrics['is_connected_orig']} → {metrics['is_connected_samp']}")
        except Exception as e:
            print(f"评估失败: {e}")

def main():
    parser = argparse.ArgumentParser(description='网络采样工具')
    parser.add_argument('--input', type=str, default='Enron.txt', help='输入图文件 (单文件模式)')
    parser.add_argument('--output', type=str, default=None, help='输出图文件 (默认: 输入文件名_sampled.txt)')
    parser.add_argument('--all', action='store_true', help='批量处理当前目录下所有 .txt 文件')
    parser.add_argument('--n', type=int, default=3000, help='采样节点数')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    parser.add_argument('--method', type=str, default='hybrid', 
                       choices=['connected', 'fixed', 'degree', 'kcore', 
                               'forestfire', 'hybrid', 'community', 'snowball'],
                       help='采样方法')
    parser.add_argument('--start_node', type=int, default=None, help='起始节点ID')
    parser.add_argument('--evaluate', action='store_true', help='评估采样质量')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # 路径处理
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.all:
        print(f"=== 开始批量处理 {base_dir} 下的所有 .txt 文件 ===")
        files = [f for f in os.listdir(base_dir) if f.endswith('.txt') and '_sampled' not in f]
        # 排序以保证顺序一致
        files.sort()
        print(f"找到 {len(files)} 个待处理文件: {files}")
        
        for f in files:
            in_path = os.path.join(base_dir, f)
            # 自动生成输出文件名: Name.txt -> Name_sampled.txt
            out_filename = os.path.splitext(f)[0] + '_sampled.txt'
            out_path = os.path.join(base_dir, out_filename)
            
            run_sampling(in_path, out_path, args)
            
        print(f"=== 批量处理完成 ===")
        
    else:
        # 单文件模式
        in_path = args.input if os.path.isabs(args.input) else os.path.join(base_dir, args.input)
        
        if args.output:
            out_path = args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)
        else:
            # 如果未指定输出，自动生成
            filename = os.path.basename(in_path)
            out_filename = os.path.splitext(filename)[0] + '_sampled.txt'
            out_path = os.path.join(os.path.dirname(in_path), out_filename)
            
        run_sampling(in_path, out_path, args)

if __name__ == '__main__':
    main()
