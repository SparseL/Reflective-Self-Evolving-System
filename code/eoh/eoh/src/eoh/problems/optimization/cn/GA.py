import random
import time

import numpy as np
from eoh.src.eoh.problems.optimization.cn.run import CriticalNode, get_anc
import heapq

code_snaps = [
    """
def select_next_node(graph, unvisited_nodes):
  max_unvisited_neighbors = -1
  next_node = None
  for node in unvisited_nodes:
    unvisited_neighbors = [n for n in graph.neighbors(node) if n in unvisited_nodes]
    if len(unvisited_neighbors) > max_unvisited_neighbors:
      max_unvisited_neighbors = len(unvisited_neighbors)
      next_node = node
  return next_node
""", """
def select_next_node(graph, unvisited_nodes):
    max_unvisited_neighbors = -1
    min_visited_neighbors = float('inf')
    next_node = None
    for node in unvisited_nodes:
        unvisited_neighbors = [n for n in graph.neighbors(node) if n in unvisited_nodes]
        visited_neighbors = [n for n in graph.neighbors(node) if n not in unvisited_nodes]
        if len(unvisited_neighbors) > max_unvisited_neighbors or (len(unvisited_neighbors) == max_unvisited_neighbors and len(visited_neighbors) < min_visited_neighbors):
            max_unvisited_neighbors = len(unvisited_neighbors)
            min_visited_neighbors = len(visited_neighbors)
            next_node = node
    return next_node
""", """
import networkx as nx
import math

def select_next_node(graph, unvisited_nodes):
    max_product = -1
    next_node = None
    for node in unvisited_nodes:
        degree = graph.degree[node]
        unvisited_neighbors = [n for n in graph.neighbors(node) if n in unvisited_nodes]
        product = len(unvisited_neighbors) * math.sqrt(degree)
        if product > max_product:
            max_product = product
            next_node = node
    return next_node
""", """
import networkx as nx
import math

def select_next_node(graph, unvisited_nodes):
    max_sum = -1
    next_node = None
    for node in unvisited_nodes:
        degree = graph.degree[node]
        unvisited_neighbors = [n for n in graph.neighbors(node) if n in unvisited_nodes]
        node_sum = len(unvisited_neighbors) + math.log(degree)
        if node_sum > max_sum:
            max_sum = node_sum
            next_node = node
    return next_node
"""]


class Problem:
    def __init__(self):
        self.dim


class GA:
    def __init__(self, problem=CriticalNode(), n=10, pm=0.1, pc=0.9):
        self.problem = problem
        self.dim = len(problem.instances[0].nodes)
        self.n = n
        self.pm = pm
        self.pc = pc

        self.pops = [np.random.permutation(self.dim) for _ in range(self.n)]
        self.fitness = [self.evaluate(x) for x in self.pops]

    def mutate(self, individual):
        """交换两个随机位置的元素来实现变异"""
        if random.random() < self.pm:
            # 随机选择两个位置进行交换
            idx1, idx2 = random.sample(range(self.dim), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

    def crossover(self, parent1, parent2):
        """部分映射交叉（PMX）"""
        if random.random() < self.pc:
            point1, point2 = sorted(random.sample(range(self.dim), 2))  # 随机选择两个交叉点
            child1, child2 = parent1.copy(), parent2.copy()

            # 交叉部分映射
            for i in range(point1, point2):
                child1[i], child2[i] = child2[i], child1[i]

            # 修复重复的元素
            self._repair(child1, parent1, parent2, point1, point2)
            self._repair(child2, parent2, parent1, point1, point2)

            return child1, child2
        else:
            return parent1.copy(), parent2.copy()

    def _repair(self, child, parent1, parent2, point1, point2):
        """修复交叉后的重复元素"""
        # 用来记录child中point1到point2区域以外的部分
        mapping = {}
        for i in range(point1, point2):
            mapping[child[i]] = parent1[i]

        # 修复子代中交叉区段外的重复值
        for i in list(range(0, point1)) + list(range(point2, self.dim)):
            while child[i] in child[point1:point2]:  # 如果有重复
                child[i] = mapping[child[i]]  # 通过映射找到非重复的值

    def evaluate(self, x):
        return get_anc(self.problem.instances[0], x)

    def tournament_selection(self, tournament_size):
        tournament = random.sample(range(self.n), tournament_size)
        best_individual_index = min(tournament, key=lambda idx: self.fitness[idx])
        return self.pops[best_individual_index]

    def run(self, num_iter):
        for i in range(num_iter):
            new_pops = []
            for i in range(0, self.n, 2):
                parent1, parent2 = random.sample(self.pops, 2)
                child1, child2 = self.crossover(parent1, parent2)
                new_pops.append(self.mutate(child1))
                new_pops.append(self.mutate(child2))

            self.pops += new_pops
            self.fitness = [self.evaluate(x) for x in self.pops]
            best_individual = self.pops[np.array(self.fitness).argmin()]
            self.pops = [self.tournament_selection(3) for _ in range(self.n - 1)]
            self.pops.append(best_individual)

        print(max(self.fitness))


def simulated_annealing(graph, initial_solution, initial_temp, cooling_rate, stopping_temp):
    num_nodes = len(initial_solution)

    current_nodes = initial_solution
    current_anc = get_anc(graph, current_nodes)

    # 设置初始温度和停止温度
    temperature = initial_temp

    while temperature > stopping_temp:
        # 生成新解
        new_nodes = current_nodes[:]
        i, j = random.sample(range(num_nodes), 2)
        new_nodes[i], new_nodes[j] = new_nodes[j], new_nodes[i]  # 交换两个城市的位置

        new_anc = get_anc(graph, new_nodes)

        # 判断是否接受新解
        if new_anc < current_anc or random.random() < np.exp((current_anc - new_anc) / temperature):
            current_nodes = new_nodes
            current_anc = new_anc

        # 降温
        temperature *= cooling_rate

    return current_anc, current_nodes


class Heap:
    def __init__(self, size):
        self.arr = []
        self.size = size

    def push(self, item):
        tmp = -item[0], item[1]
        if len(self.arr) < self.size:
            heapq.heappush(self.arr, tmp)
        else:
            heapq.heappushpop(self.arr, tmp)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, item):
        return -self.arr[item][0], self.arr[item][1]

    def __str__(self):
        return str([(-x[0], x[1]) for x in self.arr])


def solve_sa():
    num_instance = 2
    critical_node = CriticalNode(num_instance=num_instance)

    initial_solutions = []
    for snap in code_snaps:
        initial_solutions.append(critical_node.evaluate(snap, details=True))  # [4, num_instance, 2, 50]

    initial_temp = 1000  # 初始温度
    cooling_rate = 0.995  # 降温速率
    stopping_temp = 10  # 停止温度

    for i in range(1, num_instance + 1):
        if i % 10 == 0:
            print(i)

        solution_list = Heap(size=2)
        for solution in initial_solutions:
            solution_list.push((solution[0][i], solution[1][i]))
        print(solution_list)

        for trail in range(10):
            idx = random.randint(0, len(solution_list) - 1)
            item = simulated_annealing(critical_node.instances[i], solution_list[idx][1],
                                       initial_temp, cooling_rate, stopping_temp)

            solution_list.push(item)

        print(solution_list)


critical_node = CriticalNode()

import networkx as nx
def valid_hda(type="d"):
    g = critical_node.instances[0].copy()
    removed_nodes = []
    while len(g) > 0:

        if type == "d":
            degrees = g.degree()
        elif type == "b":
            degrees = nx.betweenness_centrality(g)
        elif type == "c":
            degrees = nx.closeness_centrality(g)
        elif type == "pr":
            degrees = nx.pagerank(g)

        degrees = dict(degrees)
        max_degree = max(degrees.values())
        max_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
        max_degree_node = random.choice(max_degree_nodes)

        g.remove_node(max_degree_node)
        removed_nodes.append(max_degree_node)

    ans = get_anc(critical_node.instances[0], removed_nodes) * 10000
    print(ans)
    return ans


def valid_key_player():
    # 验证一下key-player那篇文章的答案
    with open("/eoh/src/eoh/problems/optimization/cn/kp_answer.txt", "r") as file:
        print([0])
        initial_solution = [int(s) for s in file.readlines()]

        full_set = set(range(829))
        print("full", len(full_set))
        input_set = set(initial_solution)
        missing_numbers = full_set - input_set
        initial_solution += list(missing_numbers)
        print(len(initial_solution))

        critical_node = CriticalNode()
        print(get_anc(critical_node.instances[0], initial_solution))


if __name__ == "__main__":
    t = time.time()
    hda = np.array([valid_hda(type="b") for _ in range(5)])
    print(time.time() - t)
    print(np.min(hda), np.mean(hda))

    t = time.time()
    hda = np.array([valid_hda(type="c") for _ in range(10)])
    print(time.time() - t)
    print(np.min(hda), np.mean(hda))

    t = time.time()
    hda = np.array([valid_hda(type="pr") for _ in range(100)])
    print(time.time() - t)
    print(np.min(hda), np.mean(hda))

    exit(0)
    initial_solutions = []
    critical_node = CriticalNode()
    ga = GA(problem=critical_node)

    for snap in code_snaps:
        initial_solutions.append(critical_node.evaluate(snap, details=True))  # [4, num_instance, 2, 50]
    print([x[0][0] for x in initial_solutions])
    initial_solutions = [x[1][0] for x in initial_solutions]
    ga.pops[:4] = initial_solutions

    ga.run(50)
