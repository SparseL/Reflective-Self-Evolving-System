class GetPrompts():
  def __init__(self, use_precompute=True, init_prompt_profile='standard', aware_prompt_profile='none'):
    self.init_prompt_profile = init_prompt_profile or 'standard'
    self.aware_prompt_profile = aware_prompt_profile or 'none'
    self.prompt_task = "Given a graph, \
you need to find the critical nodes. \
The task can be solved step-by-step by iteratively choosing the next node from unvisited nodes. \
Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step. \
Tip: The time complexity of an algorithm is also an important metric. \
\
I will provide some performance metrics for reference: \
- Objective: The main objective value (lower is better, typically representing connectivity). \
- time_select: The time (in seconds) taken to select the critical nodes. \
- time_anc: The time (in seconds) taken to calculate the ANC (Average Normalized Connectivity) after selection. This reflects how fast the network disintegration is evaluated. \
- anc_prefix_k: The ANC calculated considering only the first k nodes removed. This indicates early-stage disruption capability. \
- fht_50: First Hitting Time (number of nodes removed) to reduce the network connectivity to 50%. Smaller is better (faster disruption). \
- fht_10: First Hitting Time (number of nodes removed) to reduce the network connectivity to 10%. Smaller is better. \
- lcc_at_k_frac: The size of the Largest Connected Component (normalized by total nodes) after removing k nodes. Smaller is better. \
- k: The specific number of nodes (typically 10 percent of total nodes) used for prefix/LCC metrics. \
\
How to interpret these metrics together: \
- **Fast vs. Thorough**: A low 'Objective' means the network is thoroughly fragmented by the end. Low 'fht_50' and 'fht_10' mean the network breaks apart *quickly*. The best algorithms achieve both: they start strong (low FHT, low anc_prefix_k) and finish strong (low Objective). \
- **Efficiency**: If two algorithms have similar connectivity scores (Objective), prefer the one with lower time complexity (time_select) or faster disruption (lower FHT). \
- **Trade-offs**: Sometimes aggressive early breaking (low FHT) might lead to suboptimal final connectivity. Your goal is to find a strategy that balances immediate impact with long-term optimality."

# self.prompt_task = "Given a graph, \
# you need to find the critical nodes. \
# The task can be solved step-by-step by iteratively choosing the next node from unvisited nodes. \
# Help me design a novel algorithm that is different from the algorithms in literature to select the next node in each step and the time complex should be small."

    self.prompt_func_name = "select_next_node"
    self.prompt_func_inputs = ["graph", "unvisited_nodes"]
    self.prompt_func_outputs = ["next_node"]
    self.prompt_inout_inf = "'graph' is a networkx.Graph, and 'unvisited_nodes' is a list. 'next_node' is the id of node."
    self.prompt_other_inf = ""
    if use_precompute:
        self.prompt_other_inf = "You can directly use precomputed features to reduce time complexity. Node attributes available: graph.nodes[node]['degree'], graph.nodes[node]['clustering'], graph.nodes[node]['core_number'], graph.nodes[node]['betweenness'], graph.nodes[node]['eigenvector']. Graph-level features available in graph.graph['precomputed']: avg_degree, degree_variance, apl_lcc, diameter_lcc, max_core_k, bridge_ratio_high_bc. Prefer reading these caches over recomputing global metrics inside loops."
    else:
        self.prompt_other_inf = ""
    self.prompt_init_guidance = ""
    if self.init_prompt_profile == 'enhanced_i1':
        self.prompt_init_guidance = "For the initial population, design a medium-complexity heuristic rather than a toy single-metric rule. Compute one compact score for each node in unvisited_nodes and return the node with the highest score. Prefer O(|unvisited_nodes|) selection time per step. Combine 3 to 5 interpretable terms from complementary signals: influence (degree, core_number, eigenvector), bridge or bottleneck potential (betweenness, bridge_ratio_high_bc), local redundancy penalty (lower clustering is usually better), and graph-level scaling (avg_degree, degree_variance, max_core_k, apl_lcc, diameter_lcc). Use simple normalization, logarithms, square roots, bounded ratios, or safe epsilons where helpful. Do not rely on only one metric. Do not copy the graph, simulate removing every candidate, or call expensive NetworkX global centrality functions inside select_next_node."
    self.prompt_aware_guidance_by_profile = {
        'random_noise_edge': (
            "Aware guidance for this evolutionary stage: the input graph contains random noise edges added to the original Enron_sampled graph. "
            "Treat these added edges as potentially unreliable shortcuts rather than genuine structural bridges. "
            "Prefer heuristics that identify robust bridge and influence nodes using edge embeddedness, common-neighbor support, local core consistency, "
            "neighbor agreement, and multi-step stability, while remaining compatible with the available precomputed node and graph features. "
            "Do not simply favor nodes incident to isolated or weakly supported edges, and do not recompute expensive global centralities or simulate graph copies inside select_next_node."
        ),
        'ci_boundary': (
            "Aware guidance for this evolutionary stage: improve the current heuristic with local dismantling mechanisms inspired by Collective Influence, without copying an external implementation. "
            "Prefer adding a lightweight radius-2 or radius-3 boundary term such as (residual_degree(node)-1) times the sum of residual degrees on the boundary. "
            "Use this CI-like boundary mass together with existing bridge/core/low-clustering signals, and keep selection practical by restricting expensive local traversals to small radii, cached adjacency, or a candidate subset when needed. "
            "Avoid full graph copies, all-candidate removal simulations, global centrality recomputation, or heavy reinsertion inside select_next_node."
        ),
        'ci_reinsertion': (
            "Aware guidance for this evolutionary stage: borrow the mechanism-level lesson from CI with reinsertion. "
            "Design rules that first favor nodes likely to rapidly reduce the largest connected component, then avoid redundant removals whose neighborhoods are already fragmented. "
            "Approximate reinsertion/pruning online with cheap signals: neighbor coverage, fraction of already removed neighbors, diversity of unvisited neighbor core/degree values, and whether unvisited neighbors look mutually disconnected. "
            "If using a CI-like radius boundary score, combine it with a redundancy penalty so the sequence behaves like a compact dismantling set followed by lower-priority cleanup. "
            "Do not implement an expensive full reinsertion pass, copy the graph for every candidate, or recompute global metrics inside select_next_node."
        ),
    }
    self.prompt_aware_guidance = self.prompt_aware_guidance_by_profile.get(self.aware_prompt_profile, "")

  def get_task(self):
    return self.prompt_task

  def get_func_name(self):
    return self.prompt_func_name

  def get_func_inputs(self):
    return self.prompt_func_inputs

  def get_func_outputs(self):
    return self.prompt_func_outputs

  def get_inout_inf(self):
    return self.prompt_inout_inf

  def get_other_inf(self):
    return self.prompt_other_inf

  def get_init_guidance(self):
    return self.prompt_init_guidance

  def get_aware_profile(self):
    return self.aware_prompt_profile

  def get_aware_guidance(self, profile=None):
    if profile is None:
      return self.prompt_aware_guidance
    return self.prompt_aware_guidance_by_profile.get(profile or 'none', "")


if __name__ == "__main__":
  getprompts = GetPrompts()
  print(getprompts.get_task())
