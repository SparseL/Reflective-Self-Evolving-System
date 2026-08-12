import heapq

def population_management(pop,size):
    pop = [individual for individual in pop if individual['objective'] is not None]
    # dedupe by code to avoid re-evaluating identical algorithms
    seen_codes = set()
    unique_by_code = []
    for individual in pop:
        code = individual.get('code')
        if code is None or code in seen_codes:
            continue
        seen_codes.add(code)
        unique_by_code.append(individual)
    pop = unique_by_code
    
    # Sort by objective and metrics to ensure best metrics are kept when deduping by objective
    def get_sort_key(ind):
        obj = ind['objective']
        fht_50 = float('inf')
        fht_10 = float('inf')
        if 'other_inf' in ind and isinstance(ind['other_inf'], dict):
            fht_50 = ind['other_inf'].get('fht_50', float('inf'))
            fht_10 = ind['other_inf'].get('fht_10', float('inf'))
        return (obj, fht_50, fht_10)

    pop.sort(key=get_sort_key)

    if size > len(pop):
        size = len(pop)
    unique_pop = [] 
    unique_objectives = []
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual['objective'])
    # Delete the worst individual
    #pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    pop_new = heapq.nsmallest(size, unique_pop, key=get_sort_key)
    return pop_new
