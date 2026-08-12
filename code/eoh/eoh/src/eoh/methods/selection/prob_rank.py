import random
def parent_selection(pop,m):
    # Sort population by objective and metrics to ensure ranks reflect all criteria
    def get_sort_key(ind):
        obj = ind['objective']
        fht_50 = float('inf')
        fht_10 = float('inf')
        if 'other_inf' in ind and isinstance(ind['other_inf'], dict):
            fht_50 = ind['other_inf'].get('fht_50', float('inf'))
            fht_10 = ind['other_inf'].get('fht_10', float('inf'))
        return (obj, fht_50, fht_10)
    
    # Create a sorted list of indices or just sort the pop temporarily (or assume pop is sorted? Better to be safe)
    # Note: Sorting here doesn't affect the original list order outside unless we modify it in place.
    # But random.choices returns elements.
    sorted_pop = sorted(pop, key=get_sort_key)
    
    ranks = [i for i in range(len(sorted_pop))]
    probs = [1 / (rank + 1 + len(sorted_pop)) for rank in ranks]
    parents = random.choices(sorted_pop, weights=probs, k=m)
    return parents