import random
def parent_selection(population, m):
    # Incorporate FHT metrics into fitness for tie-breaking
    # Effective Objective = Objective + small_weight * FHT
    # This ensures that for same ANC, lower FHT has higher fitness (1/Obj)
    effective_fitness_values = []
    for fit in population:
        obj = fit['objective']
        fht_50 = 0
        if 'other_inf' in fit and isinstance(fit['other_inf'], dict):
            fht_50 = fit['other_inf'].get('fht_50', 0)
        
        # Use a very small weight so FHT acts as a tie-breaker or secondary optimization
        # 1e-6 means 1000 nodes FHT adds 0.001 to objective.
        eff_obj = obj + (fht_50 * 1e-6) 
        effective_fitness_values.append(1 / (eff_obj + 1e-6))

    fitness_sum = sum(effective_fitness_values)
    probs = [fit / fitness_sum for fit in effective_fitness_values]
    parents = random.choices(population, weights=probs, k=m)
    return parents