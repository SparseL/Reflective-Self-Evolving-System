
import random

def parent_selection(population, m):
    tournament_size = 2
    parents = []
    while len(parents) < m:
        tournament = random.sample(population, tournament_size)
        tournament_fitness = []
        for fit in tournament:
            obj = fit['objective']
            fht_50 = float('inf')
            fht_10 = float('inf')
            if 'other_inf' in fit and isinstance(fit['other_inf'], dict):
                fht_50 = fit['other_inf'].get('fht_50', float('inf'))
                fht_10 = fit['other_inf'].get('fht_10', float('inf'))
            tournament_fitness.append((obj, fht_50, fht_10))
            
        winner = tournament[tournament_fitness.index(min(tournament_fitness))]
        parents.append(winner)
    return parents