#Let's include the parent directory in the path so we can import our custom classes
#import sys
#sys.path.append('..')

import random

from library.problem.solution import Solution


def tournament_selection(population: list[Solution], minimization: bool, tournament_size: int = 3, key=None):
    """
    Select one individual via tournament.
    
    Picks 'tournament_size' individuals without replacement and returns the best one according to 'key'. 
    By default 'key' reads the raw fitness, but passing e.g. `key=lambda ind: ind.shared_fitness` lets fitness sharing override the comparison without touching the rest of the GA.
    """

    # Find the best individual in the tournament 
    if key is None: # Default to raw fitness if no key is provided
        key = lambda ind: ind.fitness() # Use raw fitness by default

    tournament = random.sample(population, k=tournament_size) # Pick 'tournament_size' individuals without replacement
    return min(tournament, key=key) if minimization else max(tournament, key=key) # Return the best individual according to 'key'

def rank_selection(population: list[Solution], minimization: bool, key=None):
    """
    Select one individual via linear rank selection.

    Sorts the population by `key` and samples one individual with probability proportional to its rank. Passing 'key=lambda ind: ind.shared_fitness' enables fitness sharing.
    """
    
    if key is None: # Default to raw fitness if no key is provided
        key = lambda ind: ind.fitness() # Use raw fitness by default
    if not population: # Guard against empty population
        raise ValueError("Population cannot be empty.")

    sorted_pop = sorted(population, key=key, reverse=not minimization) # Sort population by fitness (best first for minimization, worst first for maximization)
    n = len(sorted_pop) # Assign selection probabilities linearly based on rank (best gets n, worst gets 1)
    probabilities = [n - rank for rank in range(n)] # Linear probabilities: best gets n, second best gets n-1, ..., worst gets 1
    return random.choices(sorted_pop, weights=probabilities, k=1)[0] # Sample one individual according to the computed probabilities