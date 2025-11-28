"""
Pure python local search functions
Slow, mainly for testing and development
Use Cythonized versions in csearch.pyx for faster 
"""

import numpy as np


VERBOSE = False


def two_opt_python(coords, start_tour):
    def eucl_dist(coords, tour, a, b):
        """
        Return the euclidean distance between cities tour[a] and tour[b].
        Supports circular indexing with modulo
        """
        n = len(tour)
        return np.hypot(coords[tour[a % n], 0] - coords[tour[b % n], 0],
                        coords[tour[a % n], 1] - coords[tour[b % n], 1])

    min_change = 0
    tour = np.copy(start_tour)
    num_cities = len(tour)
    # Find the best move
    for i in range(num_cities - 2):
        j_bound = min(num_cities + i - 1, num_cities)
        for j in range(i + 2, j_bound):
            change = eucl_dist(coords, tour, i, j) + eucl_dist(coords, tour, i+1, j+1) - eucl_dist(coords, tour, i, i+1) - eucl_dist(coords, tour, j, j+1)
            if change < min_change:
                min_change = change
                min_i, min_j = i, j
    # Update tour with best move
    if min_change < 0:
        tour[min_i+1:min_j+1] = tour[min_i+1:min_j+1][::-1]

    return tour, min_change


def two_opt_search_python(coords, start_tour):
    # compute local optimum with pure python 2-opt running until no improvement can be made
    improving = True
    curr_tour = start_tour
    cost_reduction = None
    swaps = -1

    while improving:
        if VERBOSE:
            print(f"> {curr_tour} {cost_reduction if cost_reduction is not None else 'start'}")
        
        curr_tour, cost_reduction = two_opt_python(coords, curr_tour)
        improving = cost_reduction < 0
        swaps += 1

    return curr_tour, swaps