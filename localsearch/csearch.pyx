
import numpy as np



# cdef double eucl_dist(double[:,:] coords, int[:] tour, int a, int b):
#     """
#     Return the euclidean distance between cities tour[a] and tour[b].
#     Supports circular indexing with modulo
#     """
#     cdef int n
    
#     n = len(tour)
#     return np.hypot(coords[tour[a % n], 0] - coords[tour[b % n], 0],
#                     coords[tour[a % n], 1] - coords[tour[b % n], 1])


cdef double eucl_dist(double[:,:] coords, int[:] tour, int a, int b):
    """
    Return the euclidean distance between cities tour[a] and tour[b].
    Supports circular indexing with modulo.

    Extension of above eucl_dist() to work with higher dimensional TSP
    Probably slightly slower with the extra function calls and loop compared to np.hypot
    """
    cdef int n, d

    n = coords.shape[0]
    d = coords.shape[1]

    cdef double[:] deltas = np.zeros(d, dtype=np.double)

    for i in range(d):
        deltas[i] = coords[tour[a % n], i] - coords[tour[b % n], i]

    return np.sqrt(np.sum(np.square(deltas), axis=0))


######################################################################################################
### 2 OPT ############################################################################################
######################################################################################################


cpdef two_opt(double[:,:] coords, int[:] start_tour):
    """
    Local search with neighborhood defined as all possible 2-edge swaps,
    which equates to selecting a subtour to reverse in order.
    Guaranteed to uncross all tour segements when finding a local optima.
    """
    cdef float min_change, change
    cdef int i, j, min_i, min_j, num_cities, j_bound
    cdef int[:] tour = np.copy(start_tour)

    min_change = 0
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

    return np.asarray(tour), min_change


cpdef two_opt_search(double[:,:] coords, int[:] start_tour, int max_swaps=10_000):
    """
    Iterative 2opt local search starting from start_tour until finding local optima
    """
    cdef bint improving = True
    cdef float cost_reduction = 0
    cdef int swaps = 0

    cdef int[:] curr_tour = np.copy(start_tour)
    cdef int[:] prev_tour = np.copy(start_tour)

    while improving:
        prev_tour = np.copy(curr_tour)
        curr_tour, cost_reduction = two_opt(coords, curr_tour)
        improving = cost_reduction < 0

        if improving:
            swaps += 1

            if swaps > max_swaps:
                curr_tour = np.copy(prev_tour)
                break

    return np.asarray(curr_tour), swaps


cpdef batch_two_opt_search(double[:,:,:] dataset, int[:,:] start_tours, int max_swaps=10_000):
    """
    Avoids overhead of calling cython every individual problem, instead
    just once per batch
    """
    cdef int[:,:] optima = np.empty_like(start_tours)
    cdef int[:] swaps = np.empty(len(start_tours), dtype=np.int32)
    
    cdef int[:] t_optima
    cdef int t_swaps

    for idx in range(len(dataset)):
        t_optima, t_swaps = two_opt_search(dataset[idx, :, :], start_tours[idx, :], max_swaps)

        optima[idx, :] = t_optima
        swaps[idx] = t_swaps

    return np.asarray(optima), np.asarray(swaps)


######################################################################################################
### 2 SWAP ###########################################################################################
######################################################################################################


cpdef two_swap(double[:,:] coords, int[:] start_tour):
    """
    Local search with neighborhood defined as all possible 2-*node* swaps, (aka 2-exchange)
    basically 2-opt in node-space instead of edge space, swapping 4 edges in and 4 edges out instead of 2 and 2.
    ***NOT*** guaranteed to uncross all tour segements when finding a local optima.
    In expectation, finds worse local optima than 2-opt. Useful as an increasingly naive baseline.
    """
    cdef float min_change, change, new_edge_dists, removed_edge_dists
    cdef int i, j, min_i, min_j, num_cities, exchange
    cdef int[:] tour = np.copy(start_tour)

    min_change = 0
    num_cities = len(tour)
    # Find the best move
    for i in range(num_cities - 1):
        for j in range(i + 1, num_cities):
            new_edge_dists = eucl_dist(coords, tour, i, j-1) + eucl_dist(coords, tour, i+1, j) + eucl_dist(coords, tour, i-1, j) + eucl_dist(coords, tour, i, j+1)

            removed_edge_dists = 0
            if i + 1 < j:
                removed_edge_dists += eucl_dist(coords, tour, i, i+1) + eucl_dist(coords, tour, j, j-1)
            if not (i == 0 and j == num_cities - 1):
                removed_edge_dists += eucl_dist(coords, tour, i, i-1) + eucl_dist(coords, tour, j, j+1)

            change = new_edge_dists - removed_edge_dists
            if change < min_change:
                min_change = change
                min_i, min_j = i, j

    # Update tour with best move
    if min_change < 0:
        # print(eucl_dist(coords, tour, min_i, min_j-1), eucl_dist(coords, tour, min_i+1, min_j), eucl_dist(coords, tour, min_i-1, min_j), eucl_dist(coords, tour, min_i, min_j+1))
        # print(eucl_dist(coords, tour, min_i, min_i+1), eucl_dist(coords, tour, min_i, min_i-1), eucl_dist(coords, tour, min_j, min_j+1), eucl_dist(coords, tour, min_j, min_j-1))
        # print(min_i, min_j)
        # print("b", np.asarray(tour))
        exchange = tour[min_i]
        tour[min_i] = tour[min_j]
        tour[min_j] = exchange
        # print("a", np.asarray(tour), "\n")

    return np.asarray(tour), min_change


cpdef two_swap_search(double[:,:] coords, int[:] start_tour, int max_swaps=10_000):
    """
    Copy of two_opt_search but for 2-swap neighborhood
    Ideally would use function arguments but that seems tricky with cython
    """
    cdef bint improving = True
    cdef float cost_reduction = 0
    cdef int swaps = 0

    cdef int[:] curr_tour = np.copy(start_tour)
    cdef int[:] prev_tour = np.copy(start_tour)

    while improving:
        prev_tour = np.copy(curr_tour)
        curr_tour, cost_reduction = two_swap(coords, curr_tour)  # NOTE 2-SWAP
        improving = cost_reduction < 0

        if improving:
            swaps += 1

            if swaps > max_swaps:
                curr_tour = np.copy(prev_tour)
                break

    return np.asarray(curr_tour), swaps


cpdef batch_two_swap_search(double[:,:,:] dataset, int[:,:] start_tours, int max_swaps=10_000):
    """
    Copy of batch_two_opt_search but for 2-swap neighborhood
    Ideally would use function arguments but that seems tricky with cython
    """
    cdef int[:,:] optima = np.empty_like(start_tours)
    cdef int[:] swaps = np.empty(len(start_tours), dtype=np.int32)
    
    cdef int[:] t_optima
    cdef int t_swaps

    for idx in range(len(dataset)):
        t_optima, t_swaps = two_swap_search(dataset[idx, :, :], start_tours[idx, :], max_swaps)  # NOTE 2-SWAP

        optima[idx, :] = t_optima
        swaps[idx] = t_swaps

    return np.asarray(optima), np.asarray(swaps)
