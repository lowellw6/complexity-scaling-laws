
import numpy as np


##########################################################################
### Single-Tour ##########################################################
##########################################################################

def tei_node_hamming_distance(tour_a, tour_b):
    """
    TSP-equivalence-invariant Hamming distance of two tours
    Measures *minimum* position substitutions required to achieve tour equality,
    where tour equality is agnostic to starting node or reversed traversal

    Functionally --> retv = min([Hamming(tour_a_eq, tour_b) for tour_a_eq in the 2n cost-equivalent tours for tour_a])  // where n = # nodes
    Note this function is insensitive to swapping tour_a and tour_b as inputs  TODO prove this
    """
    assert np.issubdtype(tour_a.dtype, np.integer) and np.issubdtype(tour_b.dtype, np.integer)
    assert len(tour_a) == len(tour_b)
    n = len(tour_a)

    a_eq_rolls = np.stack([np.roll(tour_a, idx, axis=0) for idx in range(n)], axis=0)
    a_eq_reversed = np.fliplr(a_eq_rolls)
    a_eq_tours = np.concatenate((a_eq_rolls, a_eq_reversed), axis=0)  # (2n, n)

    b_repeat = np.stack(2 * n * [tour_b], axis=0)

    hamming_mask = a_eq_tours != b_repeat
    hamming_scores = np.sum(hamming_mask, axis=1)

    return np.min(hamming_scores)


def edge_hamming_distance(tour_a, tour_b):
    """
    Hamming distance in space of edges, where each symmetric TSP problem has n(n-1)/2 edges total
    
    Perfect match results in 0 distance
    Imperfect match has range [4, 2n], always even (2 is impossible)
    """
    assert np.issubdtype(tour_a.dtype, np.integer) and np.issubdtype(tour_b.dtype, np.integer)
    assert len(tour_a) == len(tour_b)

    def make_edge_mat(tour):
        n = len(tour_a)
        edge_mat = np.full((n, n), False)

        edges = np.stack((tour, np.roll(tour, -1, axis=0)))
        flip_edges = np.flip(edges, axis=0)

        sym_edges = np.where(edges[0] < edges[1], edges, flip_edges)  # upper triangle only for symmetric tsp, treating edge (y, x) as (x, y) if x < y

        edge_mat[sym_edges[0], sym_edges[1]] = True
        return edge_mat

    edge_mat_a = make_edge_mat(tour_a)
    edge_mat_b = make_edge_mat(tour_b)

    return np.logical_xor(edge_mat_a, edge_mat_b).sum()


##########################################################################
### Batched ##############################################################
##########################################################################

def batch_tei_node_hamming_distance(tours_a, tours_b):
    """
    See doc string for tei_node_hamming_distance
    Expects each input to have shape (batch, n)
    """
    assert np.issubdtype(tours_a.dtype, np.integer) and np.issubdtype(tours_b.dtype, np.integer)
    assert tours_a.shape == tours_b.shape

    _, n = tours_a.shape

    a_eq_rolls = np.stack([np.roll(tours_a, idx, axis=1) for idx in range(n)], axis=1)  # (batch, n, n)
    a_eq_reversed = np.flip(a_eq_rolls, axis=2)
    a_eq_tours = np.concatenate((a_eq_rolls, a_eq_reversed), axis=1)  # (batch, 2n, n)

    b_repeat = np.stack(2 * n * [tours_b], axis=1)

    hamming_mask = a_eq_tours != b_repeat
    hamming_scores = np.sum(hamming_mask, axis=2)  # (batch, 2n)

    return np.min(hamming_scores, axis=1)  # (batch,)


def batch_edge_hamming_distance(tours_a, tours_b):
    """
    See doc string for edge_hamming_distance
    Expects each input to have shape (batch, n)
    """
    assert np.issubdtype(tours_a.dtype, np.integer) and np.issubdtype(tours_b.dtype, np.integer)
    assert tours_a.shape == tours_b.shape

    b, n = tours_a.shape

    def batch_edge_mat(tours):
        edge_mat = np.full((b, n, n), False)

        edges = np.stack((tours, np.roll(tours, -1, axis=1)), axis=-1)  # (b, n, 2)
        flip_edges = np.flip(edges, axis=-1)

        start_edges = np.stack(2 * [edges[:, :, 0]], axis=-1)
        end_edges = np.stack(2 * [edges[:, :, 1]], axis=-1)
        sym_edges = np.where(start_edges < end_edges, edges, flip_edges)  # upper triangle only for symmetric tsp, treating edge (y, x) as (x, y) if x < y

        batch_idxs = np.repeat(np.arange(b), n, axis=0)
        flat_sym_edges = sym_edges.reshape(-1, 2)
        sym_edge_rows, sym_edge_cols = flat_sym_edges[:, 0], flat_sym_edges[:, 1]

        edge_mat[batch_idxs, sym_edge_rows, sym_edge_cols] = True
        return edge_mat

    edge_mat_a = batch_edge_mat(tours_a)
    edge_mat_b = batch_edge_mat(tours_b)

    return np.logical_xor(edge_mat_a, edge_mat_b).sum(axis=(1, 2))
 