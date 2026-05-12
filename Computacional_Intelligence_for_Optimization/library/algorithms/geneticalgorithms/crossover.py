import random
from copy import deepcopy

genes_per_triangle = 10


def one_point_crossover(parent1, parent2, xo_prob, verbose=False):
    """
    One-point crossover aligned to triangle boundaries.

    A single split point is chosen uniformly among the triangle-block
    boundaries (multiples of genes_per_triangle = 10), so triangles are
    never split mid-gene. The tails after the cut are swapped between
    the offspring:

        offspring1 = parent1[:p] + parent2[p:]
        offspring2 = parent2[:p] + parent1[p:]

    Falls back to plain replication when random() > xo_prob.
    """
    if random.random() > xo_prob:
        if verbose:
            print("Replication (no crossover)")
        return deepcopy(parent1), deepcopy(parent2)

    block    = 10  # genes per triangle
    n_blocks = len(parent1.repr) // block # number of triangles

    # Single cut at a triangle boundary in [1, n_blocks - 1].
    b = random.randint(1, n_blocks - 1)
    p = b * block

    g1 = parent1.repr[:p] + parent2.repr[p:]
    g2 = parent2.repr[:p] + parent1.repr[p:]

    if verbose:
        print(f"One-point crossover at gene index {p} (triangle {b})")

    return parent1.with_repr(g1), parent2.with_repr(g2)


def two_point_crossover(parent1, parent2, xo_prob, verbose=False):
    """
    Two-point crossover aligned to triangle boundaries.

    Two distinct split points are chosen uniformly among the triangle-block
    boundaries (multiples of genes_per_triangle = 10), so triangles are
    never split mid-gene.  The middle segment between the two cuts is
    swapped between the offspring:

        offspring1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        offspring2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    Falls back to plain replication when random() > xo_prob.
    """
    if random.random() > xo_prob:
        if verbose:
            print("Replication (no crossover)")
        return deepcopy(parent1), deepcopy(parent2)

    block    = 10  # genes per triangle
    n_blocks = len(parent1.repr) // block # number of triangles

    # Pick two distinct boundary indices in [1, n_blocks-1] and sort them.
    # random.sample gives us distinctness for free.
    b1, b2 = sorted(random.sample(range(1, n_blocks), 2))
    p1 = b1 * block
    p2 = b2 * block

    g1 = parent1.repr[:p1] + parent2.repr[p1:p2] + parent1.repr[p2:]
    g2 = parent2.repr[:p1] + parent1.repr[p1:p2] + parent2.repr[p2:]

    if verbose:
        print(f"Two-point crossover at gene indices {p1} and {p2} "
              f"(triangles {b1} and {b2})")

    return parent1.with_repr(g1), parent2.with_repr(g2)
