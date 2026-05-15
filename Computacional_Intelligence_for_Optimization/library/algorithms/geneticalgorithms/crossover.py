import random

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
        return parent1.clone(), parent2.clone()

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
        return parent1.clone(), parent2.clone()

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


def arithmetic_crossover(parent1, parent2, xo_prob, verbose=False):
    """
    Arithmetic (whole-genome) crossover — an instance of geometric crossover
    under the Euclidean metric.

    With probability `xo_prob`, the offspring are convex combinations of the
    parents' genomes:

        offspring1 = α · parent1 + (1 - α) · parent2
        offspring2 = (1 - α) · parent1 + α · parent2

    A single α is drawn uniformly from [0, 1] per crossover event, yielding
    two symmetric offspring on the line segment between the parents.

    Because α ∈ [0, 1], offspring gene values stay within the convex hull of
    the parents' values — every gene is automatically inside its valid range,
    so no clamping is needed.

    Falls back to plain replication when random() > xo_prob.
    """
    if random.random() > xo_prob:
        if verbose:
            print("Replication (no crossover)")
        return parent1.clone(), parent2.clone()

    a = random.random()
    b = 1.0 - a

    g1 = [a * x + b * y for x, y in zip(parent1.repr, parent2.repr)]
    g2 = [b * x + a * y for x, y in zip(parent1.repr, parent2.repr)]

    if verbose:
        print(f"Arithmetic crossover with α = {a:.3f}")

    return parent1.with_repr(g1), parent2.with_repr(g2)
