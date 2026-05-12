import numpy as np
from scipy.spatial.distance import pdist, squareform

genes_per_triangle = 10

def apply_fitness_sharing(population, sigma_share, alpha=1.0):
    """
    Annotate every individual with a 'shared_fitness' value via Goldberg-style fitness sharing on the per-gene-normalised Euclidean distance between genomes.

    The same normalisation used by 'compute_diversity' is applied: each gene is divided by its natural range (W, H, 255, or 1) before the distance is computed. 
    Pairwise distances are then passed through a triangular sharing kernel:

        sh(d) = max(0, 1 - (d / sigma_share) ** alpha)

    The niche count of an individual is the sum of sh(d_ij) over every j (j = i contributes 1). For a minimisation problem the shared fitness is

        shared_fitness(i) = raw_fitness(i) * niche_count(i)

    So individuals in crowded regions of genome space are penalised while isolated ones keep their raw value. Raw 'fitness()' is left
    untouched, so elitism and final reporting still use the true RMSE.

    Parameters:
        population (list[TriangleSolution]): Current population.
        sigma_share (float): Sharing radius on the normalised-genome scale. Pairs of genomes within 'sigma_share' of each other share niche credit; pairs farther apart do not interact.
        alpha (float, optional): Shape exponent of the sharing kernel. Alpha = 1 (default) gives the standard triangular kernel.

    Returns:
        None. Each individual gains a `shared_fitness` attribute. Raw 'fitness()' values are not modified.
    """
    W, H = population[0].img_width, population[0].img_height
    D = len(population[0].repr)

    # Same normalisation as compute_diversity.
    block_scale = np.array([W, H, W, H, W, H, 255.0, 255.0, 255.0, 1.0])
    scale = np.tile(block_scale, D // genes_per_triangle)
    reprs = np.asarray([ind.repr for ind in population], dtype=np.float64) / scale

    # Full N×N normalised-Euclidean distance matrix.
    dists = squareform(pdist(reprs, metric='euclidean'))

    # Triangular sharing kernel, zero beyond sigma_share.
    sh = np.maximum(0.0, 1.0 - (dists / sigma_share) ** alpha)
    niche_counts = sh.sum(axis=1)  # includes sh(0) = 1 for self

    raw_fitnesses = [ind.fitness() for ind in population]
    for ind, raw, m in zip(population, raw_fitnesses, niche_counts):
        ind.shared_fitness = raw * m