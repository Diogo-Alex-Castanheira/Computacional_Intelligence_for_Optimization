import random

genes_per_triangle = 10 # number of genes that define a single triangle (x1, y1, x2, y2, x3, y3, r, g, b, alpha)


def gaussian_mutation(indiv, mut_prob,
                      sigma_vertex_frac=0.03,
                      sigma_color_frac=0.1,
                      sigma_alpha=0.1):
    """
    Apply per-gene Gaussian mutation to a TriangleSolution.

    Each gene in the genome is independently perturbed with probability 'mut_prob' by adding zero-mean Gaussian noise, then clamped back into its valid range. 
    With 10 genes per triangle laid out as [x1, y1, x2, y2, x3, y3, r, g, b, α], a different step size is used for each gene type so that the magnitude of the perturbation 
    matches the scale of the gene:

        • vertex x : σ = sigma_vertex_frac * img_width   ,  clamp [0, W]
        • vertex y : σ = sigma_vertex_frac * img_height  ,  clamp [0, H]
        • r, g, b  : σ = sigma_color_frac  * 255         ,  clamp [0, 255]
        • alpha    : σ = sigma_alpha                     ,  clamp [0, 1]

    Parameters:
        indiv (TriangleSolution): The individual to mutate.
        mut_prob (float): Probability in [0, 1] of mutating each gene independently.
        sigma_vertex_frac (float, optional): Std-dev of the vertex-gene perturbation, expressed as a fraction of the image dimension. Defaults to 0.03 (≈3% of width/height).
        sigma_color_frac (float, optional): Std-dev of the colour-gene perturbation, expressed as a fraction of the [0, 255] range. Defaults to 0.1 (≈25.5 colour units).
        sigma_alpha (float, optional): Std-dev of the alpha-gene perturbation, in absolute units (alpha already lives in [0, 1]). Defaults to 0.1.

    Returns:
        TriangleSolution: A new individual built via `indiv.with_repr(...)` whose representation is the mutated copy. The input individual is not modified.
    """
    W, H = indiv.img_width, indiv.img_height
    new_repr = indiv.repr.copy()

    for i in range(len(new_repr)):
        if random.random() > mut_prob:
            continue

        within = i % genes_per_triangle # which gene type you're looking at within a triangle, regardless of which triangle it belongs to

        if within in (0, 2, 4):          # x coordinates
            new_repr[i] += random.gauss(0, sigma_vertex_frac * W)
            new_repr[i] = max(0.0, min(float(W), new_repr[i]))
        elif within in (1, 3, 5):        # y coordinates
            new_repr[i] += random.gauss(0, sigma_vertex_frac * H)
            new_repr[i] = max(0.0, min(float(H), new_repr[i]))
        elif within in (6, 7, 8):        # r, g, b
            new_repr[i] += random.gauss(0, sigma_color_frac * 255)
            new_repr[i] = max(0.0, min(255.0, new_repr[i]))
        else:                            # alpha (within == 9)
            new_repr[i] += random.gauss(0, sigma_alpha)
            new_repr[i] = max(0.0, min(1.0, new_repr[i]))

    return indiv.with_repr(new_repr)


def uniform_mutation(indiv, mut_prob):
    """
    Apply per-gene uniform-reset mutation to a TriangleSolution.

    With probability `mut_prob`, each gene is independently replaced by a fresh uniformly-random value drawn from its valid range:

        • x coordinates → U(0, img_width)
        • y coordinates → U(0, img_height)
        • r, g, b       → U(0, 255)
        • alpha         → U(0, 1)

    Unlike Gaussian mutation, the new value does not depend on the current one, which makes uniform mutation strong at escaping local optima but disruptive if applied with a high probability. 
    Typical usage is a low `mut_prob` (e.g. 0.01-0.03) so most genes are preserved each generation.

    Parameters:
        indiv (TriangleSolution): The individual to mutate. 
        mut_prob (float): Probability in [0, 1] of resetting each gene independently.

    Returns:
        TriangleSolution: A new individual whose representation is the mutated copy. The input individual is not modified.
    """
    W, H = indiv.img_width, indiv.img_height
    new_repr = indiv.repr.copy()

    for i in range(len(new_repr)):
        if random.random() > mut_prob:
            continue

        within = i % genes_per_triangle # which gene type you're looking at within a triangle, regardless of which triangle it belongs to

        if within in (0, 2, 4):          # x coordinates
            new_repr[i] = random.uniform(0.0, W)
        elif within in (1, 3, 5):        # y coordinates
            new_repr[i] = random.uniform(0.0, H)
        elif within in (6, 7, 8):        # r, g, b
            new_repr[i] = random.uniform(0.0, 255.0)
        else:                            # alpha
            new_repr[i] = random.random()

    return indiv.with_repr(new_repr)