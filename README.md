# Girl with a Pearl Earring — Genetic Algorithm

A Genetic Algorithm that approximates Vermeer's *Girl with a Pearl Earring* using **100 semi-transparent triangles**. Built for the Computational Intelligence for Optimization.

## Result

The Final GA achieves an **RMSE of 26.10** against the target image after 1 000 generations (elitism off). With elitism on, the result is 27.22 — close, but elitism off is consistently better at this generation budget.

> Initialisation note: the genome is initialised with **fully random colours**. The target image is only consulted inside `fitness()`, never during initialisation or evolution.

## Problem

Each individual encodes 100 triangles as a flat list of 1 000 floats (10 genes per triangle):

```
[x1, y1, x2, y2, x3, y3, r, g, b, α]
```

with vertex coordinates in pixel space, RGB in `[0, 255]`, and alpha in `[0, 1]`. The fitness is the root mean squared error (RMSE) between the rendered candidate image and the target — to be **minimised**.

## Repository structure

```
.
├── Computacional_Intelligence_for_Optimization/
    ├── girl_with_pearl.ipynb                     # Main notebook
    └── library/
        ├── algorithms/geneticalgorithms/
        │   ├── selection.py                      # tournament, rank
        │   ├── crossover.py                      # one-point, two-point, arithmetic
        │   └── mutation.py                       # gaussian, uniform
        └── problem/
            ├── solution.py                       # Abstract Solution
            ├── triangle_solution.py              # Triangle-mosaic representation + rendering
            ├── fitness_sharing.py                # Goldberg-style fitness sharing
            └── data/girl_pearl_earing.png        # Target image
    └── output
```

## Notebook structure

1. **Setup** — imports
2. **Load and display target image**
3. **The First Genetic Algorithm** — defines the GA loop, runs an elitism on/off comparison, visualises the result
4. **Ablation Study** — three operator comparisons:
   - Tournament vs Rank Selection
   - One-Point vs Two-Point vs Arithmetic Crossover
   - Gaussian vs Uniform Mutation
5. **Final GA Algorithm** — combines the winning operators and runs for 1 000 generations

All runs use `random.seed(1)` for reproducibility.

## Final configuration

| Component | Choice |
|-----------|--------|
| Population size | 100 |
| Generations | 1 000 |
| Selection | Tournament (size 3) |
| Crossover | Two-Point — `xo_prob = 0.9` |
| Mutation | Gaussian — `mut_prob = 0.005` |
| Elitism | Off (~1 RMSE better than on at 1 000 generations) |
| Fitness | RMSE (minimisation) |

## Key findings

- **`mut_prob` is the single most impactful hyperparameter.** The original default of `0.1` mutates ~100 of the 1 000 genes per offspring, overwhelming selection. Lowering it to `0.005` (~5 genes per offspring) lets good triangles propagate across generations and was the largest single improvement.
- **Two-point crossover was chosen over one-point.** Both preserve complete triangles. Multi-seed experiments showed two-point is more robust; on this single seed the two are within noise of each other, but two-point is the validated choice.
- **Gaussian mutation dominated uniform mutation** by a wide margin — small perturbations let selection accumulate improvements, while uniform mutation effectively randomises good triangles.
- **Ablation studies can hide operator interactions.** Earlier multi-seed exploration on a separate branch showed that the "best" operator depends on the others (e.g. arithmetic crossover wins at high `mut_prob` but loses at low `mut_prob`). The chosen configuration was validated across multiple seeds before being pinned to `seed = 1` in this clean version.

## Running

1. Install dependencies (see below).
2. Open `Computacional_Intelligence_for_Optimization/girl_with_pearl.ipynb` and run the cells top-to-bottom.

### Requirements

- Python 3.10+
- `numpy`
- `scipy`
- `opencv-python` (provides `cv2`)
- `matplotlib`
- `pillow` (provides `PIL`)
- `jupyter` (or `jupyterlab` / `notebook`) to run the `.ipynb`

Install with:

```bash
pip install numpy scipy opencv-python matplotlib pillow jupyter
```
