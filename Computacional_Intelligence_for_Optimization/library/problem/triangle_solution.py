"""Problem Definition
Description: Modern Interpretation of the painting Vermeer's Girl with a Pearl Earing.
The objective is to use 100 semi-transparent triangles to create a painting that's closely
similar to the original.

Genome layout (10 genes per triangle):
    [x1, y1, x2, y2, x3, y3, r, g, b, α]
Ranges:
    x* ∈ [0, img_width]
    y* ∈ [0, img_height]
    r, g, b ∈ [0, 255]
    α ∈ [0, 1]
"""

import numpy as np
import random
import cv2

from library.problem.solution import Solution

genes_per_triangle = 10


class TriangleSolution(Solution):
    def __init__(
        self,
        target_array,
        img_width,
        img_height,
        n_triangles,
        repr=None,
    ):
        self._target_array = target_array   # RGB, shape (H, W, 3), uint8
        self.img_width = img_width
        self.img_height = img_height
        self.n_triangles = n_triangles
        super().__init__(repr=repr)

    def random_initial_representation(self):
        """
        Initialise each triangle with three independent random vertices anywhere on the canvas.
        Colour is sampled from the target image at the centroid of the three vertices,
        and alpha is initialised uniformly in [0, 1]. This gives a natural mix of triangle
        sizes (some small, some large) and much higher initial canvas coverage than a
        center-and-jitter scheme.
        """
        H, W = self._target_array.shape[:2]

        genes = []
        for _ in range(self.n_triangles):
            # Three independent random vertices anywhere on the canvas.
            x1, y1 = random.uniform(0, W), random.uniform(0, H)
            x2, y2 = random.uniform(0, W), random.uniform(0, H)
            x3, y3 = random.uniform(0, W), random.uniform(0, H)
            genes += [x1, y1, x2, y2, x3, y3]

            # Sample colour at the centroid of the triangle.
            cx, cy = (x1 + x2 + x3) / 3.0, (y1 + y2 + y3) / 3.0
            px = min(W - 1, int(cx))
            py = min(H - 1, int(cy))
            r, g, b = self._target_array[py, px][:3]
            genes += [float(r), float(g), float(b)]

            genes.append(random.random())  # alpha

        return genes

    def fitness(self):
        """
        We are going to calculate the RMSE between the reproduced iamge and the target image

        Returns:
            RMSE (float): The root mean squared error between the rendered image and the target image.
        """
        if self._fitness is not None:
            return self._fitness
        rendered = self._render_f32()               # stay in float32 — no quantisation error
        target = self._target_array.astype(np.float32)
        self._fitness = float(np.sqrt(np.mean((rendered - target) ** 2)))
        return self._fitness

    def _render_f32(self):
        """Render to a float32 canvas in [0, 255]. Used internally by fitness() and render()."""
        W, H = self.img_width, self.img_height
        canvas = np.zeros((H, W, 3), dtype=np.float32) # float32 avoids uint8 rounding error across 100 blends
        overlay = np.empty_like(canvas)                 # allocated once; reused each iteration

        for i in range(self.n_triangles):
            base = i * genes_per_triangle

            pts = np.array([
                [int(self.repr[base + 0]), int(self.repr[base + 1])],
                [int(self.repr[base + 2]), int(self.repr[base + 3])],
                [int(self.repr[base + 4]), int(self.repr[base + 5])],
            ], dtype=np.int32)

            color = (
                float(self.repr[base + 6]),
                float(self.repr[base + 7]),
                float(self.repr[base + 8]),
            )

            alpha = float(self.repr[base + 9])

            np.copyto(overlay, canvas)
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas)

        return canvas

    def render(self):
        """Render the genotype to an RGB uint8 array using OpenCV.

        For each triangle we draw a solid-colour copy of the canvas and
        then blend it back via cv2.addWeighted with the triangle's alpha:
            canvas = alpha * overlay + (1 - alpha) * canvas
        which is exactly per-triangle alpha compositing.
        """
        return np.clip(self._render_f32(), 0, 255).astype(np.uint8) # quantise once, only for display

    def with_repr(self, new_repr):
        return TriangleSolution(
            target_array=self._target_array,
            img_width=self.img_width,
            img_height=self.img_height,
            n_triangles=self.n_triangles,
            repr=list(new_repr),
        )

    def clone(self):
        twin = TriangleSolution(
            target_array=self._target_array,
            img_width=self.img_width,
            img_height=self.img_height,
            n_triangles=self.n_triangles,
            repr=self.repr.copy(),
        ) # clone the representation list to ensure independence
        twin._fitness = self._fitness
        return twin