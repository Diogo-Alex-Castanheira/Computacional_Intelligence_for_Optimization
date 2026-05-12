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

    def random_initial_representation(self, vertex_jitter_frac=0.05): # fraction of image size for vertex jitter
        H, W = self._target_array.shape[:2] # note: shape is (H, W, 3) so height is first
        jitter_x = vertex_jitter_frac * W # fraction of image width for vertex jitter
        jitter_y = vertex_jitter_frac * H # fraction of image height for vertex jitter

        repr = []
        for _ in range(self.n_triangles): # for each triangle, we generate a random center point and then jitter the vertices around it
            cx = random.uniform(0, W) # random center x
            cy = random.uniform(0, H) # random center y

            for _ in range(3): # for each of the 3 vertices, we jitter around the center point and clamp to image bounds
                vx = min(W, max(0.0, cx + random.uniform(-jitter_x, jitter_x))) # jitter x around center and clamp to [0, W]
                vy = min(H, max(0.0, cy + random.uniform(-jitter_y, jitter_y))) # jitter y around center and clamp to [0, H]
                repr += [vx, vy] # add vertex coordinates to representation

            px = min(W - 1, int(cx)) # pixel x for color sampling, clamped to [0, W-1]
            py = min(H - 1, int(cy)) # pixel y for color sampling, clamped to [0, H-1]
            r, g, b = self._target_array[py, px][:3] # sample color from target image at the center point (note: array indexing is [y, x])
            repr += [float(r), float(g), float(b)] # add color to representation as floats for mutation

            repr.append(random.random())  # alpha is initialized randomly in [0, 1]

        return repr

    def fitness(self):
        """
        We are going to calculate the RMSE between the reproduced iamge and the target image

        Returns:
            RMSE (float): The root mean squared error between the rendered image and the target image.
        """
        if self._fitness is not None:
            return self._fitness
        rendered = self.render().astype(np.float32)
        target = self._target_array.astype(np.float32)
        self._fitness = float(np.sqrt(np.mean((rendered - target) ** 2)))
        return self._fitness

    def render(self):
        """Render the genotype to an RGB uint8 array using OpenCV.

        For each triangle we draw a solid-colour copy of the canvas and
        then blend it back via cv2.addWeighted with the triangle's alpha:
            canvas = alpha * overlay + (1 - alpha) * canvas
        which is exactly per-triangle alpha compositing.
        """
        W, H = self.img_width, self.img_height # weight and height of the target image
        canvas = np.zeros((H, W, 3), dtype=np.uint8) # start with a blank canvas

        for i in range(self.n_triangles): # for each triangle, we extract the vertex coordinates, color, and alpha from the representation and draw it on the canvas
            base = i * genes_per_triangle # base index for the current triangle in the representation

            pts = np.array([
                [int(self.repr[base + 0]), int(self.repr[base + 1])],
                [int(self.repr[base + 2]), int(self.repr[base + 3])],
                [int(self.repr[base + 4]), int(self.repr[base + 5])],
            ], dtype=np.int32) # extract vertex coordinates and convert to int for OpenCV

            color = (
                int(self.repr[base + 6]),
                int(self.repr[base + 7]),
                int(self.repr[base + 8]),
            ) # extract color and convert to int for OpenCV
            
            alpha = float(self.repr[base + 9]) # extract alpha as float

            overlay = canvas.copy() # create a copy of the current canvas to draw the triangle on
            cv2.fillPoly(overlay, [pts], color) # draw the triangle on the overlay canvas with the specified color
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, dst=canvas) # blend the overlay back onto the canvas using the triangle's alpha

        return canvas

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