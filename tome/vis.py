# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from scipy.ndimage import binary_erosion
except ImportError:
    pass  # Don't fail if scipy is not installed. It's only necessary for this one file.


def generate_colormap(N: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """Generates a equidistant colormap with N elements."""
    random.seed(seed)

    def generate_color():
        return (random.random(), random.random(), random.random())

    return [generate_color() for _ in range(N)]


def make_visualization(
    img: Image, source: torch.Tensor, patch_size: int = 16, class_token: bool = True
) -> Image:
    """
    Create a visualization like in the paper.

    Args:
     -

    Returns:
     - A PIL image the same size as the input.
    """

    img = np.array(img.convert("RGB")) / 255.0
    source = source.detach().cpu()

    h, w, _ = img.shape
    ph = h // patch_size
    pw = w // patch_size

    if class_token:
        source = source[:, :, 1:]

    vis = source.argmax(dim=1)
    num_groups = vis.max().item() + 1

    cmap = generate_colormap(num_groups)
    vis_img = 0

    for i in range(num_groups):
        mask = (vis == i).float().view(1, 1, ph, pw)
        mask = F.interpolate(mask, size=(h, w), mode="nearest")
        mask = mask.view(h, w, 1).numpy()

        color = (mask * img).sum(axis=(0, 1)) / mask.sum()
        mask_eroded = binary_erosion(mask[..., 0])[..., None]
        mask_edge = mask - mask_eroded

        if not np.isfinite(color).all():
            color = np.zeros(3)

        vis_img = vis_img + mask_eroded * color.reshape(1, 1, 3)
        vis_img = vis_img + mask_edge * np.array(cmap[i]).reshape(1, 1, 3)

    # Convert back into a PIL image
    vis_img = Image.fromarray(np.uint8(vis_img * 255))

    return vis_img
