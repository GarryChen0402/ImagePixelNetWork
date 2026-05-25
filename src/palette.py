"""Differentiable palette color quantization for pixel art."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaletteQuantizer(nn.Module):
    """Maps continuous RGB values to a learnable discrete palette via soft assignment."""

    def __init__(self, palette_size: int = 64, temperature: float = 1.0):
        super().__init__()
        self.palette_size = palette_size
        self.temperature = temperature
        palette = torch.randn(palette_size, 3) * 0.1
        self.palette = nn.Parameter(palette)

    def init_from_kmeans(self, colors_float: np.ndarray):
        """Initialize palette from pre-computed k-means cluster centers (RGB, [0,1])."""
        assert len(colors_float) == self.palette_size
        with torch.no_grad():
            self.palette.copy_(torch.from_numpy(colors_float).float())

    def forward(self, x: torch.Tensor, hard: bool = False, return_weights: bool = False):
        """x: (B, 3, H, W) RGB in [0,1].

        Returns:
            quantized (B, 3, H, W) — always returned
            weights (B*H*W, K) — returned only if return_weights=True and not hard
        """
        B, C, H, W = x.shape
        pixels = x.permute(0, 2, 3, 1).reshape(-1, 3)  # (N, 3)

        dist = torch.cdist(pixels, self.palette)  # (N, K)
        if hard:
            idx = dist.argmin(dim=1)
            quantized = self.palette[idx]
            weights = None
        else:
            weights = F.softmax(-dist / self.temperature, dim=1)  # (N, K)
            quantized = weights @ self.palette  # (N, 3)

        out = quantized.reshape(B, H, W, C).permute(0, 3, 1, 2)
        if return_weights and not hard:
            return out, weights
        return out


def extract_palette_from_tiles(tile_dir: str, palette_size: int = 64) -> np.ndarray:
    """Run k-means on all Minecraft tiles to get initial palette colors."""
    import os
    from pathlib import Path

    from PIL import Image
    from sklearn.cluster import KMeans

    tiles = list(Path(tile_dir).glob("*.png"))
    if not tiles:
        raise FileNotFoundError(f"No PNG tiles found in {tile_dir}")

    all_colors = []
    for path in tiles:
        img = Image.open(path).convert("RGBA")
        arr = np.array(img).astype(np.float32) / 255.0
        alpha = arr[..., 3]
        mask = alpha > 0.05
        rgb = arr[mask, :3]
        if len(rgb) > 0:
            all_colors.append(rgb)

    pixels = np.concatenate(all_colors, axis=0)
    print(f"Clustering {len(pixels):,} pixels into {palette_size} colors...")
    kmeans = KMeans(n_clusters=palette_size, random_state=42, n_init=10)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_.astype(np.float32)
