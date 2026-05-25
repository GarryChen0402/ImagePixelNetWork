"""Unpaired dataset: landscape photos (Domain A) and Minecraft tiles (Domain B)."""

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PhotoDataset(Dataset):
    """Landscape photos — Domain A."""

    def __init__(self, root: str, image_size: int = 256):
        self.root = Path(root)
        self.paths = sorted(
            p for p in self.root.glob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


class MinecraftTileBank(Dataset):
    """Minecraft 16×16 texture tiles — Domain B.

    Returns raw tiles for the discriminator (16×16) and also provides
    batch sampling for training.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.png"))
        self.tiles = []
        for p in self.paths:
            img = Image.open(p).convert("RGBA")
            arr = np.array(img).astype(np.float32) / 255.0
            rgb = arr[..., :3]
            alpha = arr[..., 3:4]
            # Composite over white background for RGB-only tiles
            rgb = rgb * alpha + (1 - alpha)
            self.tiles.append(torch.from_numpy(rgb).permute(2, 0, 1))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        return self.tiles[idx]

    def sample(self, n: int, device="cpu") -> torch.Tensor:
        """Sample n random tiles as a batch (with augmentation)."""
        idxs = random.choices(range(len(self.tiles)), k=n)
        batch = []
        for i in idxs:
            tile = self.tiles[i].clone()
            # Random flip
            if random.random() < 0.5:
                tile = tile.flip(-1)
            if random.random() < 0.5:
                tile = tile.flip(-2)
            # Random 90° rotation
            k = random.randint(0, 3)
            if k > 0:
                tile = tile.rot90(k, dims=[1, 2])
            # Slight color jitter
            if random.random() < 0.5:
                tile = tile + torch.randn(3, 1, 1) * 0.02
                tile = tile.clamp(0, 1)
            batch.append(tile)
        return torch.stack(batch).to(device)

    def sample_upscaled(self, n: int, scale: int = 2, device="cpu") -> torch.Tensor:
        """Sample tiles upscaled to 16*scale × 16*scale (for multi-scale D)."""
        tiles = self.sample(n, device)
        if scale > 1:
            tiles = torch.nn.functional.interpolate(
                tiles, scale_factor=scale, mode="nearest"
            )
        return tiles
