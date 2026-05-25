"""Pre-compute mean Gram matrix statistics of Minecraft tiles for style loss."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_gram_stats(tile_dir: str, batch_size: int = 64, device: str = "cuda"):
    """Compute mean Gram matrices for all Minecraft tiles at VGG style layers."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.dataset import MinecraftTileBank
    from src.losses import VGGFeatures, gram_matrix

    bank = MinecraftTileBank(tile_dir)
    loader = DataLoader(bank, batch_size=batch_size, shuffle=False, num_workers=0)

    vgg = VGGFeatures().to(device).eval()
    style_layers = ["relu1_1", "relu2_1", "relu3_1"]

    # Accumulate sum of Gram matrices
    gram_sums = {layer: None for layer in style_layers}
    n_tiles = 0

    print(f"Computing Gram stats for {len(bank)} tiles...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            # Upscale 16×16 tiles to 64×64 for VGG features (need minimum size)
            batch_up = F.interpolate(batch, size=(64, 64), mode="nearest")
            feats = vgg(batch_up, style_layers)

            for layer in style_layers:
                G = gram_matrix(feats[layer])  # (B, C, C)
                G_mean = G.mean(dim=0)  # (C, C)
                if gram_sums[layer] is None:
                    gram_sums[layer] = G_mean * len(batch)
                else:
                    gram_sums[layer] += G_mean * len(batch)
            n_tiles += len(batch)

    # Average
    gram_stats = {layer: gram_sums[layer] / n_tiles for layer in style_layers}
    return gram_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-dir", help="Path to Minecraft tiles")
    parser.add_argument("-o", "--output", help="Output .pt path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    tile_dir = args.tile_dir or str(root / "Datasets" / "MinecraftImage" / "tiles")
    output = args.output or str(root / "Datasets" / "style" / "minecraft_gram_stats.pt")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    stats = compute_gram_stats(tile_dir, device=args.device)
    torch.save(stats, output)

    for layer, g in stats.items():
        print(f"  {layer}: Gram shape {tuple(g.shape)}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
