"""Extract Minecraft color palette via k-means on all tiles."""

import argparse
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.palette import extract_palette_from_tiles


def main():
    parser = argparse.ArgumentParser(description="Extract k-means palette from Minecraft tiles")
    parser.add_argument("--tile-dir", help="Path to tile directory")
    parser.add_argument("--palette-size", type=int, default=64)
    parser.add_argument("-o", "--output", help="Output .npy path")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    tile_dir = args.tile_dir or str(root / "Datasets" / "MinecraftImage" / "tiles")
    output = args.output or str(root / "Datasets" / "palette" / "minecraft_64.npy")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    centers = extract_palette_from_tiles(tile_dir, args.palette_size)
    np.save(output, centers)
    print(f"Palette ({args.palette_size} colors) saved to {output}")


if __name__ == "__main__":
    main()
