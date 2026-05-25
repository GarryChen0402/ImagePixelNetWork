"""Split Collections.png into 16x16 pixel art tiles for dataset."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def split_tiles(
    image_path: str,
    output_dir: str,
    tile_size: int = 16,
    skip_transparent: bool = False,
    skip_empty_threshold: float = 0.95,
):
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    if w % tile_size != 0 or h % tile_size != 0:
        raise ValueError(
            f"Image size {w}×{h} not divisible by tile size {tile_size}"
        )

    cols = w // tile_size
    rows = h // tile_size
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arr = np.array(img)
    tiles = []
    saved = 0
    skipped = 0

    for row in tqdm(range(rows), desc="Splitting rows"):
        for col in range(cols):
            y1, y2 = row * tile_size, (row + 1) * tile_size
            x1, x2 = col * tile_size, (col + 1) * tile_size
            tile = arr[y1:y2, x1:x2]

            if skip_transparent:
                alpha = tile[..., 3]
                if (alpha == 0).mean() > skip_empty_threshold:
                    skipped += 1
                    continue

            tile_img = Image.fromarray(tile, mode="RGBA")
            tile_img.save(out / f"tile_{row:04d}_{col:04d}.png")
            saved += 1

    print(f"Done. {saved} tiles saved to {out.resolve()}", end="")
    if skipped:
        print(f" ({skipped} skipped as transparent)")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Split a sprite sheet into 16x16 pixel art tiles"
    )
    parser.add_argument("input", nargs="?", help="Path to the sprite sheet image")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument(
        "-s", "--tile-size", type=int, default=16, help="Tile size in pixels"
    )
    parser.add_argument(
        "--skip-transparent",
        action="store_true",
        help="Skip tiles that are mostly transparent",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    default_input = root / "Datasets" / "MinecraftImage" / "Collections.png"
    default_output = root / "Datasets" / "MinecraftImage" / "tiles"

    split_tiles(
        image_path=args.input or str(default_input),
        output_dir=args.output or str(default_output),
        tile_size=args.tile_size,
        skip_transparent=args.skip_transparent,
    )


if __name__ == "__main__":
    main()
