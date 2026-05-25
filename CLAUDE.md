# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ImagePixelNetWork — deep learning network that converts natural images to pixel art style (Minecraft texture-like). Currently at v0.1 (design phase, no source code yet).

Full design docs live in `Docs/`:
- `Docs/README.md` — project overview, tech stack, directory plan
- `Docs/architecture.md` — network architecture
- `Docs/dataset.md` — dataset strategy
- `Docs/training.md` — training strategy and hyperparameters
- `Docs/CHANGELOG.md` — version history and outstanding tasks

## Tech stack

Python 3.10+, PyTorch 2.0+, torchvision, NumPy, PIL/OpenCV.

## Architecture (planned)

```
Input → Encoder → Bottleneck → Decoder → PaletteQuantizer → Output
```

**Encoder**: ResNet-style CNN, 4 blocks (64→128→256→512 channels), max 8× downsampling. Preserves spatial structure needed for pixel art.

**Bottleneck**: 6 consecutive ResBlocks at 512 channels.

**Decoder**: Progressive upsampling via PixelShuffle + skip connections from encoder blocks. Skip connections use channel-wise concatenation.

**PaletteQuantizer** (core module): Differentiable color quantization via temperature-annealed soft assignment. Maps continuous RGB to a learnable discrete palette (default 32 colors). Training uses softmax(-distance/τ), inference uses argmax for hard assignment. τ anneals from 1.0 → 0.1 over training.

## Loss function

```
L_total = 1.0 × L_content + 10.0 × L_style + 1e-4 × L_tv + 0.1 × L_palette + 0.5 × L_edge
```

- Content: VGG-19 perceptual loss at `relu4_2`
- Style: Gram matrix loss at `relu1_1` through `relu5_1`
- TV: total variation regularization
- Palette: entropy regularization to prevent color collapse
- Edge: Laplacian edge loss for sharp pixel-art boundaries

## Training strategy

Three-phase progressive training over 200 epochs:

| Phase | Epochs | Focus | τ | LR |
|-------|--------|-------|---|---|
| 1 (pretrain) | 1–30 | Content reconstruction + quantization, no style loss | 1.0 | 1e-3 |
| 2 (style injection) | 31–120 | Ramp style loss 0→10, anneal τ 1.0→0.2 | 0.2 | 1e-4 |
| 3 (fine-tune) | 121–200 | All losses active, τ=0.1, cosine LR decay | 0.1 | 1e-4→1e-6 |

Key hyperparameters: image size 256×256, batch size 16, Adam optimizer, cosine annealing, mixed precision (FP16).

Watch for **color collapse** during training — if palette usage entropy drops, increase palette loss weight or lower LR.

## Dataset

Primary: synthetic paired data (10K+ images). High-res natural images → downsample → k-means color quantization → nearest-neighbor upscale → target.

Secondary: real pixel art (Minecraft resource packs, PixelJoint, Lospec, OpenGameArt) for palette initialization and style reference.

Fallback: CycleGAN-style unpaired training if paired data is infeasible.

## Inference

- Hard palette assignment (argmax)
- Optional grid post-processing: split output into 16×16 grids, replace each grid with its mode color to enhance blocky Minecraft feel.
