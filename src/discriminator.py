"""PatchGAN discriminator — classifies 16×16 patches as real (Minecraft) or fake."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchDiscriminator(nn.Module):
    """Discriminator that operates on 16×16 patches.

    Architecture: 5 conv layers with stride 2, receptive field ~16×16.
    Input: (B, 3, 16, 16)
    Output: (B, 1) logit
    """

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        c = base_ch
        self.layers = nn.Sequential(
            # 16→8
            nn.Conv2d(in_ch, c, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 8→4
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 4→2
            nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 2→1
            nn.Conv2d(c * 4, 1, 2, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x).view(x.shape[0], -1).mean(dim=1, keepdim=True)


class MultiScaleDiscriminator(nn.Module):
    """Two discriminators: one for 16×16 patches, one for 32×32 upscaled."""

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.d16 = PatchDiscriminator(in_ch, base_ch)
        self.d32 = PatchDiscriminator(in_ch, base_ch)

    def forward(self, x16, x32=None):
        """x16: (B, 3, 16, 16), x32: (B, 3, 32, 32) or None"""
        out16 = self.d16(x16)
        if x32 is not None:
            out32 = self.d32(x32)
            return [out16, out32]
        return [out16]


def _test_discriminator():
    d = PatchDiscriminator()
    x = torch.randn(4, 3, 16, 16)
    y = d(x)
    print(f"PatchD: {x.shape} → {y.shape}, params={sum(p.numel() for p in d.parameters()):,}")
    d2 = MultiScaleDiscriminator()
    x32 = torch.randn(4, 3, 32, 32)
    y2 = d2(x, x32)
    print(f"MultiScaleD: outputs={[o.shape for o in y2]}")


if __name__ == "__main__":
    _test_discriminator()
