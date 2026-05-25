"""U-Net generator: Encoder → Bottleneck → Decoder with skip connections."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + r, inplace=True)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res1 = ResBlock(in_ch)
        self.res2 = ResBlock(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False)

    def forward(self, x):
        x = self.res2(self.res1(x))
        identity = self.skip(x)
        return F.relu(self.bn(self.conv(x)) + identity, inplace=True)


class Generator(nn.Module):
    """U-Net generator with PixelShuffle upsampling and skip connections."""

    def __init__(self, in_ch=3, out_ch=3, base_ch=64):
        super().__init__()
        c = base_ch
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, 7, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )

        # Encoder: 64→128→256→512
        self.enc1 = EncoderBlock(c, c * 2)     # 64→128, /2
        self.enc2 = EncoderBlock(c * 2, c * 4)  # 128→256, /4
        self.enc3 = EncoderBlock(c * 4, c * 8)  # 256→512, /8
        self.enc4 = EncoderBlock(c * 8, c * 8)  # 512→512, /16

        # Bottleneck
        self.bottleneck = nn.Sequential(*[ResBlock(c * 8) for _ in range(6)])

        # Decoder: in_ch → mid_ch output, concat projected skip, fuse to mid_ch
        # (in_ch, mid_ch, skip_ch_raw)
        self.dec3 = self._make_decoder(c * 8, c * 4, c * 8)  # 512→256 out
        self.dec2 = self._make_decoder(c * 4, c * 2, c * 4)  # 256→128 out
        self.dec1 = self._make_decoder(c * 2, c, c * 2)      # 128→64 out
        self.dec0 = self._make_decoder(c, c, c)              # 64→64 out

        self.head = nn.Sequential(
            nn.Conv2d(c, out_ch, 7, padding=3),
            nn.Sigmoid(),
        )

    def _make_decoder(self, in_ch, mid_ch, skip_ch):
        """Build decoder stage.

        in_ch: input channels (from previous decoder or bottleneck)
        mid_ch: output channels after this stage
        skip_ch: raw skip connection channels (before projection)
        """
        return nn.ModuleDict({
            "up": nn.Sequential(
                nn.Conv2d(in_ch, mid_ch * 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(mid_ch * 4),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True),
            ),
            "skip_proj": nn.Conv2d(skip_ch, mid_ch, 1, bias=False),
            "fuse": nn.Conv2d(mid_ch * 2, mid_ch, 1, bias=False),
            "res": nn.Sequential(ResBlock(mid_ch), ResBlock(mid_ch)),
        })

    def _decode(self, dec, x, skip):
        x = dec["up"](x)                          # (B, mid_ch, 2H, 2W)
        skip_proj = dec["skip_proj"](skip)        # (B, mid_ch, 2H, 2W)
        x = torch.cat([x, skip_proj], dim=1)      # (B, 2*mid_ch, 2H, 2W)
        x = dec["fuse"](x)                        # (B, mid_ch, 2H, 2W)
        return dec["res"](x)

    def forward(self, x):
        s0 = self.stem(x)          # 64ch, 256×256
        s1 = self.enc1(s0)         # 128ch, 128×128
        s2 = self.enc2(s1)         # 256ch, 64×64
        s3 = self.enc3(s2)         # 512ch, 32×32
        s4 = self.enc4(s3)         # 512ch, 16×16

        b = self.bottleneck(s4)    # 512ch, 16×16

        d3 = self._decode(self.dec3, b, s3)    # 256ch, 32×32
        d2 = self._decode(self.dec2, d3, s2)   # 128ch, 64×64
        d1 = self._decode(self.dec1, d2, s1)   # 64ch, 128×128
        d0 = self._decode(self.dec0, d1, s0)   # 64ch, 256×256

        return self.head(d0)


def _test_generator():
    g = Generator(base_ch=64)
    x = torch.randn(2, 3, 256, 256)
    y = g(x)
    print(f"Generator: {x.shape} → {y.shape}, params={sum(p.numel() for p in g.parameters()):,}")


if __name__ == "__main__":
    _test_generator()
