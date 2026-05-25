"""Loss functions for unpaired photo-to-pixel-art translation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ─── VGG Feature Extractor ────────────────────────────────────────────

class VGGFeatures(nn.Module):
    """Pretrained VGG-19 with access to intermediate feature maps."""

    # Layer indices in VGG-19 features sequential (0-indexed)
    LAYER_IDS = {
        "relu1_1": 1,   # after 1st ReLU
        "relu2_1": 6,   # after 2nd group ReLU
        "relu3_1": 11,  # after 3rd group ReLU
        "relu4_1": 20,  # after 4th group ReLU
        "relu4_2": 22,  # after 5th group, 2nd ReLU
        "relu5_1": 29,  # after 5th group ReLU
    }

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1).features
        self.layers = nn.ModuleList([*vgg])
        for p in self.parameters():
            p.requires_grad = False
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x, layers):
        x = self.normalize(x)
        feats = {}
        target_set = set(layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            for name, idx in self.LAYER_IDS.items():
                if i == idx and name in target_set:
                    feats[name] = x
        return feats


# ─── Content Loss ─────────────────────────────────────────────────────

def content_loss_from_feats(feats_out: dict, feats_in: dict,
                            layer: str = "relu4_2") -> torch.Tensor:
    return F.l1_loss(feats_out[layer], feats_in[layer])


def content_loss(vgg: VGGFeatures, output: torch.Tensor, input_photo: torch.Tensor,
                 layer: str = "relu4_2") -> torch.Tensor:
    feats_out = vgg(output, [layer])
    feats_in = vgg(input_photo, [layer])
    return F.l1_loss(feats_out[layer], feats_in[layer])


# ─── Patch Style Loss ─────────────────────────────────────────────────

def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """x: (B, C, H, W) → (B, C, C) normalized Gram matrix."""
    B, C, H, W = x.shape
    feat = x.view(B, C, H * W)
    G = feat @ feat.transpose(1, 2)  # (B, C, C)
    return G / (C * H * W)


def style_patch_loss_from_feats(feats_out: dict, target_gram_stats: dict,
                                layers=("relu1_1", "relu2_1", "relu3_1"),
                                n_patches: int = 8) -> torch.Tensor:
    """Gram matrix loss on random patches from pre-computed VGG features."""
    B = next(iter(feats_out.values())).shape[0]
    loss = 0.0
    for layer in layers:
        feats = feats_out[layer]  # (B, C, fH, fW)
        _, C, fH, fW = feats.shape
        p_h = max(1, fH // 4)
        p_w = max(1, fW // 4)

        for _ in range(n_patches):
            y = torch.randint(0, max(1, fH - p_h + 1), (1,)).item()
            x = torch.randint(0, max(1, fW - p_w + 1), (1,)).item()
            patch = feats[:, :, y:y + p_h, x:x + p_w]
            G = gram_matrix(patch)
            target = target_gram_stats[layer].to(G.device)
            loss += F.mse_loss(G, target.unsqueeze(0).expand(B, -1, -1))

    return loss / (len(layers) * n_patches)


# ─── Edge Loss ─────────────────────────────────────────────────────────

def _laplacian(x: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


def edge_loss(output: torch.Tensor, input_photo: torch.Tensor) -> torch.Tensor:
    """Laplacian edge preservation — keeps sharp boundaries from input."""
    edge_out = _laplacian(output)
    edge_in = _laplacian(input_photo)
    return F.l1_loss(edge_out, edge_in)


# ─── TV Loss ───────────────────────────────────────────────────────────

def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation regularization."""
    d_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    d_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return d_h + d_w


# ─── Palette Entropy Loss ─────────────────────────────────────────────

def palette_entropy_loss(soft_weights: torch.Tensor) -> torch.Tensor:
    """Entropy regularization: maximize palette usage diversity.

    soft_weights: (N, palette_size) soft assignment probabilities from PaletteQuantizer.
    Returns negative mean entropy — minimizing this maximizes diversity.
    """
    mean_usage = soft_weights.mean(dim=0)  # (K,)
    mean_usage = mean_usage / (mean_usage.sum() + 1e-8)
    entropy = -(mean_usage * (mean_usage + 1e-8).log()).sum()
    return -entropy  # negative → minimizing = maximizing diversity


# ─── Adversarial Loss ──────────────────────────────────────────────────

def generator_hinge_loss(fake_logits: list) -> torch.Tensor:
    """Hinge loss for generator: -E[D(G(z))]."""
    loss = 0.0
    for logit in fake_logits:
        loss += -logit.mean()
    return loss / len(fake_logits)


def discriminator_hinge_loss(real_logits: list, fake_logits: list) -> torch.Tensor:
    """Hinge loss for discriminator."""
    loss = 0.0
    for real, fake in zip(real_logits, fake_logits):
        loss += F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss / len(real_logits)


# ─── R1 Gradient Penalty ───────────────────────────────────────────────

def r1_penalty(discriminator: nn.Module, real_samples: list) -> torch.Tensor:
    """R1 gradient penalty for discriminator regularization."""
    grad_penalty = 0.0
    for real in real_samples:
        real.requires_grad_(True)
        logits = discriminator(real)
        if isinstance(logits, list):
            logits = logits[0]
        grad = torch.autograd.grad(
            outputs=logits.sum(), inputs=real,
            create_graph=True, retain_graph=True,
        )[0]
        grad_penalty += grad.view(grad.shape[0], -1).pow(2).sum(dim=1).mean()
    return grad_penalty / len(real_samples)


# ─── Combined Loss ─────────────────────────────────────────────────────

class LossManager:
    def __init__(self, device="cuda"):
        self.vgg = VGGFeatures().to(device).eval()
        self.device = device
        self.gram_stats = None  # set after computing style stats

    def set_gram_stats(self, gram_stats: dict):
        self.gram_stats = {k: v.to(self.device) for k, v in gram_stats.items()}

    def compute_g_losses(self, output, input_photo, fake_logits, soft_weights=None, phase=2):
        losses = {}

        # Single VGG forward pass for each image
        vgg_layers = ["relu4_2", "relu1_1", "relu2_1", "relu3_1"]
        feats_out = self.vgg(output, vgg_layers)
        feats_in = self.vgg(input_photo, vgg_layers)

        losses["content"] = content_loss_from_feats(feats_out, feats_in)
        losses["edge"] = edge_loss(output, input_photo)
        losses["tv"] = tv_loss(output)

        if phase >= 2:
            losses["adv"] = generator_hinge_loss(fake_logits)
            if self.gram_stats is not None:
                losses["style_patch"] = style_patch_loss_from_feats(
                    feats_out, self.gram_stats
                )

        if soft_weights is not None:
            losses["palette"] = palette_entropy_loss(soft_weights)

        return losses

    def compute_d_losses(self, real_logits, fake_logits):
        return {"d_adv": discriminator_hinge_loss(real_logits, fake_logits)}
