"""Three-phase adversarial training for Photo → Pixel Art translation."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import MinecraftTileBank, PhotoDataset
from discriminator import MultiScaleDiscriminator
from generator import Generator
from losses import LossManager, r1_penalty
from palette import PaletteQuantizer


def crop_patches(x: torch.Tensor, patch_size: int, n: int):
    """Randomly crop n patches of patch_size×patch_size from x (B, C, H, W)."""
    B, C, H, W = x.shape
    patches = []
    for _ in range(n):
        y = random.randint(0, H - patch_size)
        x0 = random.randint(0, W - patch_size)
        patches.append(x[:, :, y:y + patch_size, x0:x0 + patch_size])
    return torch.cat(patches, dim=0)  # (B*n, C, patch, patch)


def get_temperature(epoch: int, phase: int) -> float:
    if phase == 1:
        return 1.0
    elif phase == 2:
        progress = min(1.0, (epoch - 50) / 150)  # epoch 51→200
        return max(0.3, 1.0 * np.exp(-1.2 * progress))
    else:
        progress = min(1.0, (epoch - 200) / 100)  # epoch 201→300
        return max(0.1, 0.3 * np.exp(-1.5 * progress))


def get_phase(epoch: int) -> int:
    if epoch <= 50:
        return 1
    elif epoch <= 200:
        return 2
    else:
        return 3


def get_loss_weights(epoch: int, phase: int):
    w = {"content": 1.0, "edge": 0.5, "tv": 1e-4, "palette": 0.1}
    if phase == 1:
        w["adv"] = 0.0
        w["style_patch"] = 0.0
    elif phase == 2:
        progress = min(1.0, (epoch - 50) / 80)
        w["adv"] = 0.5 * progress
        w["style_patch"] = 3.0 * progress
    else:
        w["adv"] = 0.5
        w["style_patch"] = 3.0
    return w


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──
    root = Path(__file__).resolve().parent.parent
    photo_dir = args.photo_dir or str(root / "Datasets" / "SceneImage" / "landscape_dataset")
    tile_dir = args.tile_dir or str(root / "Datasets" / "MinecraftImage" / "tiles")
    palette_path = args.palette or str(root / "Datasets" / "palette" / "minecraft_64.npy")
    gram_path = args.gram_stats or str(root / "Datasets" / "style" / "minecraft_gram_stats.pt")

    photo_ds = PhotoDataset(photo_dir, image_size=args.image_size)
    tile_bank = MinecraftTileBank(tile_dir)
    photo_loader = DataLoader(
        photo_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    print(f"Photos: {len(photo_ds)}, Tiles: {len(tile_bank)}")

    # ── Models ──
    g = Generator(base_ch=args.base_ch).to(device)
    d = MultiScaleDiscriminator(base_ch=args.d_ch).to(device)
    pq = PaletteQuantizer(palette_size=args.palette_size, temperature=1.0).to(device)

    if Path(palette_path).exists():
        centers = np.load(palette_path)
        pq.init_from_kmeans(centers)
        print(f"Palette initialized from {palette_path}")

    g_opt = torch.optim.Adam(g.parameters(), lr=args.lr_g, betas=(0.9, 0.999))
    d_opt = torch.optim.Adam(d.parameters(), lr=args.lr_d, betas=(0.9, 0.999))
    pq_opt = torch.optim.Adam(pq.parameters(), lr=args.lr_g * 0.1, betas=(0.9, 0.999))

    # ── Loss Manager ──
    loss_mgr = LossManager(device=str(device))
    if Path(gram_path).exists():
        gram_stats = torch.load(gram_path, map_location=device, weights_only=True)
        loss_mgr.set_gram_stats(gram_stats)
        print(f"Gram stats loaded from {gram_path}")

    # ── Logging ──
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "logs")
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ── AMP ──
    scaler_g = torch.amp.GradScaler("cuda") if args.fp16 else None
    scaler_d = torch.amp.GradScaler("cuda") if args.fp16 else None

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        phase = get_phase(epoch)
        tau = get_temperature(epoch, phase)
        pq.temperature = tau
        weights = get_loss_weights(epoch, phase)

        # Update LR for phase transitions
        if epoch == 1:
            for pg in g_opt.param_groups:
                pg["lr"] = args.lr_g * 10  # Phase 1 LR = 1e-3
        elif epoch == 51:
            for pg in g_opt.param_groups:
                pg["lr"] = args.lr_g  # Phase 2 LR = 1e-4
            for pg in d_opt.param_groups:
                pg["lr"] = args.lr_d
        elif epoch == 201:
            for pg in g_opt.param_groups:
                pg["lr"] = args.lr_g
            for pg in d_opt.param_groups:
                pg["lr"] = args.lr_d * 0.25

        pbar = tqdm(photo_loader, desc=f"E{epoch:03d} P{phase} τ={tau:.2f}")
        epoch_losses = {}

        for photo in pbar:
            photo = photo.to(device)
            B = photo.shape[0]

            # ─── Generator Update ───
            g_opt.zero_grad(set_to_none=True)
            pq_opt.zero_grad(set_to_none=True)

            raw = g(photo)
            if phase >= 2:
                quantized, soft_w = pq(raw, return_weights=True)
            else:
                quantized = pq(raw)
                soft_w = None

            # Discriminator forward (for G loss)
            fake_patches_16 = crop_patches(quantized, 16, 4)  # (B*4, 3, 16, 16)
            fake_logits = d(fake_patches_16)

            g_losses = loss_mgr.compute_g_losses(
                quantized, photo, fake_logits, soft_w, phase,
            )
            g_total = sum(weights.get(k, 0) * v for k, v in g_losses.items())

            if scaler_g:
                scaler_g.scale(g_total).backward()
                scaler_g.step(g_opt)
                scaler_g.step(pq_opt)
                scaler_g.update()
            else:
                g_total.backward()
                g_opt.step()
                pq_opt.step()

            # ─── Discriminator Update ───
            if phase >= 2:
                d_opt.zero_grad(set_to_none=True)

                real_16 = tile_bank.sample(B * 4, device)
                real_logits = d(real_16)

                with torch.no_grad():
                    fake_16 = crop_patches(quantized.detach(), 16, 4)
                fake_logits_d = d(fake_16)

                d_losses = loss_mgr.compute_d_losses(real_logits, fake_logits_d)
                d_total = sum(d_losses.values())

                if scaler_d:
                    scaler_d.scale(d_total).backward()
                else:
                    d_total.backward()

                # R1 penalty every 2 steps
                if global_step % 2 == 0:
                    r1 = r1_penalty(d, [real_16]) * 10.0
                    if scaler_d:
                        scaler_d.scale(r1).backward()
                    else:
                        r1.backward()
                    d_losses["r1"] = r1.detach()

                if scaler_d:
                    scaler_d.step(d_opt)
                    scaler_d.update()
                else:
                    d_opt.step()

            # ─── Logging ───
            all_losses = {**g_losses}
            if phase >= 2:
                all_losses.update(d_losses)
            all_losses["g_total"] = g_total.detach()

            for k, v in all_losses.items():
                epoch_losses.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )

            pbar.set_postfix(
                g=f"{g_total.item():.3f}",
                d=f"{all_losses.get('d_adv', 0):.3f}",
                content=f"{g_losses.get('content', 0):.3f}",
            )

            global_step += 1

        # ─── End of Epoch ───
        for k, vals in epoch_losses.items():
            writer.add_scalar(f"loss/{k}", np.mean(vals), epoch)
        writer.add_scalar("params/tau", tau, epoch)
        writer.add_scalar("params/phase", phase, epoch)

        print(f"Epoch {epoch:03d} | G={np.mean(epoch_losses['g_total']):.4f} "
              f"Content={np.mean(epoch_losses.get('content',[0])):.4f} "
              f"D={np.mean(epoch_losses.get('d_adv',[0])):.4f} "
              f"τ={tau:.2f}")

        # Save checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "phase": phase,
                "g": g.state_dict(),
                "d": d.state_dict(),
                "pq": pq.state_dict(),
                "g_opt": g_opt.state_dict(),
                "d_opt": d_opt.state_dict(),
                "pq_opt": pq_opt.state_dict(),
            }
            torch.save(ckpt, ckpt_dir / f"ckpt_epoch{epoch:04d}.pt")

        # Save samples
        if epoch % args.sample_every == 0 or epoch == 1:
            sample_dir = out_dir / "samples"
            sample_dir.mkdir(exist_ok=True)
            with torch.no_grad():
                sample_photo = photo[:4].to(device)
                sample_out = pq(g(sample_photo), hard=True)
                grid = torch.cat([sample_photo, sample_out], dim=0)
                from torchvision.utils import save_image
                save_image(grid, sample_dir / f"epoch{epoch:04d}.png", nrow=4)

    writer.close()
    print(f"Training complete. Output in {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Train PatchStyleGAN for pixel art transfer")
    parser.add_argument("--photo-dir", help="Path to landscape photos")
    parser.add_argument("--tile-dir", help="Path to Minecraft texture tiles")
    parser.add_argument("--palette", help="Path to k-means palette .npy")
    parser.add_argument("--gram-stats", help="Path to pre-computed Gram stats .pt")
    parser.add_argument("--output", default="outputs/train", help="Output directory")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr-g", type=float, default=1e-4)
    parser.add_argument("--lr-d", type=float, default=4e-4)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--d-ch", type=int, default=64)
    parser.add_argument("--palette-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
