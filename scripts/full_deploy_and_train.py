#!/usr/bin/env python3
"""
One-click deploy & train on remote GPU server.

Usage:
  python scripts/full_deploy_and_train.py [--run-name run01] [--batch-size 8] [--epochs 300]

What it does:
  1. Package local project into tar.gz
  2. Upload to remote server via SFTP
  3. Install PyTorch + deps in conda env
  4. Precompute palette & Gram style stats on remote
  5. Launch nohup training (survives SSH disconnect)
  6. Print commands to monitor / download results

Requires: pip install paramiko
"""

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import paramiko

# ─── Config ───────────────────────────────────────────────────────────

REMOTE_HOST = "connect.westc.seetacloud.com"
REMOTE_PORT = 48897
REMOTE_USER = "root"
REMOTE_PASS = "BohrzDStaCx5"
REMOTE_ROOT = "/root/ImagePixelNetWork"

# ─── SSH Helpers ──────────────────────────────────────────────────────


class Remote:
    def __init__(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            REMOTE_HOST, port=REMOTE_PORT,
            username=REMOTE_USER, password=REMOTE_PASS,
            timeout=30,
        )
        self.sftp = self.ssh.open_sftp()

    def close(self):
        self.sftp.close()
        self.ssh.close()

    def run(self, cmd, timeout=120):
        """Run command, return (stdout, stderr, exit_code)."""
        stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return out, err, exit_code

    def run_nohup(self, cmd, timeout=5):
        """Run a command that launches a background process (nohup ... &).
        Fires the command and returns immediately.
        """
        transport = self.ssh.get_transport()
        channel = transport.open_session()
        channel.settimeout(timeout)
        channel.exec_command(cmd)
        # Wait briefly for shell to exit (it should, since cmd is backgrounded with &)
        try:
            channel.recv_exit_status()
        except Exception:
            pass  # Channel may stay open; that's fine
        channel.close()

    def run_verbose(self, cmd, timeout=120):
        """Run and print output in real-time."""
        print(f"  $ {cmd[:120]}{'...' if len(cmd) > 120 else ''}")
        stdin, stdout, stderr = self.ssh.exec_command(cmd, timeout=timeout)
        for line in stdout:
            print(f"    {line.rstrip()}")
        for line in stderr:
            print(f"    [E] {line.rstrip()}")
        exit_code = stdout.channel.recv_exit_status()
        if exit_code != 0:
            print(f"  !!! Exit code: {exit_code}")
        return exit_code

    def mkdir(self, path):
        self.run(f"mkdir -p {path}")

    def file_exists(self, path):
        out, _, code = self.run(f"test -f {path} && echo YES || echo NO")
        return "YES" in out

    def dir_exists(self, path):
        out, _, code = self.run(f"test -d {path} && echo YES || echo NO")
        return "YES" in out

    def put_file(self, local, remote):
        """Upload a single file."""
        self.sftp.put(str(local), str(remote))

    def put_dir_tar(self, local_dir, remote_dir, exclude=None):
        """Upload a directory efficiently: tar.gz locally, SFTP, untar remote.
        exclude: list of patterns to exclude.
        remote_dir must be a Linux path string (not Path object on Windows).
        """
        local_path = Path(local_dir)
        remote_dir = str(remote_dir)  # keep as string, don't convert to Path
        self.mkdir(remote_dir)

        # Build tar in memory
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in local_path.rglob("*"):
                if f.is_dir():
                    continue
                rel = f.relative_to(local_path)
                parts = str(rel).replace("\\", "/")
                if exclude:
                    skip = False
                    for pat in exclude:
                        if pat in parts:
                            skip = True
                            break
                    if skip:
                        continue
                tar.add(str(f), arcname=parts)

        buf.seek(0)
        tar_name = f".upload_{int(time.time())}.tar.gz"
        remote_tar = f"{remote_dir}/{tar_name}"
        print(f"  Uploading {len(buf.getvalue())/1024/1024:.1f} MB to {remote_dir}...")

        with self.sftp.file(remote_tar, "wb") as f:
            f.write(buf.read())

        # Extract
        self.run(f"cd {remote_dir} && tar xzf {tar_name} && rm {tar_name}")


# ─── Main Workflow ────────────────────────────────────────────────────


def step_deploy_project(r: Remote, local_root: Path):
    """Package and upload source code + scripts to remote."""
    print("\n" + "=" * 60)
    print("[STEP 1] Deploying project code")
    print("=" * 60)

    r.mkdir(f"{REMOTE_ROOT}/src")
    r.mkdir(f"{REMOTE_ROOT}/scripts")
    r.mkdir(f"{REMOTE_ROOT}/Docs")

    # Upload src/
    print("  Uploading src/ ...")
    r.put_dir_tar(local_root / "src", f"{REMOTE_ROOT}/src",
                  exclude=["__pycache__"])

    # Upload scripts/
    print("  Uploading scripts/ ...")
    r.put_dir_tar(local_root / "scripts", f"{REMOTE_ROOT}/scripts",
                  exclude=["__pycache__", ".pyc"])

    print("  Done.")


def step_upload_datasets(r: Remote, local_root: Path):
    """Upload image datasets + precomputed palette/stats."""
    print("\n" + "=" * 60)
    print("[STEP 2] Uploading datasets")
    print("=" * 60)

    local_ds = local_root / "Datasets"

    # SceneImage
    photo_dir = local_ds / "SceneImage" / "landscape_dataset"
    if photo_dir.is_dir():
        n = len(list(photo_dir.glob("*.jpg")))
        remote_dir = f"{REMOTE_ROOT}/Datasets/SceneImage/landscape_dataset"
        # Only upload if not already there
        cnt_out, _, _ = r.run(f"ls {remote_dir}/*.jpg 2>/dev/null | wc -l")
        if cnt_out.strip() == str(n) and n > 0:
            print(f"  SceneImage: {n} photos already on remote, skipping")
        else:
            print(f"  SceneImage: {n} photos uploading...")
            r.put_dir_tar(photo_dir, remote_dir)
    else:
        print(f"  [WARN] SceneImage not found at {photo_dir}")

    # Minecraft tiles
    tile_dir = local_ds / "MinecraftImage" / "tiles"
    if tile_dir.is_dir():
        n = len(list(tile_dir.glob("*.png")))
        remote_dir = f"{REMOTE_ROOT}/Datasets/MinecraftImage/tiles"
        cnt_out, _, _ = r.run(f"ls {remote_dir}/*.png 2>/dev/null | wc -l")
        if cnt_out.strip() == str(n) and n > 0:
            print(f"  Minecraft tiles: {n} already on remote, skipping")
        else:
            print(f"  Minecraft tiles: {n} uploading...")
            r.put_dir_tar(tile_dir, remote_dir)
    else:
        print(f"  [WARN] Minecraft tiles not found at {tile_dir}")

    # Palette and Gram stats (small, always sync)
    for sub, fname in [("palette", "minecraft_64.npy"), ("style", "minecraft_gram_stats.pt")]:
        local_f = local_ds / sub / fname
        remote_f = f"{REMOTE_ROOT}/Datasets/{sub}/{fname}"
        if local_f.is_file():
            r.mkdir(f"{REMOTE_ROOT}/Datasets/{sub}")
            r.put_file(local_f, remote_f)
            print(f"  {sub}/{fname} uploaded")
        else:
            print(f"  [WARN] {sub}/{fname} not found locally — will compute on remote")

    print("  Done.")


def step_setup_env(r: Remote):
    """Install PyTorch + deps in conda env on remote."""
    print("\n" + "=" * 60)
    print("[STEP 3] Setting up Python environment")
    print("=" * 60)

    # Check if conda available
    out, _, _ = r.run("test -f /root/miniconda3/bin/python && echo YES || echo NO")
    if "YES" not in out:
        print("  [ERROR] No conda/python found on remote!")
        return False

    # Check PyTorch
    out, _, _ = r.run(
        "/root/miniconda3/bin/python -c 'import torch; print(torch.__version__)' 2>&1"
    )
    if "torch" in out and "Error" not in out:
        print(f"  PyTorch already installed: {out.strip()}")
    else:
        print("  Installing PyTorch (CUDA 12.1)...")
        r.run_verbose(
            "/root/miniconda3/bin/pip install torch torchvision "
            "--index-url https://download.pytorch.org/whl/cu121",
            timeout=600,
        )

    # Install other deps
    print("  Installing dependencies...")
    r.run_verbose(
        "/root/miniconda3/bin/pip install scikit-learn pillow numpy tqdm tensorboard -q",
        timeout=300,
    )

    # Verify GPU
    out, _, _ = r.run(
        "/root/miniconda3/bin/python -c 'import torch; "
        "print(f\"PyTorch={torch.__version__}\"); "
        "print(f\"CUDA={torch.cuda.is_available()}\"); "
        "print(f\"GPU={torch.cuda.get_device_name(0)}\"); "
        "print(f\"VRAM={torch.cuda.get_device_properties(0).total_mem/1024**3:.0f}GB\")' 2>&1"
    )
    print(f"  GPU check:\n{out}")
    print("  Done.")
    return True


def step_precompute(r: Remote):
    """Run palette extraction and Gram stats on remote."""
    print("\n" + "=" * 60)
    print("[STEP 4] Precomputing palette & style stats")
    print("=" * 60)

    py = "/root/miniconda3/bin/python"
    root = REMOTE_ROOT

    # Palette
    if r.file_exists(f"{root}/Datasets/palette/minecraft_64.npy"):
        print("  Palette already exists, skipping")
    else:
        print("  Computing k-means palette (64 colors)...")
        r.run_verbose(
            f"{py} {root}/scripts/extract_palette.py "
            f"--tile-dir {root}/Datasets/MinecraftImage/tiles "
            f"--palette-size 64 "
            f"-o {root}/Datasets/palette/minecraft_64.npy",
            timeout=300,
        )

    # Gram stats
    if r.file_exists(f"{root}/Datasets/style/minecraft_gram_stats.pt"):
        print("  Gram stats already exist, skipping")
    else:
        print("  Computing Gram style statistics...")
        r.run_verbose(
            f"{py} {root}/scripts/compute_style_stats.py "
            f"--tile-dir {root}/Datasets/MinecraftImage/tiles "
            f"--device cuda "
            f"-o {root}/Datasets/style/minecraft_gram_stats.pt",
            timeout=600,
        )

    print("  Done.")


def step_launch_train(r: Remote, run_name, batch_size, epochs, fp16, extra_args):
    """Launch nohup training on remote."""
    print("\n" + "=" * 60)
    print("[STEP 5] Launching training")
    print("=" * 60)

    py = "/root/miniconda3/bin/python"
    root = REMOTE_ROOT
    out_dir = f"{root}/outputs/{run_name}"

    r.mkdir(out_dir)

    fp16_flag = "--fp16" if fp16 else ""
    cmd = (
        f"cd {root} && "
        f"nohup {py} -m src.train "
        f"--photo-dir {root}/Datasets/SceneImage/landscape_dataset "
        f"--tile-dir {root}/Datasets/MinecraftImage/tiles "
        f"--palette {root}/Datasets/palette/minecraft_64.npy "
        f"--gram-stats {root}/Datasets/style/minecraft_gram_stats.pt "
        f"--output {out_dir} "
        f"--batch-size {batch_size} "
        f"--epochs {epochs} "
        f"{fp16_flag} "
        f"{extra_args} "
        f"> {out_dir}/train.log 2>&1 & "
        f"echo $! > {out_dir}/train.pid"
    )

    print(f"  Run name: {run_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  FP16: {fp16}")
    print(f"  Output: {out_dir}")

    r.run_nohup(cmd, timeout=5)

    # Verify started
    pid_out, _, _ = r.run(f"cat {out_dir}/train.pid 2>/dev/null")
    pid = pid_out.strip()
    if pid:
        alive, _, _ = r.run(f"kill -0 {pid} 2>&1 && echo ALIVE || echo DEAD")
        if "ALIVE" in alive:
            print(f"\n  [OK] Training started! PID: {pid}")

            # Show GPU
            gpu_out, _, _ = r.run(
                "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total "
                "--format=csv,noheader 2>&1"
            )
            print(f"  GPU status:\n{gpu_out}")

            return True
        else:
            print(f"  [FAIL] Process died immediately. Last log lines:")
            log, _, _ = r.run(f"tail -30 {out_dir}/train.log 2>&1")
            print(log)
            return False
    else:
        print("  [FAIL] No PID file created.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="One-click deploy & train on remote GPU server"
    )
    parser.add_argument("--run-name", default="run01", help="Training run name")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--extra-args", default="", help="Extra training args")
    parser.add_argument("--skip-upload", action="store_true", help="Skip dataset upload")
    parser.add_argument("--skip-packages", action="store_true", help="Skip pip install")
    parser.add_argument("--skip-precompute", action="store_true", help="Skip palette/stats")
    args = parser.parse_args()

    if args.no_fp16:
        args.fp16 = False

    local_root = Path(__file__).resolve().parent.parent

    print("=" * 60)
    print("  ImagePixelNetWork — Remote Deploy & Train")
    print("=" * 60)
    print(f"  Server: {REMOTE_HOST}:{REMOTE_PORT}")
    print(f"  GPU: RTX 4080 SUPER (32 GB VRAM)")
    print(f"  Run: {args.run_name} | Batch: {args.batch_size} | Epochs: {args.epochs}")
    print("=" * 60)

    r = Remote()
    try:
        step_deploy_project(r, local_root)

        if not args.skip_upload:
            step_upload_datasets(r, local_root)

        if not args.skip_packages:
            step_setup_env(r)

        if not args.skip_precompute:
            step_precompute(r)

        success = step_launch_train(
            r, args.run_name, args.batch_size, args.epochs,
            args.fp16, args.extra_args,
        )

        print("\n" + "=" * 60)
        if success:
            print("  TRAINING LAUNCHED SUCCESSFULLY")
            print("=" * 60)
            print(f"""
  Monitor progress:
    python scripts/check_remote.py --run-name {args.run_name}

  Or SSH directly:
    ssh -p {REMOTE_PORT} {REMOTE_USER}@{REMOTE_HOST}
    tail -f {REMOTE_ROOT}/outputs/{args.run_name}/train.log

  Download results:
    python scripts/fetch_results.py --run-name {args.run_name}
    (results will be saved to outputs/{args.run_name}/)
""")
        else:
            print("  TRAINING FAILED TO START")
            print("=" * 60)
            print("  Check the error messages above.")
    finally:
        r.close()


if __name__ == "__main__":
    main()
