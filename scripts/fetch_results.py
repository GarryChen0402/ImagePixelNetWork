#!/usr/bin/env python3
"""Download training results: checkpoints, samples, logs, and metrics.

Usage:
  python scripts/fetch_results.py --run-name run01          # latest checkpoint only
  python scripts/fetch_results.py --run-name run01 --all    # all checkpoints
"""

import argparse
import json
import os
import re
from pathlib import Path

import paramiko

REMOTE_HOST = "connect.westc.seetacloud.com"
REMOTE_PORT = 48897
REMOTE_USER = "root"
REMOTE_PASS = "BohrzDStaCx5"
REMOTE_ROOT = "/root/ImagePixelNetWork"


def parse_metrics(log_text: str) -> list:
    """Extract per-epoch metrics from training log."""
    metrics = []
    for line in log_text.split("\n"):
        m = re.match(
            r"Epoch (\d+)\s+\|\s+G=([\d.]+)\s+Content=([\d.]+)\s+"
            r"D=([\d.]+)\s+τ=([\d.]+)", line
        )
        if m:
            metrics.append({
                "epoch": int(m.group(1)),
                "g_loss": float(m.group(2)),
                "content_loss": float(m.group(3)),
                "d_loss": float(m.group(4)),
                "tau": float(m.group(5)),
            })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fetch training results from remote")
    parser.add_argument("--run-name", default="run01")
    parser.add_argument("--all", action="store_true", help="Download all checkpoints")
    args = parser.parse_args()

    local_root = Path(__file__).resolve().parent.parent
    local_out = local_root / "outputs" / args.run_name
    remote_out = f"{REMOTE_ROOT}/outputs/{args.run_name}"

    local_out.mkdir(parents=True, exist_ok=True)
    (local_out / "checkpoints").mkdir(exist_ok=True)
    (local_out / "samples").mkdir(exist_ok=True)
    (local_out / "logs").mkdir(exist_ok=True)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER,
                password=REMOTE_PASS, timeout=15)
    sftp = ssh.open_sftp()

    print(f"=== Fetching results: {args.run_name} ===")

    # Download log
    print("Fetching training log...")
    try:
        sftp.get(f"{remote_out}/train.log", str(local_out / "logs" / "train.log"))
        print("  train.log downloaded")
    except FileNotFoundError:
        print("  (no log yet)")

    # Extract and save metrics JSON
    log_path = local_out / "logs" / "train.log"
    if log_path.exists():
        log_text = log_path.read_text(errors="replace")
        metrics = parse_metrics(log_text)
        if metrics:
            (local_out / "logs" / "metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False)
            )
            print(f"  metrics.json extracted ({len(metrics)} epochs)")

            # Summary
            last = metrics[-1]
            print(f"\n  Final metrics (epoch {last['epoch']}):")
            print(f"    G Loss:       {last['g_loss']:.4f}")
            print(f"    Content Loss: {last['content_loss']:.4f}")
            print(f"    D Loss:       {last['d_loss']:.4f}")
            print(f"    Tau:          {last['tau']:.4f}")

    # Download checkpoints
    print("\nFetching checkpoints...")
    stdin, stdout, stderr = ssh.exec_command(f"ls -1t {remote_out}/checkpoints/ 2>/dev/null")
    ckpts = stdout.read().decode().strip().split("\n")
    ckpts = [c for c in ckpts if c]

    if ckpts:
        if args.all:
            to_fetch = ckpts
        else:
            # Latest only
            to_fetch = [ckpts[0]]
            if len(ckpts) > 1:
                # Also fetch the final model (last epoch) if different from latest
                final = ckpts[-1]
                if final not in to_fetch and len(ckpts) > 1:
                    to_fetch.append(final)

        for ckpt in to_fetch:
            local = local_out / "checkpoints" / ckpt
            if not local.exists():
                print(f"  Downloading {ckpt}...")
                sftp.get(f"{remote_out}/checkpoints/{ckpt}", str(local))
            else:
                print(f"  {ckpt} (already downloaded)")

        print(f"  {len(to_fetch)} checkpoint(s) downloaded")
    else:
        print("  (no checkpoints yet)")

    # Download samples
    print("\nFetching samples...")
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -1t {remote_out}/samples/ 2>/dev/null | head -10"
    )
    samples = stdout.read().decode().strip().split("\n")
    samples = [s for s in samples if s]

    if samples:
        for s in samples:
            local = local_out / "samples" / s
            if not local.exists():
                sftp.get(f"{remote_out}/samples/{s}", str(local))
        print(f"  {len(samples)} sample(s) downloaded")
    else:
        print("  (no samples yet)")

    sftp.close()
    ssh.close()

    print(f"\n=== Results saved to {local_out.resolve()} ===")
    print(f"  logs/        training log + metrics.json")
    print(f"  checkpoints/ model weights")
    print(f"  samples/     generated images")


if __name__ == "__main__":
    main()
