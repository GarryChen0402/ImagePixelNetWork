#!/usr/bin/env python3
"""Check remote training status, GPU usage, and recent logs."""

import argparse
import paramiko

REMOTE_HOST = "connect.westc.seetacloud.com"
REMOTE_PORT = 48897
REMOTE_USER = "root"
REMOTE_PASS = "BohrzDStaCx5"
REMOTE_ROOT = "/root/ImagePixelNetWork"


def main():
    parser = argparse.ArgumentParser(description="Check remote training status")
    parser.add_argument("--run-name", default="run01")
    parser.add_argument("--tail", type=int, default=40, help="Lines of log to show")
    args = parser.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER,
                password=REMOTE_PASS, timeout=15)

    out_dir = f"{REMOTE_ROOT}/outputs/{args.run_name}"

    print(f"=== Training Status: {args.run_name} ===")
    print(f"Server: {REMOTE_HOST}:{REMOTE_PORT}")

    # PID check
    stdin, stdout, stderr = ssh.exec_command(
        f"if [ -f {out_dir}/train.pid ]; then "
        f"  PID=$(cat {out_dir}/train.pid); "
        f"  if kill -0 $PID 2>/dev/null; then "
        f"    echo RUNNING PID=$PID; "
        f"    ps -o etime= -p $PID 2>/dev/null | awk '{{print \"UPTIME=\"$1}}'; "
        f"  else echo STOPPED; fi; "
        f"else echo NO_PID_FILE; fi"
    )
    print(stdout.read().decode().strip())

    # GPU
    print("\n--- GPU ---")
    stdin, stdout, stderr = ssh.exec_command(
        "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader 2>&1"
    )
    print(stdout.read().decode().strip())

    # Checkpoints
    print(f"\n--- Checkpoints ---")
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -1t {out_dir}/checkpoints/ 2>/dev/null | head -5 || echo '  (none yet)'"
    )
    print(stdout.read().decode().strip())

    # Recent losses
    print(f"\n--- Recent Epoch Summaries ---")
    stdin, stdout, stderr = ssh.exec_command(
        f"grep 'Epoch' {out_dir}/train.log 2>/dev/null | tail -5 || echo '  (no epochs yet)'"
    )
    print(stdout.read().decode().strip())

    # Tail of log
    print(f"\n=== Last {args.tail} Log Lines ===")
    stdin, stdout, stderr = ssh.exec_command(
        f"tail -{args.tail} {out_dir}/train.log 2>/dev/null || echo '  (no log yet)'"
    )
    print(stdout.read().decode())

    ssh.close()


if __name__ == "__main__":
    main()
