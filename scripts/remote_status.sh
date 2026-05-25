#!/bin/bash
# ============================================================
# Check remote training status and tail logs.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

RUN_NAME="${1:-run01}"
REMOTE_OUTPUT="${REMOTE_OUTPUT_DIR}/${RUN_NAME}"
TAIL_LINES="${2:-40}"

echo "=== Remote Training Status: ${RUN_NAME} ==="

ssh_remote "bash -c '
REMOTE_OUTPUT=\"${REMOTE_OUTPUT}\"

echo \"Server: \$(hostname)\"
echo \"Time:   \$(date)\"
echo \"\"

# Check PID
if [ -f ${REMOTE_OUTPUT}/train.pid ]; then
    PID=\$(cat ${REMOTE_OUTPUT}/train.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo \"Status:  RUNNING (PID: \$PID)\"
        # Show process uptime
        ELAPSED=\$(ps -o etime= -p \$PID 2>/dev/null | tr -d \" \")
        echo \"Uptime:  \$ELAPSED\"
    else
        echo \"Status:  STOPPED (PID \$PID no longer running)\"
    fi
else
    echo \"Status:  NO PID FILE — training may not have been started\"
    echo \"\"
    echo \"Last 20 lines of log:\"
    [ -f ${REMOTE_OUTPUT}/train.log ] && tail -20 ${REMOTE_OUTPUT}/train.log || echo \"  (no log file)\"
    exit 0
fi

echo \"\"
echo \"--- GPU Status ---\"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || echo \"  (nvidia-smi not available)\"

echo \"\"
echo \"--- Checkpoints ---\"
CKPT_DIR=${REMOTE_OUTPUT}/checkpoints
if [ -d \$CKPT_DIR ]; then
    ls -1 \$CKPT_DIR 2>/dev/null | tail -5 || echo \"  (none)\"
else
    echo \"  (no checkpoints yet)\"
fi

echo \"\"
echo \"--- Recent Losses (from log) ---\"
if [ -f ${REMOTE_OUTPUT}/train.log ]; then
    grep \"Epoch\" ${REMOTE_OUTPUT}/train.log 2>/dev/null | tail -5 || echo \"  (no epoch summaries yet)\"
else
    echo \"  (no log file)\"
fi

echo \"\"
echo \"=== Last ${TAIL_LINES} lines of log ===\"
[ -f ${REMOTE_OUTPUT}/train.log ] && tail -${TAIL_LINES} ${REMOTE_OUTPUT}/train.log || echo \"  (no log file)\"
'"
