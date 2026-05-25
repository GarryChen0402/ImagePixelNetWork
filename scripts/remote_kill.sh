#!/bin/bash
# ============================================================
# Kill remote training process.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

RUN_NAME="${1:-run01}"
REMOTE_OUTPUT="${REMOTE_OUTPUT_DIR}/${RUN_NAME}"

echo "=== Killing remote training: ${RUN_NAME} ==="

ssh_remote "bash -c '
REMOTE_OUTPUT=\"${REMOTE_OUTPUT}\"
if [ -f ${REMOTE_OUTPUT}/train.pid ]; then
    PID=\$(cat ${REMOTE_OUTPUT}/train.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo \"Sending SIGTERM to PID \$PID...\"
        kill \$PID
        sleep 2
        if kill -0 \$PID 2>/dev/null; then
            echo \"Process still alive, sending SIGKILL...\"
            kill -9 \$PID
        fi
        echo \"Process killed.\"
    else
        echo \"Process \$PID is not running.\"
    fi
else
    echo \"No PID file found at ${REMOTE_OUTPUT}/train.pid\"
fi
'"

echo "Done."
