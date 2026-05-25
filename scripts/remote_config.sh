#!/bin/bash
# ============================================================
# Remote GPU Server Configuration
# Modify these values to match your remote server setup.
# ============================================================

# --- SSH Connection ---
REMOTE_HOST="gpu-server.example.com"
REMOTE_USER="${REMOTE_USER:-$(whoami)}"
REMOTE_PORT="${REMOTE_PORT:-22}"
SSH_OPTS="-o StrictHostKeyChecking=no -o ServerAliveInterval=60"

# --- Remote Paths ---
REMOTE_ROOT="~/ImagePixelNetWork"
REMOTE_DATA_DIR="${REMOTE_ROOT}/Datasets"
REMOTE_OUTPUT_DIR="${REMOTE_ROOT}/outputs"

# --- Local Paths ---
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCAL_OUTPUT_DIR="${LOCAL_ROOT}/outputs"

# --- Training Parameters ---
# Override by setting env vars before running scripts
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-300}"
FP16="${FP16:-true}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# --- Derived SSH string ---
SSH_DEST="${REMOTE_USER}@${REMOTE_HOST}"
if [ "${REMOTE_PORT}" != "22" ]; then
    SSH_OPTS="${SSH_OPTS} -p ${REMOTE_PORT}"
fi

# --- Helper functions ---
ssh_remote() {
    ssh ${SSH_OPTS} "${SSH_DEST}" "$@"
}

scp_to_remote() {
    scp ${SSH_OPTS} "$1" "${SSH_DEST}:$2"
}

scp_from_remote() {
    scp ${SSH_OPTS} "${SSH_DEST}:$1" "$2"
}
