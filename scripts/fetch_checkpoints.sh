#!/bin/bash
# ============================================================
# Download checkpoints and samples from remote server.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

RUN_NAME="${1:-run01}"
REMOTE_OUTPUT="${REMOTE_OUTPUT_DIR}/${RUN_NAME}"
LOCAL_DEST="${LOCAL_OUTPUT_DIR}/${RUN_NAME}"

echo "=== Fetching results: ${RUN_NAME} ==="

mkdir -p "${LOCAL_DEST}/checkpoints"
mkdir -p "${LOCAL_DEST}/samples"
mkdir -p "${LOCAL_DEST}/logs"

# Fetch logs
echo "Fetching logs..."
scp_from_remote "${REMOTE_OUTPUT}/train.log" "${LOCAL_DEST}/logs/" 2>/dev/null || echo "  (no log yet)"

# Fetch latest checkpoint
echo "Fetching latest checkpoint..."
LATEST_CKPT=$(ssh_remote "ls -t ${REMOTE_OUTPUT}/checkpoints/ 2>/dev/null | head -1" || echo "")
if [ -n "${LATEST_CKPT}" ]; then
    scp_from_remote "${REMOTE_OUTPUT}/checkpoints/${LATEST_CKPT}" "${LOCAL_DEST}/checkpoints/"
    echo "  Downloaded: ${LATEST_CKPT}"
else
    echo "  (no checkpoints yet)"
fi

# Fetch all checkpoints (uncomment if needed)
# echo "Fetching all checkpoints..."
# ssh_remote "ls ${REMOTE_OUTPUT}/checkpoints/" 2>/dev/null | while read ckpt; do
#     scp_from_remote "${REMOTE_OUTPUT}/checkpoints/${ckpt}" "${LOCAL_DEST}/checkpoints/"
# done

# Fetch latest samples
echo "Fetching latest samples..."
LATEST_SAMPLE=$(ssh_remote "ls -t ${REMOTE_OUTPUT}/samples/ 2>/dev/null | head -1" || echo "")
if [ -n "${LATEST_SAMPLE}" ]; then
    scp_from_remote "${REMOTE_OUTPUT}/samples/${LATEST_SAMPLE}" "${LOCAL_DEST}/samples/"
    echo "  Downloaded: ${LATEST_SAMPLE}"
else
    echo "  (no samples yet)"
fi

echo ""
echo "=== Fetch complete ==="
echo "Results in: ${LOCAL_DEST}"
ls -la "${LOCAL_DEST}/checkpoints/" 2>/dev/null || echo "  (checkpoints dir empty)"
ls -la "${LOCAL_DEST}/samples/" 2>/dev/null || echo "  (samples dir empty)"
ls -la "${LOCAL_DEST}/logs/" 2>/dev/null || echo "  (logs dir empty)"
