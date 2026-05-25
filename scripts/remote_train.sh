#!/bin/bash
# ============================================================
# Launch training on remote GPU server via nohup.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

RUN_NAME="${1:-run01}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_OUTPUT="${REMOTE_OUTPUT_DIR}/${RUN_NAME}"

echo "=== Launching remote training: ${RUN_NAME} ==="
echo "Remote: ${SSH_DEST}"
echo "Output: ${REMOTE_OUTPUT}"

# Build the training command
FP16_FLAG=""
[ "${FP16}" = "true" ] && FP16_FLAG="--fp16"

TRAIN_CMD="cd ${REMOTE_ROOT} && \
  nohup python -m src.train \
    --photo-dir ${REMOTE_DATA_DIR}/SceneImage/landscape_dataset \
    --tile-dir ${REMOTE_DATA_DIR}/MinecraftImage/tiles \
    --palette ${REMOTE_DATA_DIR}/palette/minecraft_64.npy \
    --gram-stats ${REMOTE_DATA_DIR}/style/minecraft_gram_stats.pt \
    --output ${REMOTE_OUTPUT} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    ${FP16_FLAG} \
    ${EXTRA_ARGS} \
    > ${REMOTE_OUTPUT}/train.log 2>&1 & \
  echo \$! > ${REMOTE_OUTPUT}/train.pid"

# Create remote output directory
ssh_remote "mkdir -p ${REMOTE_OUTPUT}"

# Check that datasets exist on remote
echo ""
echo "Checking remote datasets..."
ssh_remote "bash -c '
ok=true
if [ ! -d ${REMOTE_DATA_DIR}/SceneImage/landscape_dataset ]; then
    echo \"  MISSING: SceneImage/landscape_dataset/\"; ok=false
fi
if [ ! -d ${REMOTE_DATA_DIR}/MinecraftImage/tiles ]; then
    echo \"  MISSING: MinecraftImage/tiles/\"; ok=false
fi
if [ ! -f ${REMOTE_DATA_DIR}/palette/minecraft_64.npy ]; then
    echo \"  MISSING: palette/minecraft_64.npy (run extract_palette.py on remote)\"; ok=false
fi
if [ ! -f ${REMOTE_DATA_DIR}/style/minecraft_gram_stats.pt ]; then
    echo \"  MISSING: style/minecraft_gram_stats.pt (run compute_style_stats.py on remote)\"; ok=false
fi
if [ \"\$ok\" = true ]; then echo \"  All datasets found.\"; fi
'"

echo ""
echo "Launching training..."
ssh_remote "bash -c '${TRAIN_CMD}'"

# Wait a moment and verify process started
sleep 3

echo ""
echo "Checking remote process..."
ssh_remote "bash -c '
if [ -f ${REMOTE_OUTPUT}/train.pid ]; then
    PID=\$(cat ${REMOTE_OUTPUT}/train.pid)
    if kill -0 \$PID 2>/dev/null; then
        echo \"  Training running (PID: \$PID)\"
        echo \"  GPU status:\"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo \"  (nvidia-smi not available)\"
    else
        echo \"  WARNING: Process died. Check log:\"
        tail -20 ${REMOTE_OUTPUT}/train.log
    fi
else
    echo \"  WARNING: PID file not created. Check log:\"
    tail -20 ${REMOTE_OUTPUT}/train.log 2>/dev/null || echo \"  (no log yet)\"
fi
'"

echo ""
echo "=== Training launched ==="
echo "Monitor:  ./scripts/remote_status.sh ${RUN_NAME}"
echo "Fetch:    ./scripts/fetch_checkpoints.sh ${RUN_NAME}"
echo "Kill:     ./scripts/remote_kill.sh ${RUN_NAME}"
echo "SSH:      ssh ${SSH_OPTS} ${SSH_DEST}"
