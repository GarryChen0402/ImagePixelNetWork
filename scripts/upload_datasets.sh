#!/bin/bash
# ============================================================
# Upload large image datasets to remote server.
# Datasets are often too large for git; use this to sync them.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

LOCAL_DATASETS="${LOCAL_ROOT}/Datasets"
REMOTE_DATASETS="${REMOTE_DATA_DIR}"

echo "=== Uploading datasets to ${SSH_DEST}:${REMOTE_DATASETS} ==="
echo "This may take a while for large datasets..."

# Create remote dataset dirs
ssh_remote "mkdir -p ${REMOTE_DATASETS}/SceneImage/landscape_dataset"
ssh_remote "mkdir -p ${REMOTE_DATASETS}/MinecraftImage/tiles"
ssh_remote "mkdir -p ${REMOTE_DATASETS}/palette"
ssh_remote "mkdir -p ${REMOTE_DATASETS}/style"

# Upload SceneImage (7,268 files, ~200MB total estimated)
echo ""
echo "[1/4] Uploading SceneImage photos..."
PHOTO_COUNT=$(ls "${LOCAL_DATASETS}/SceneImage/landscape_dataset/"*.jpg 2>/dev/null | wc -l)
echo "  ${PHOTO_COUNT} photos to upload"
# Use tar+ssh for efficiency
cd "${LOCAL_DATASETS}/SceneImage"
tar czf - landscape_dataset/ | ssh ${SSH_OPTS} "${SSH_DEST}" "cd ${REMOTE_DATASETS}/SceneImage && tar xzf -"
echo "  Done."

# Upload Minecraft tiles (2,004 files)
echo ""
echo "[2/4] Uploading Minecraft tiles..."
TILE_COUNT=$(ls "${LOCAL_DATASETS}/MinecraftImage/tiles/"*.png 2>/dev/null | wc -l)
echo "  ${TILE_COUNT} tiles to upload"
cd "${LOCAL_DATASETS}/MinecraftImage"
tar czf - tiles/ | ssh ${SSH_OPTS} "${SSH_DEST}" "cd ${REMOTE_DATASETS}/MinecraftImage && tar xzf -"
echo "  Done."

# Upload palette (small)
echo ""
echo "[3/4] Uploading palette..."
scp_to_remote "${LOCAL_DATASETS}/palette/minecraft_64.npy" "${REMOTE_DATASETS}/palette/"

# Upload Gram stats (small)
echo ""
echo "[4/4] Uploading style stats..."
scp_to_remote "${LOCAL_DATASETS}/style/minecraft_gram_stats.pt" "${REMOTE_DATASETS}/style/"

echo ""
echo "=== Upload complete ==="
echo "Verifying remote file counts..."
ssh_remote "bash -c '
echo \"  Photos: \$(ls ${REMOTE_DATASETS}/SceneImage/landscape_dataset/*.jpg 2>/dev/null | wc -l)\"
echo \"  Tiles:  \$(ls ${REMOTE_DATASETS}/MinecraftImage/tiles/*.png 2>/dev/null | wc -l)\"
echo \"  Palette: \$([ -f ${REMOTE_DATASETS}/palette/minecraft_64.npy ] && echo OK || echo MISSING)\"
echo \"  Gram:    \$([ -f ${REMOTE_DATASETS}/style/minecraft_gram_stats.pt ] && echo OK || echo MISSING)\"
'"
