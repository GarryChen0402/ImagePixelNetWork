#!/bin/bash
# ============================================================
# Deploy: sync project code to remote GPU server.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

echo "=== Deploying to ${SSH_DEST}:${REMOTE_ROOT} ==="

# Ensure remote directory exists
ssh_remote "mkdir -p ${REMOTE_ROOT}/src ${REMOTE_ROOT}/scripts ${REMOTE_ROOT}/Docs"

# Sync source files (src/) — excludes __pycache__
echo "[1/4] Syncing src/"
ssh_remote "mkdir -p ${REMOTE_ROOT}/src"
for f in "${LOCAL_ROOT}"/src/*.py; do
    scp_to_remote "$f" "${REMOTE_ROOT}/src/"
done

# Sync scripts
echo "[2/4] Syncing scripts/"
for f in "${LOCAL_ROOT}"/scripts/*.{sh,py} 2>/dev/null; do
    [ -f "$f" ] && scp_to_remote "$f" "${REMOTE_ROOT}/scripts/"
done

# Sync CLAUDE.md and Docs
echo "[3/4] Syncing Docs & config"
scp_to_remote "${LOCAL_ROOT}/CLAUDE.md" "${REMOTE_ROOT}/CLAUDE.md" 2>/dev/null || true
ssh_remote "mkdir -p ${REMOTE_ROOT}/Docs"
for f in "${LOCAL_ROOT}"/Docs/*.md; do
    scp_to_remote "$f" "${REMOTE_ROOT}/Docs/"
done

# Sync DataSets (only palette and gram stats, not images — images should be uploaded separately)
echo "[4/4] Syncing palette & style stats (image datasets must be uploaded separately)"
ssh_remote "mkdir -p ${REMOTE_ROOT}/Datasets/palette ${REMOTE_ROOT}/Datasets/style"
scp_to_remote "${LOCAL_ROOT}/Datasets/palette/minecraft_64.npy" "${REMOTE_ROOT}/Datasets/palette/" 2>/dev/null || echo "  (palette not found, run extract_palette.py first)"
scp_to_remote "${LOCAL_ROOT}/Datasets/style/minecraft_gram_stats.pt" "${REMOTE_ROOT}/Datasets/style/" 2>/dev/null || echo "  (gram stats not found, run compute_style_stats.py first)"

echo ""
echo "=== Deploy complete ==="
echo "Next steps:"
echo "  1. Upload image datasets to ${SSH_DEST}:${REMOTE_DATA_DIR}/"
echo "     SceneImage/landscape_dataset/  (7,268 photos)"
echo "     MinecraftImage/tiles/          (2,004 tiles)"
echo "  2. Run: ./scripts/remote_train.sh"
