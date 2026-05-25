#!/bin/bash
# ============================================================
# Setup Python environment on remote GPU server.
# One-time setup: creates venv, installs PyTorch + deps.
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/remote_config.sh"

echo "=== Setting up remote environment on ${SSH_DEST} ==="

ssh_remote "bash -s" << 'ENDSSH'
set -euo pipefail

# --- Detect CUDA version ---
echo "[1/5] Detecting CUDA..."
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | sed 's/.*release //;s/,.*//')
    echo "  CUDA ${CUDA_VER} found"
else
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' 2>/dev/null || echo "unknown")
    echo "  CUDA driver version: ${CUDA_VER}"
fi

# --- Python ---
echo "[2/5] Checking Python..."
PYTHON=$(which python3.10 2>/dev/null || which python3 2>/dev/null || which python 2>/dev/null)
echo "  Using: $(${PYTHON} --version)"

# --- Create venv ---
cd ~/ImagePixelNetWork
VENV_DIR="./venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "[3/5] Creating virtual environment..."
    ${PYTHON} -m venv ${VENV_DIR}
else
    echo "[3/5] Virtual environment exists"
fi

source ${VENV_DIR}/bin/activate

# --- Install PyTorch ---
echo "[4/5] Installing PyTorch..."
if ${PYTHON} -c "import torch" 2>/dev/null; then
    echo "  PyTorch already installed: $(${PYTHON} -c 'import torch; print(torch.__version__)')"
else
    # CUDA 12.x
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# --- Install deps ---
echo "[5/5] Installing dependencies..."
pip install -q scikit-learn pillow numpy tqdm tensorboard

# --- Verify GPU ---
echo ""
echo "=== Verifying GPU access ==="
${PYTHON} -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'       Memory: {torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo "Run the following on remote to precompute data:"
echo "  cd ~/ImagePixelNetWork && source venv/bin/activate"
echo "  python scripts/extract_palette.py"
echo "  python scripts/compute_style_stats.py --device cuda"
ENDSSH
