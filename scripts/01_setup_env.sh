#!/usr/bin/env bash
# =============================================================================
# OTP-VLA Environment Setup Script
# Supports conda (preferred) or venv (fallback).
# Creates env 'otp_vla', installs all dependencies,
# clones third-party repos, and verifies the installation.
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

# Repo root = parent of this script's directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="otp_vla"
PYTHON_VERSION="3.10"
THIRD_PARTY="${REPO_ROOT}/third_party"
VENV_DIR="${REPO_ROOT}/.venv"

info "=== OTP-VLA Environment Setup ==="
info "Repo root : ${REPO_ROOT}"
info "Target env: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo ""

# --------------------------------------------------------------------------
# 1. Detect environment manager (conda preferred, venv fallback)
# --------------------------------------------------------------------------
info "[1/7] Detecting environment manager..."

USE_CONDA=false
USE_VENV=false

if command -v conda &>/dev/null; then
    USE_CONDA=true
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    success "conda found at: ${CONDA_BASE}"
else
    warn "conda not found — falling back to Python venv."
    warn "For production use, install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    USE_VENV=true

    # Find a suitable Python (prefer 3.10, accept 3.11/3.12)
    PYTHON_BIN=""
    for candidate in python3.10 python3.11 python3.12 python3; do
        if command -v "${candidate}" &>/dev/null; then
            ver=$("${candidate}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "${ver}" == "3.10" || "${ver}" == "3.11" || "${ver}" == "3.12" ]]; then
                PYTHON_BIN="${candidate}"
                success "Found Python ${ver} at: $(command -v ${candidate})"
                break
            fi
        fi
    done
    [ -z "${PYTHON_BIN}" ] && die "Python 3.10+ not found. Install it first."
fi

# --------------------------------------------------------------------------
# 2. Create / activate environment
# --------------------------------------------------------------------------
info "[2/7] Setting up environment..."

if ${USE_CONDA}; then
    if conda env list | grep -qE "^${ENV_NAME}\s"; then
        warn "Conda env '${ENV_NAME}' already exists — skipping creation."
    else
        conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
        success "Conda env '${ENV_NAME}' created."
    fi
    conda activate "${ENV_NAME}"
    PYTHON_BIN="python"
    success "Activated conda env: ${ENV_NAME}"
else
    if [ -d "${VENV_DIR}" ]; then
        warn "venv '${VENV_DIR}' already exists — reusing."
    else
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
        success "venv created at: ${VENV_DIR}"
    fi
    # shellcheck source=/dev/null
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="python"
    success "Activated venv: ${VENV_DIR}"
fi

# --------------------------------------------------------------------------
# 3. Upgrade pip and install requirements
# --------------------------------------------------------------------------
info "[3/7] Installing pip dependencies from requirements.txt..."
${PYTHON_BIN} -m pip install --upgrade pip setuptools wheel

# Install the main requirements (flash-attn line is commented out there)
${PYTHON_BIN} -m pip install -r "${REPO_ROOT}/requirements.txt"
success "Core requirements installed."

# --------------------------------------------------------------------------
# 4. Install flash-attn separately (requires --no-build-isolation)
# --------------------------------------------------------------------------
info "[4/7] Installing flash-attn==2.5.5 (--no-build-isolation)..."
if ${PYTHON_BIN} -c "import flash_attn" 2>/dev/null; then
    warn "flash_attn already importable — skipping."
else
    ${PYTHON_BIN} -m pip install flash-attn==2.5.5 --no-build-isolation || {
        warn "flash-attn installation failed. This is non-fatal — continuing."
        warn "To retry manually: pip install flash-attn==2.5.5 --no-build-isolation"
    }
fi

# --------------------------------------------------------------------------
# 5. Clone and install third-party repos
# --------------------------------------------------------------------------
info "[5/7] Cloning third-party repositories into ${THIRD_PARTY}..."
mkdir -p "${THIRD_PARTY}"

# OpenVLA-OFT
OPENVLA_DIR="${THIRD_PARTY}/openvla-oft"
if [ -d "${OPENVLA_DIR}/.git" ]; then
    warn "openvla-oft already cloned — skipping."
else
    info "  Cloning OpenVLA-OFT..."
    git clone https://github.com/moojink/openvla-oft.git "${OPENVLA_DIR}"
fi
info "  Installing OpenVLA-OFT (pip install -e)..."
${PYTHON_BIN} -m pip install -e "${OPENVLA_DIR}" || \
    warn "openvla-oft editable install failed — check repo for setup.py/pyproject.toml."
success "openvla-oft ready."

# LIBERO
LIBERO_DIR="${THIRD_PARTY}/LIBERO"
if [ -d "${LIBERO_DIR}/.git" ]; then
    warn "LIBERO already cloned — skipping."
else
    info "  Cloning LIBERO..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi
info "  Installing LIBERO (pip install -e)..."
${PYTHON_BIN} -m pip install -e "${LIBERO_DIR}" || \
    warn "LIBERO editable install failed — check repo for setup.py/pyproject.toml."
success "LIBERO ready."

# Install this package itself in editable mode
info "  Installing otp_vla package (pip install -e ${REPO_ROOT})..."
${PYTHON_BIN} -m pip install -e "${REPO_ROOT}"
success "otp_vla package installed."

# --------------------------------------------------------------------------
# 6. Verify imports
# --------------------------------------------------------------------------
info "[6/7] Verifying critical imports..."
${PYTHON_BIN} - <<'PYEOF'
import sys, importlib

checks = [
    ("torch",          "PyTorch"),
    ("torchvision",    "torchvision"),
    ("transformers",   "transformers"),
    ("peft",           "PEFT"),
    ("accelerate",     "accelerate"),
    ("dlimp",          "dlimp (OpenVLA data)"),
    ("einops",         "einops"),
    ("hydra",          "hydra-core"),
    ("wandb",          "wandb"),
    ("cv2",            "opencv-python"),
]

failed = []
for mod, label in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  [OK]  {label:<30} {ver}")
    except ImportError as e:
        print(f"  [FAIL] {label:<30} {e}")
        failed.append(label)

# CUDA check
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        dev = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        print(f"\n  [OK]  CUDA available: {dev} ({mem} GB)")
    else:
        print(f"\n  [WARN] CUDA not available (CPU-only mode)")
except Exception as e:
    print(f"\n  [WARN] CUDA check failed: {e}")

if failed:
    print(f"\n  [FAIL] Failed imports: {failed}", file=sys.stderr)
    sys.exit(1)
else:
    print("\n  All critical imports passed.")
PYEOF
success "Import verification complete."

# --------------------------------------------------------------------------
# 7. Done
# --------------------------------------------------------------------------
info "[7/7] Setup complete!"
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} OTP-VLA environment is ready.${NC}"
echo ""
if ${USE_CONDA}; then
    echo -e " Activate with:  ${CYAN}conda activate ${ENV_NAME}${NC}"
else
    echo -e " Activate with:  ${CYAN}source ${VENV_DIR}/bin/activate${NC}"
fi
echo -e " Quick checks:"
echo -e "   ${CYAN}python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\"${NC}"
echo -e "   ${CYAN}python -c \"import transformers, peft, dlimp\"${NC}"
echo -e "${GREEN}============================================================${NC}"
