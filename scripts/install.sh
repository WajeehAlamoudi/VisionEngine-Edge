#!/usr/bin/env bash
# install.sh — single entry point to set up VisionEngine Edge on any device
# Run once from project root: bash scripts/install.sh

set -euo pipefail

# resolve project root (one level up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=10

# ── colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

echo ""
echo "  VisionEngine Edge — installer"
echo "  ──────────────────────────────"
echo ""

# ── python version check ──────────────────────────────────────────────────────
PYTHON=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge "$REQUIRED_PYTHON_MAJOR" ] && [ "$minor" -ge "$REQUIRED_PYTHON_MINOR" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python ${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}+ not found. Install it and retry."
fi
info "Python $("$PYTHON" --version)"

# ── device selection ──────────────────────────────────────────────────────────
echo ""
echo "  What device is this?"
echo "    1) Raspberry Pi / CPU-only device"
echo "    2) NVIDIA Jetson (CUDA via JetPack)"
echo "    3) Generic PC/server with an NVIDIA GPU (CUDA)"
echo "    4) Mac (Apple Silicon / MPS)"
echo "    5) Raspberry Pi + Hailo accelerator"
echo ""
read -r -p "  Choice [1]: " device_choice
device_choice="${device_choice:-1}"

USE_SYSTEM_SITE_PACKAGES=0
TORCH_INSTALL_CMD=""
POST_INSTALL_NOTE=""

case "$device_choice" in
    2)
        echo ""
        info "Jetson — checking for an existing GPU-enabled torch first"
        # NVIDIA factory images usually ship a working, JetPack-matched torch
        # already installed system-wide. A generic `pip install torch` here
        # would silently grab a non-GPU build instead — see requirements.txt
        # for why. So: reuse the working one if it's there.
        if "$PYTHON" -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            info "Found a working system torch with CUDA — venv will reuse it"
            USE_SYSTEM_SITE_PACKAGES=1
        else
            warn "No working CUDA torch found system-wide."
            echo ""
            read -r -p "  JetPack major version (e.g. 6): " jp_version
            read -r -p "  CUDA version on this JetPack, no dot (e.g. 126 for 12.6): " cuda_version
            echo ""
            warn "Run this BEFORE re-running this installer, so the venv can find it:"
            echo "    $PYTHON -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp${jp_version}/cu${cuda_version}"
            echo ""
            read -r -p "  Continue anyway without confirmed GPU torch? [y/N]: " cont
            [[ "$cont" =~ ^[Yy]$ ]] || error "Install the wheel above, then re-run this script."
        fi
        ;;
    3)
        info "Generic NVIDIA GPU — standard PyPI torch build has real CUDA support here"
        TORCH_INSTALL_CMD=""  # default PyPI wheel is correct for a normal desktop/server GPU
        ;;
    4)
        info "Mac (Apple Silicon) — standard PyPI torch build includes MPS support"
        TORCH_INSTALL_CMD=""
        ;;
    5)
        info "Raspberry Pi + Hailo — installing CPU-only torch (Hailo runtime, not torch, does inference)"
        TORCH_INSTALL_CMD="$VENV_DIR/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu"
        POST_INSTALL_NOTE="Hailo needs its own SDK (hailo_platform) set up separately — see the 'Hailo Backend' section in README.md. Its numpy requirement conflicts with this project's other dependencies in one shared environment; that isolation is not yet automated by this script."
        ;;
    *)
        info "CPU-only device — installing the smaller CPU-only torch build"
        TORCH_INSTALL_CMD="$VENV_DIR/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu"
        ;;
esac

# ── virtual environment ───────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at .venv — skipping creation"
else
    info "Creating virtual environment..."
    if [ "$USE_SYSTEM_SITE_PACKAGES" -eq 1 ]; then
        "$PYTHON" -m venv "$VENV_DIR" --system-site-packages
    else
        "$PYTHON" -m venv "$VENV_DIR"
    fi
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

info "Upgrading pip..."
"$PIP" install --quiet --upgrade pip

# ── torch (device-specific, if needed) ────────────────────────────────────────
if [ -n "$TORCH_INSTALL_CMD" ]; then
    info "Installing torch for this device..."
    eval "$TORCH_INSTALL_CMD"
fi

# ── dependencies ──────────────────────────────────────────────────────────────
# requirements.txt's own `torch>=2.0.0` line is satisfied by whatever was
# just set up above (system-wide via --system-site-packages, CPU-only, or
# left to the default PyPI resolution) — it will not be reinstalled or
# overridden here.
info "Installing dependencies from requirements.txt..."
"$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── runtime directories ───────────────────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p "$SCRIPT_DIR/collected"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data"

# model weight files (.pt/.onnx/.engine/.hef) are gitignored — a fresh clone
# has nowhere to put them until this exists
mkdir -p "$SCRIPT_DIR/models"

# ── config ─────────────────────────────────────────────────────────────────────
# The four hardware-agnostic files can be copied straight from the samples —
# they don't need per-device edits before the pipeline can at least start.
# models.yaml / cameras.yaml / device.yaml are intentionally NOT auto-copied:
# they need real device-specific values (model paths, camera sources, device
# identity) that no script can safely guess.
mkdir -p "$SCRIPT_DIR/config"
info "Copying hardware-agnostic config files (skipping any that already exist)..."
for f in api notifications rules collection; do
    dest="$SCRIPT_DIR/config/$f.yaml"
    src="$SCRIPT_DIR/config/config_sample/$f.sample.yaml"
    if [ -f "$dest" ]; then
        warn "config/$f.yaml already exists — left untouched"
    elif [ -f "$src" ]; then
        cp "$src" "$dest"
        info "Created config/$f.yaml from sample"
    fi
done

missing=()
for f in device.yaml models.yaml cameras.yaml; do
    [ -f "$SCRIPT_DIR/config/$f" ] || missing+=("$f")
done
if [ ${#missing[@]} -gt 0 ]; then
    warn "Still need real, device-specific values in: ${missing[*]}"
    warn "See config/config_sample/ for a fully-commented reference of each field."
else
    info "All config files present"
fi

if [ -z "$(ls -A "$SCRIPT_DIR/models" 2>/dev/null)" ]; then
    warn "models/ is empty — copy your model weights (.pt/.onnx/.engine/.hef) there before running"
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "  ── Installation complete ──────────────────────────────────"
echo ""
if [ -n "$POST_INSTALL_NOTE" ]; then
    warn "$POST_INSTALL_NOTE"
    echo ""
fi
echo "  Run manually:      $PYTHON_VENV main.py"
echo "  Run as a service:  sudo bash scripts/service.sh install"
echo ""
