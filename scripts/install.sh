#!/usr/bin/env bash
# install.sh — set up VisionEngine Edge on a fresh device
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

# ── virtual environment ───────────────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at .venv — skipping creation"
else
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON_VENV="$VENV_DIR/bin/python"

info "Upgrading pip..."
"$PIP" install --quiet --upgrade pip

# ── torch install hint ────────────────────────────────────────────────────────
echo ""
echo "  Select PyTorch variant for this device:"
echo "    1) Standard (CUDA or CPU auto-detect) — default"
echo "    2) CPU-only  (smaller, good for Pi / Mac Mini without GPU)"
echo "    3) Skip      (Jetson: install manually via NVIDIA wheel)"
echo ""
read -r -p "  Choice [1]: " torch_choice
torch_choice="${torch_choice:-1}"

case "$torch_choice" in
    2)
        info "Installing CPU-only PyTorch..."
        "$PIP" install --quiet torch --index-url https://download.pytorch.org/whl/cpu
        ;;
    3)
        warn "Skipping PyTorch — install the NVIDIA Jetson wheel manually, then re-run this script."
        ;;
    *)
        # torch will be pulled in by the main requirements install below
        ;;
esac

# ── dependencies ──────────────────────────────────────────────────────────────
info "Installing dependencies from requirements.txt..."
"$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── runtime directories ───────────────────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p "$SCRIPT_DIR/collected"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data"

# ── config check ──────────────────────────────────────────────────────────────
if [ ! -d "$SCRIPT_DIR/config" ]; then
    warn "No config/ directory found. Copy your YAML config files before starting."
else
    missing=()
    for f in device.yaml api.yaml models.yaml cameras.yaml rules.yaml notifications.yaml collection.yaml; do
        [ -f "$SCRIPT_DIR/config/$f" ] || missing+=("$f")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        warn "Missing config files: ${missing[*]}"
    else
        info "All config files present"
    fi
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo "  ── Installation complete ──────────────────────────────────"
echo ""
echo "  Run manually:      $PYTHON_VENV main.py"
echo "  Run as a service:  sudo bash scripts/service.sh install"
echo ""
