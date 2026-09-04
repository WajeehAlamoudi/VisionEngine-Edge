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
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*" >&2; exit 1; }

# ── banner ─────────────────────────────────────────────────────────────────────
print_banner() {
    echo -e "${CYAN}"
    cat <<'BANNER'
  ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ███████╗██████╗  ██████╗ ███████╗
  ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ██╔════╝██╔══██╗██╔════╝ ██╔════╝
  ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║    █████╗  ██║  ██║██║  ███╗█████╗
  ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║    ██╔══╝  ██║  ██║██║   ██║██╔══╝
   ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ███████╗██████╔╝╚██████╔╝███████╗
    ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝   ╚══════╝╚═════╝  ╚═════╝ ╚══════╝
BANNER
    echo -e "${NC}"
    echo -e "  ${BOLD}Edge-native computer vision. Detect. Track. Ingest.${NC}"
    echo ""
}

# ── spinner ────────────────────────────────────────────────────────────────────
# Runs a command in the background with a live spinner in front of it — mainly
# for the --quiet pip steps below, which otherwise print nothing for a while
# and make it look like the installer has frozen.
run_with_spinner() {
    local label="$1"; shift
    local log; log="$(mktemp)"
    ("$@" >"$log" 2>&1) &
    local pid=$!
    local frames='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    while kill -0 "$pid" 2>/dev/null; do
        printf "\r  ${CYAN}%s${NC} %s" "${frames:$i:1}" "$label"
        i=$(( (i + 1) % ${#frames} ))
        sleep 0.1
    done
    if wait "$pid"; then
        printf "\r${GREEN}[✓]${NC} %s          \n" "$label"
        rm -f "$log"
        return 0
    else
        printf "\r${RED}[✗]${NC} %s          \n" "$label"
        echo "  ── output ──"
        cat "$log"
        rm -f "$log"
        error "Step failed: $label"
    fi
}

# ── validated input ────────────────────────────────────────────────────────────
# Every interactive prompt in this script goes through one of these three —
# reject anything that isn't a real answer and ask again, instead of silently
# falling through to a default that may be wrong for this device.

# $1 = prompt text, $2 = default value, $3 = space-separated list of valid values
prompt_choice() {
    local prompt="$1" default="$2" valid="$3" answer v
    while true; do
        read -r -p "$prompt [$default]: " answer
        answer="${answer:-$default}"
        for v in $valid; do
            [ "$answer" = "$v" ] && { echo "$answer"; return 0; }
        done
        echo "  Not a valid choice — enter one of: $valid" >&2
    done
}

# $1 = prompt text — accepts digits only, no default (a number is required)
prompt_number() {
    local prompt="$1" answer
    while true; do
        read -r -p "$prompt: " answer
        [[ "$answer" =~ ^[0-9]+$ ]] && { echo "$answer"; return 0; }
        echo "  Not a valid number — digits only, try again." >&2
    done
}

# $1 = prompt text, $2 = default ("y" or "n") — echoes "y" or "n"
prompt_yes_no() {
    local prompt="$1" default="${2:-n}" answer label
    [ "$default" = "y" ] && label="Y/n" || label="y/N"
    while true; do
        read -r -p "$prompt [$label]: " answer
        answer="${answer:-$default}"
        case "$answer" in
            [Yy]|[Yy][Ee][Ss]) echo "y"; return 0 ;;
            [Nn]|[Nn][Oo])     echo "n"; return 0 ;;
            *) echo "  Please answer y or n." >&2 ;;
        esac
    done
}

# $1 = prompt text, $2 = default (optional — if omitted, empty answers are rejected)
prompt_text() {
    local prompt="$1" default="${2:-}" answer
    while true; do
        if [ -n "$default" ]; then
            read -r -p "$prompt [$default]: " answer
            answer="${answer:-$default}"
        else
            read -r -p "$prompt: " answer
        fi
        [ -n "$answer" ] && { echo "$answer"; return 0; }
        echo "  This can't be empty — try again." >&2
    done
}

print_banner

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
echo "    5) NVIDIA Jetson + DeepStream (PeopleNet / nvinfer + nvtracker)"
echo ""
device_choice="$(prompt_choice "  Choice" "1" "1 2 3 4 5")"

USE_SYSTEM_SITE_PACKAGES=0
TORCH_INSTALL_CMD=""
POST_INSTALL_NOTE=""
SETUP_DEEPSTREAM=0

# what models.yaml's own runtime:/accelerator: fields should be for this hardware
MODEL_RUNTIME="ultralytics"
case "$device_choice" in
    2|3) MODEL_ACCELERATOR="cuda" ;;
    4)   MODEL_ACCELERATOR="mps"  ;;
    5)   MODEL_RUNTIME="deepstream"; MODEL_ACCELERATOR="cuda" ;;
    *)   MODEL_ACCELERATOR="cpu"  ;;
esac

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
            jp_version="$(prompt_number "  JetPack major version (e.g. 6)")"
            cuda_version="$(prompt_number "  CUDA version on this JetPack, no dot (e.g. 126 for 12.6)")"
            echo ""
            warn "Run this BEFORE re-running this installer, so the venv can find it:"
            echo "    $PYTHON -m pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp${jp_version}/cu${cuda_version}"
            echo ""
            cont="$(prompt_yes_no "  Continue anyway without confirmed GPU torch?" "n")"
            [ "$cont" = "y" ] || error "Install the wheel above, then re-run this script."
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
        echo ""
        info "Jetson + DeepStream — checking the SDK before anything is installed"

        DS_ROOT="/opt/nvidia/deepstream/deepstream"
        [ -d "$DS_ROOT" ] || error "DeepStream not found at $DS_ROOT. Install it through SDK Manager or apt (deepstream-<version>), then re-run."

        if [ -f "$DS_ROOT/version" ]; then
            info "DeepStream: $(tr -d '\0' < "$DS_ROOT/version" | head -1)"
        else
            warn "DeepStream found but its version file is missing — continuing"
        fi

        [ -f "$DS_ROOT/lib/libnvds_nvmultiobjecttracker.so" ] \
            || error "DeepStream is present but the tracker library is missing: $DS_ROOT/lib/libnvds_nvmultiobjecttracker.so"
        info "NvTracker library present"

        # GStreamer's Python bindings are separate from DeepStream's own
        # plugins — the C elements being installed does not imply gi is. The
        # pipeline is built from Python, so both are needed.
        if ! "$PYTHON" -c "import gi; gi.require_version('Gst','1.0'); from gi.repository import Gst" 2>/dev/null; then
            warn "GStreamer Python bindings (gi) are missing."
            gi_install="$(prompt_yes_no "  Install python3-gi and python3-gst-1.0 with apt now?" "y")"
            if [ "$gi_install" = "y" ]; then
                run_with_spinner "Installing GStreamer Python bindings..." \
                    sudo apt-get install -y python3-gi python3-gst-1.0 gir1.2-gstreamer-1.0
            else
                error "gi is required for device: deepstream. Install it and re-run."
            fi
        fi
        info "GStreamer Python bindings present"

        # System packages, so the venv has to be able to see them. Same reason
        # the plain Jetson path uses this flag for torch.
        USE_SYSTEM_SITE_PACKAGES=1
        SETUP_DEEPSTREAM=1

        # torch is not used for inference here — nvinfer is — but device
        # auto-selection and the boxmot tracker still import it, so a working
        # build must exist for the rest of the project.
        if "$PYTHON" -c "import torch" 2>/dev/null; then
            info "torch present — venv will reuse it"
        else
            warn "torch is not installed system-wide. It is not used for DeepStream inference,"
            warn "but other backends and config validation import it. See requirements.txt"
            warn "for the JetPack-matched wheel index."
        fi

        if ! "$PYTHON" -c "import pyds" 2>/dev/null; then
            POST_INSTALL_NOTE="pyds is NOT installed, and device: deepstream cannot run without it. It does not ship with DeepStream and is not on PyPI: download the linux_aarch64 wheel matching YOUR DeepStream version from https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases and install it into the venv with '.venv/bin/pip install <wheel>'. A wheel built for a different DeepStream version will import cleanly and then crash on the first frame, so match the version exactly."
        else
            info "pyds present"
        fi
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

run_with_spinner "Upgrading pip..." "$PIP" install --quiet --upgrade pip

# ── torch (device-specific, if needed) ────────────────────────────────────────
if [ -n "$TORCH_INSTALL_CMD" ]; then
    run_with_spinner "Installing torch for this device..." bash -c "$TORCH_INSTALL_CMD"
fi

# ── dependencies ──────────────────────────────────────────────────────────────
# requirements.txt's own `torch>=2.0.0` line is satisfied by whatever was
# just set up above (system-wide via --system-site-packages, CPU-only, or
# left to the default PyPI resolution) — it will not be reinstalled or
# overridden here.
run_with_spinner "Installing dependencies from requirements.txt..." "$PIP" install --quiet -r "$SCRIPT_DIR/requirements.txt"

# ── runtime directories ───────────────────────────────────────────────────────
info "Creating runtime directories..."
mkdir -p "$SCRIPT_DIR/collected"
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/data"

# model weight files (.pt/.onnx/.engine) are gitignored — a fresh clone
# has nowhere to put them until this exists
mkdir -p "$SCRIPT_DIR/models"

# ── config ─────────────────────────────────────────────────────────────────────
# Every file here is copied as-is from config_sample/ — none of them get real
# content generated. Camera sources, model paths, device identity, and every
# other business-specific value are filled in manually after this script
# finishes, which is what keeps this installer identical for any deployment.
mkdir -p "$SCRIPT_DIR/config"
info "Copying config templates (skipping any that already exist)..."
for f in api notifications rules collection device models cameras botsort_tracker; do
    dest="$SCRIPT_DIR/config/$f.yaml"
    src="$SCRIPT_DIR/config/config_sample/$f.sample.yaml"
    if [ -f "$dest" ]; then
        warn "config/$f.yaml already exists — left untouched"
    elif [ -f "$src" ]; then
        cp "$src" "$dest"
        info "Created config/$f.yaml from sample"
    fi
done

# ── DeepStream config files ───────────────────────────────────────────────────
# Only for device option 6. These are not .sample.yaml files so they sit
# outside the loop above: two are plain text, and the tracker config is copied
# from the installed SDK so it always matches the DeepStream version on this
# device rather than a stale copy checked into the repo.
if [ "$SETUP_DEEPSTREAM" -eq 1 ]; then
    echo ""
    info "Setting up DeepStream config files..."

    src="$SCRIPT_DIR/config/config_sample/deepstream_infer.sample.txt"
    dest="$SCRIPT_DIR/config/deepstream_infer.txt"
    if [ -f "$dest" ]; then
        warn "config/deepstream_infer.txt already exists — left untouched"
    elif [ -f "$src" ]; then
        cp "$src" "$dest"
        info "Created config/deepstream_infer.txt from sample"
    fi

    # Written rather than shipped as a sample: it is three well-known names.
    # The file is required though — class_id indexes into the model's own
    # ordered class list, which models.yaml `classes` cannot supply because it
    # is a filter naming only the classes you want. Replace these lines when
    # swapping to another model.
    if [ -f "$SCRIPT_DIR/config/peoplenet_labels.txt" ]; then
        warn "config/peoplenet_labels.txt already exists — left untouched"
    else
        printf 'person\nbag\nface\n' > "$SCRIPT_DIR/config/peoplenet_labels.txt"
        info "Created config/peoplenet_labels.txt (person, bag, face)"
    fi

    DS_TRACKER_DIR="$DS_ROOT/samples/configs/deepstream-app"
    if [ -f "$SCRIPT_DIR/config/nvdcf_tracker.yml" ]; then
        warn "config/nvdcf_tracker.yml already exists — left untouched"
    else
        echo ""
        echo "  Which NvTracker algorithm? (changeable later — it is just this file)"
        echo "    1) NvDCF perf      — balanced, recommended"
        echo "    2) NvDCF accuracy  — slower, better through occlusion"
        echo "    3) NvSORT          — lightweight Kalman"
        echo "    4) IOU             — cheapest, weakest"
        echo "    5) NvDeepSORT      — strongest, runs a ReID network"
        echo ""
        tracker_choice="$(prompt_choice "  Choice" "1" "1 2 3 4 5")"
        case "$tracker_choice" in
            2) TRACKER_SRC="config_tracker_NvDCF_accuracy.yml" ;;
            3) TRACKER_SRC="config_tracker_NvSORT.yml"         ;;
            4) TRACKER_SRC="config_tracker_IOU.yml"            ;;
            5) TRACKER_SRC="config_tracker_NvDeepSORT.yml"     ;;
            *) TRACKER_SRC="config_tracker_NvDCF_perf.yml"     ;;
        esac
        # The SDK's own copy is preferred: it matches the DeepStream version
        # installed here, and NVIDIA revises these values between releases. The
        # bundled sample is only a fallback for an unusual install layout, and
        # it is an NvDCF config regardless of the choice above.
        if [ -f "$DS_TRACKER_DIR/$TRACKER_SRC" ]; then
            cp "$DS_TRACKER_DIR/$TRACKER_SRC" "$SCRIPT_DIR/config/nvdcf_tracker.yml"
            info "Created config/nvdcf_tracker.yml from the SDK's $TRACKER_SRC"
        elif [ -f "$SCRIPT_DIR/config/config_sample/nvdcf_tracker.sample.yml" ]; then
            cp "$SCRIPT_DIR/config/config_sample/nvdcf_tracker.sample.yml" \
               "$SCRIPT_DIR/config/nvdcf_tracker.yml"
            warn "$DS_TRACKER_DIR/$TRACKER_SRC not found — used the bundled NvDCF sample instead"
            warn "Prefer your SDK's own config if you can find it: it matches your DeepStream version"
        else
            warn "No tracker config available — copy one into config/nvdcf_tracker.yml manually"
        fi
    fi

    echo ""
    warn "models/ needs a TensorRT engine built ON THIS DEVICE — engines are not portable"
    warn "across GPUs or TensorRT versions. Download PeopleNet from NGC, then:"
    echo "    /usr/src/tensorrt/bin/trtexec --onnx=resnet34_peoplenet.onnx \\"
    echo "        --fp16 --saveEngine=$SCRIPT_DIR/models/peoplenet_fp16.engine"
fi

if [ -z "$(ls -A "$SCRIPT_DIR/models" 2>/dev/null)" ]; then
    warn "models/ is empty — copy your model weight file there before editing config/models.yaml"
fi

# ── done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "  ${GREEN}${BOLD}✓ VisionEngine Edge is ready on this device${NC}"
echo -e "  ${CYAN}────────────────────────────────────────────${NC}"
echo ""
if [ -n "$POST_INSTALL_NOTE" ]; then
    warn "$POST_INSTALL_NOTE"
    echo ""
fi
echo "  Run manually:      $PYTHON_VENV main.py"
echo "  Run as a service:  sudo bash scripts/service.sh install"
echo ""
