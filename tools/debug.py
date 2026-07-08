"""
VisionEngine Edge — Debug Tool

Three modes for configuring and verifying a camera deployment:

  view       Open the camera stream and display resolution + FPS.
             Use to verify the camera source is working.

  zones      Interactive zone builder. Click to draw polygons on the live frame.
             Outputs ready-to-paste YAML for cameras.yaml.

  inference  Run the configured model live. Shows bounding boxes, class names,
             confidence scores, zones, and real inference FPS.

Usage:
  python tools/debug.py --mode view      --source 0
  python tools/debug.py --mode zones     --source 0
  python tools/debug.py --mode zones     --source 0        --config config/
  python tools/debug.py --mode inference --camera cam-01   --config config/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── make sure project root is on sys.path ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.debug.viewer           import run as run_view
from tools.debug.zone_builder     import run as run_zones
from tools.debug.inference_viewer import run as run_inference


def _parse_args():
    p = argparse.ArgumentParser(
        description="VisionEngine Edge debug tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode",   required=True, choices=["view", "zones", "inference"],
                   help="view | zones | inference")
    p.add_argument("--config", default="config",
                   help="path to config directory (default: config)")
    p.add_argument("--camera", default=None,
                   help="camera id from cameras.yaml (required for inference mode)")
    p.add_argument("--source", default=None,
                   help="camera source override — int index, rtsp://, or file path "
                        "(view and zones modes; if omitted, --camera + config are used)")
    return p.parse_args()


def _resolve_source(args) -> str | int | None:
    if args.source is not None:
        src = args.source
        return int(src) if src.isdigit() else src
    return None


def main() -> None:
    args = _parse_args()

    # ── view mode ─────────────────────────────────────────────────────────────
    if args.mode == "view":
        source = _resolve_source(args)
        if source is None:
            source = _source_from_config(args)
        if source is None:
            print("ERROR  provide --source or --camera + --config")
            sys.exit(1)
        run_view(source)

    # ── zones mode ────────────────────────────────────────────────────────────
    elif args.mode == "zones":
        source = _resolve_source(args)
        existing_zones = None

        if source is None or args.camera:
            cfg = _load_config(args.config)
            if cfg and args.camera:
                cam = cfg.get_camera(args.camera)
                if cam:
                    if source is None:
                        source = cam.source
                    existing_zones = cam.zones if cam.zones else None

        if source is None:
            print("ERROR  provide --source or --camera + --config")
            sys.exit(1)

        run_zones(source, existing_zones=existing_zones)

    # ── inference mode ────────────────────────────────────────────────────────
    elif args.mode == "inference":
        if not args.camera:
            print("ERROR  --camera is required for inference mode")
            sys.exit(1)
        cfg = _load_config(args.config)
        if cfg is None:
            sys.exit(1)
        run_inference(cfg, args.camera)


def _load_config(config_dir: str):
    try:
        from core.config import load_config
        return load_config(config_dir)
    except SystemExit as e:
        print(f"Config error: {e}")
        return None
    except Exception as e:
        print(f"Failed to load config from '{config_dir}': {e}")
        return None


def _source_from_config(args) -> str | int | None:
    if not args.camera:
        return None
    cfg = _load_config(args.config)
    if cfg is None:
        return None
    cam = cfg.get_camera(args.camera)
    if cam is None:
        print(f"ERROR  camera '{args.camera}' not found in config")
        return None
    return cam.source


if __name__ == "__main__":
    main()
