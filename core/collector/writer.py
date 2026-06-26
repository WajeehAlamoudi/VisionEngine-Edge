from __future__ import annotations

import json
from pathlib import Path

import cv2

from core.model import InferenceResult


def _save_files(
        session_dir: Path,
        stem: str,
        frame,
        results: list[InferenceResult],
        save_cfg,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)

    if save_cfg.raw:
        cv2.imwrite(str(session_dir / f"{stem}_raw.jpg"), frame)

    if save_cfg.annotated:
        annotated = _draw_boxes(frame, results)
        cv2.imwrite(str(session_dir / f"{stem}.jpg"), annotated)

    if save_cfg.metadata:
        meta = {
            "detections": [
                {
                    "class":      r.class_name,
                    "confidence": round(r.confidence, 4),
                    "bbox":       [round(v, 1) for v in r.bbox],
                }
                for r in results
            ],
        }
        (session_dir / f"{stem}.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )


def _draw_boxes(frame, results: list[InferenceResult]):
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r.bbox)
        label = f"{r.class_name} {r.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out, label, (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return out


def _filename_stem(ts: str, camera_id: str) -> str:
    # "2026-06-22T08:14:33Z" → "2026-06-22_08-14-33_cam-01"
    clean = ts.replace("T", "_").replace(":", "-").rstrip("Z")
    safe_cam = camera_id.replace("/", "-").replace(" ", "_")
    return f"{clean}_{safe_cam}"
