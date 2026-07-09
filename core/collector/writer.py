from __future__ import annotations

from pathlib import Path

import cv2

from core.model import InferenceResult


def _save_files(
        session_dir: Path,
        stem: str,
        frame,
        results: list[InferenceResult],
        class_ids: dict[str, int],
) -> None:
    """
    Write one clean image + one YOLO-format label file, in the standard
    training-dataset layout:

        <session>/images/<stem>.jpg
        <session>/labels/<stem>.txt

    Label lines: "<class_id> <cx> <cy> <w> <h>" — all normalized 0-1.
    An empty label file is written when there are no detections (a valid
    YOLO background/negative sample).
    """
    images_dir = session_dir / "images"
    labels_dir = session_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)

    h, w = frame.shape[:2]
    lines: list[str] = []
    for r in results:
        cid = class_ids.get(r.class_name)
        if cid is None:
            continue
        x1, y1, x2, y2 = r.bbox
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    (labels_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_dataset_yaml(session_dir: Path, class_names: list[str]) -> None:
    """
    Write the YOLO dataset descriptor. The class ids here are the source of
    truth for every label in this session — keep this file with the images/
    and labels/ folders when moving or merging the dataset.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        "# VisionEngine Edge — collected dataset\n"
        "# Class ids below define the labels. Ordering comes from the camera's\n"
        "# model class list in models.yaml. Keep this file with images/ + labels/.\n"
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(class_names)}\n"
        "names:\n"
        f"{names_block}\n"
    )
    (session_dir / "data.yaml").write_text(content, encoding="utf-8")


def _filename_stem(ts: str, camera_id: str) -> str:
    # "2026-06-22T08:14:33Z" → "2026-06-22_08-14-33_cam-01"
    clean = ts.replace("T", "_").replace(":", "-").rstrip("Z")
    safe_cam = camera_id.replace("/", "-").replace(" ", "_")
    return f"{clean}_{safe_cam}"
