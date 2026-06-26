from __future__ import annotations

import logging

from core.config import CameraConfig, ModelConfig
from .runner import ModelRunner

log = logging.getLogger(__name__)


class ModelRegistry:
    """
    Loads and holds model runners for all enabled cameras.

    Memory strategy:
      - Cameras that DO NOT need tracking share one ModelRunner per model_id
        (YOLO weights loaded once, reused — saves RAM).
      - Cameras that DO need tracking each get a dedicated ModelRunner
        (tracker state is per-stream and cannot be shared between cameras).

    Access: get(camera_id) → always returns the right runner for that camera.
    """

    def __init__(self) -> None:
        self._runners: dict[str, ModelRunner] = {}   # camera_id → ModelRunner

    def load_for_cameras(
            self,
            models: dict[str, ModelConfig],
            cameras: list[CameraConfig],
    ) -> None:
        shared: dict[str, ModelRunner] = {}   # model_id → shared runner (no tracker)

        for cam in cameras:
            if cam.model_id not in models:
                raise RuntimeError(
                    f"ModelRegistry: model_id '{cam.model_id}' not found — "
                    f"this should have been caught by config validation"
                )
            needs_tracker = cam.analytics.needs_tracker

            if needs_tracker:
                runner = ModelRunner(
                    models[cam.model_id],
                    use_tracker=True,
                    tracker=cam.analytics.tracker,
                )
                runner.load()
                self._runners[cam.id] = runner
                log.info(
                    "model '%s' loaded (dedicated, tracker=%s) for camera '%s'",
                    cam.model_id, cam.analytics.tracker, cam.id,
                )
            else:
                if cam.model_id not in shared:
                    runner = ModelRunner(models[cam.model_id], use_tracker=False)
                    runner.load()
                    shared[cam.model_id] = runner
                self._runners[cam.id] = shared[cam.model_id]

    def get(self, camera_id: str) -> ModelRunner:
        return self._runners[camera_id]
