from __future__ import annotations

import logging

from core.config import CameraConfig, ModelConfig
from .runner import ModelRunner

log = logging.getLogger(__name__)


class ModelRegistry:
    """
    Loads and holds model runners for all enabled cameras.
    Cameras sharing the same model_id share one ModelRunner instance to save RAM.
    """

    def __init__(self) -> None:
        self._runners: dict[str, ModelRunner] = {}   # camera_id → ModelRunner

    def load_for_cameras(
            self,
            models: dict[str, ModelConfig],
            cameras: list[CameraConfig],
    ) -> None:
        shared: dict[str, ModelRunner] = {}   # model_id → shared runner

        for cam in cameras:
            if cam.model_id not in models:
                raise RuntimeError(
                    f"ModelRegistry: model_id '{cam.model_id}' not found — "
                    f"this should have been caught by config validation"
                )
            if cam.model_id not in shared:
                runner = ModelRunner(models[cam.model_id])
                runner.load()
                shared[cam.model_id] = runner
                log.info("model '%s' loaded (shared) for camera '%s'", cam.model_id, cam.id)

            self._runners[cam.id] = shared[cam.model_id]

    def get(self, camera_id: str) -> ModelRunner:
        return self._runners[camera_id]
