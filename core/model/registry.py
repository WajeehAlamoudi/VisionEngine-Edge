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
            tracker: str = "botsort.yaml",
    ) -> None:
        shared: dict[str, ModelRunner] = {}   # model_id → shared runner (predict-only cameras)

        for cam in cameras:
            if cam.model_id not in models:
                raise RuntimeError(
                    f"ModelRegistry: model_id '{cam.model_id}' not found — "
                    f"this should have been caught by config validation"
                )
            model_cfg = models[cam.model_id]
            if model_cfg.use_tracker:
                # tracker is stateful — each camera needs its own dedicated runner
                runner = ModelRunner(model_cfg, use_tracker=True, tracker=tracker)
                runner.load()
                log.info("model '%s' loaded (dedicated+tracker) for camera '%s'", cam.model_id, cam.id)
            else:
                # stateless predict — safe to share across cameras using the same model
                if cam.model_id not in shared:
                    shared[cam.model_id] = ModelRunner(model_cfg)
                    shared[cam.model_id].load()
                    log.info("model '%s' loaded (shared) for camera '%s'", cam.model_id, cam.id)
                runner = shared[cam.model_id]

            self._runners[cam.id] = runner

    def get(self, camera_id: str) -> ModelRunner:
        return self._runners[camera_id]
