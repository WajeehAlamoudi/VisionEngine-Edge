from __future__ import annotations

import logging

from core.config import CameraConfig, ModelConfig
from core.config.model import needs_model_runner
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
        shared: dict[str, ModelRunner] = {}   # model_id → shared runner (predict-only cameras)

        for cam in cameras:
            if cam.model_id not in models:
                raise RuntimeError(
                    f"ModelRegistry: model_id '{cam.model_id}' not found — "
                    f"this should have been caught by config validation"
                )
            model_cfg = models[cam.model_id]
            # Models whose runtime opens the camera itself run entirely in
            # core/model/detector/ and never touch a ModelRunner, so building one
            # would load an engine nothing would ever call.
            if not needs_model_runner(model_cfg):
                log.info("model '%s' runs in the %s camera runtime for camera '%s'",
                         cam.model_id, model_cfg.runtime, cam.id)
                continue

            if model_cfg.use_tracker:
                # tracker is stateful — each camera needs its own dedicated runner
                runner = ModelRunner(model_cfg, use_tracker=True, tracker=model_cfg.tracker)
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

    def get(self, camera_id: str) -> ModelRunner | None:
        """The runner for a camera, or None when its runtime supplies its own."""
        return self._runners.get(camera_id)

    def close(self) -> None:
        """
        Release every runner's resources on shutdown.

        Deduplicated by identity because predict-only cameras share a runner —
        the map is keyed by camera, so several keys can point at one instance.
        Each close is guarded: a backend that fails to shut down cleanly must
        not stop the others from being released.
        """
        seen: set[int] = set()
        for camera_id, runner in self._runners.items():
            if id(runner) in seen:
                continue
            seen.add(id(runner))
            try:
                runner.close()
            except Exception:
                log.exception("error closing model runner for camera '%s'", camera_id)
        self._runners.clear()
