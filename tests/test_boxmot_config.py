from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import yaml
from boxmot.trackers.base import BaseTracker

from core.config import ModelConfig
from core.model.tracker.boxmot_registry import (
    BOXMOT_ALGORITHMS,
    COMMON_DEFAULTS,
    resolve_params,
)
from core.model.tracker.boxmot_tracker import BoxMotTracker
from core.model.types import InferenceResult


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "config" / "config_sample" / "boxmot_tracker.sample.yaml"


def model_config(tracker: str) -> ModelConfig:
    return ModelConfig(
        id="test-model",
        name="Test",
        version="1",
        path="unused.pt",
        runtime="ultralytics",
        accelerator="cpu",
        classes=["person"],
        confidence_threshold=0.5,
        iou_threshold=0.45,
        input_size=[640, 640],
        use_tracker=True,
        tracker=tracker,
        half=False,
    )


class BoxMotConfigTests(unittest.TestCase):
    def test_sample_contains_only_selected_algorithm_and_validates(self):
        sample_text = SAMPLE.read_text(encoding="utf-8")
        raw = yaml.safe_load(sample_text)
        self.assertEqual(raw["algorithm"], "bytetrack")
        self.assertIn("params", raw)
        self.assertNotIn("algorithms", raw)
        for name in set(BOXMOT_ALGORITHMS) - {"bytetrack"}:
            self.assertIn(f"# algorithm: {name}", sample_text)

        algorithm, params, reid = BoxMotTracker(
            model_config(str(SAMPLE))
        )._load_config()

        self.assertEqual(algorithm, "bytetrack")
        self.assertEqual(params["frame_rate"], 12)
        self.assertEqual(reid["backend"], "pytorch")

    def test_registry_matches_boxmot_19_constructor_parameters(self):
        base = set(inspect.signature(BaseTracker.__init__).parameters) - {
            "self", "class_ids", "class_names", "is_obb", "kwargs"
        }
        self.assertEqual(set(COMMON_DEFAULTS), base)

        for name, spec in BOXMOT_ALGORITHMS.items():
            accepted = set(base)
            for cls in spec.load_class().__mro__:
                if "__init__" in cls.__dict__:
                    accepted.update(inspect.signature(cls.__init__).parameters)
            self.assertFalse(
                set(spec.defaults) - accepted,
                f"{name} registry contains parameters absent from BoxMOT 19",
            )

    def test_every_algorithm_constructs_from_its_complete_defaults(self):
        for name, spec in BOXMOT_ALGORITHMS.items():
            params = resolve_params(name, {}, {})
            if spec.needs_reid(params):
                params["reid_model"] = object()
            tracker = spec.load_class()(**params)
            self.assertIsNotNone(tracker, name)

    def test_selected_bytetrack_profile_preserves_id_across_frames(self):
        tracker = BoxMotTracker(model_config(str(SAMPLE)))
        tracker.load()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detection = InferenceResult("person", 0.9, [10, 10, 40, 80], None)

        first = tracker.update(frame, [detection])
        second = tracker.update(frame, [detection])

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_unknown_parameter_fails_in_selected_profile(self):
        raw = yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))
        raw["params"]["misspelled_threshold"] = 0.5
        reader = mock_open(read_data=yaml.safe_dump(raw))
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "open", reader):
            with self.assertRaisesRegex(RuntimeError, "misspelled_threshold"):
                BoxMotTracker(model_config("boxmot_tracker.yaml"))._load_config()

    def test_other_algorithm_profiles_are_not_required(self):
        raw = yaml.safe_load(SAMPLE.read_text(encoding="utf-8"))
        reader = mock_open(read_data=yaml.safe_dump(raw))
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "open", reader):
            algorithm, params, _ = BoxMotTracker(
                model_config("boxmot_tracker.yaml")
            )._load_config()
        self.assertEqual(algorithm, "bytetrack")
        self.assertNotIn("with_reid", params)

    def test_legacy_flat_botsort_config_remains_deployable(self):
        raw = {
            "algorithm": "botsort",
            "with_reid": False,
            "use_cmc": False,
            "frame_rate": 12,
            "track_buffer": 75,
        }
        reader = mock_open(read_data=yaml.safe_dump(raw))
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "open", reader):
            algorithm, params, _ = BoxMotTracker(
                model_config("botsort_tracker.yaml")
            )._load_config()

        self.assertEqual(algorithm, "botsort")
        self.assertFalse(params["with_reid"])
        self.assertEqual(params["frame_rate"], 12)


if __name__ == "__main__":
    unittest.main()
