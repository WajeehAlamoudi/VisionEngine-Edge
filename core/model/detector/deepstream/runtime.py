from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from core.config import CameraConfig, ModelConfig
from ...types import InferenceResult
from ..base import CameraRuntime, SourceUnavailable
from .detector import DeepStreamDetections

log = logging.getLogger(__name__)

# Gst.init() is global to the process and must run exactly once, however many
# cameras build a pipeline.
_GST_INIT_LOCK = threading.Lock()
_gst_ready = False

# Shipped with DeepStream; the single library behind IOU, NvSORT, NvDCF and
# NvDeepSORT. Which one runs is decided entirely by the config file named in
# models.yaml `tracker:`, never by this path.
_LL_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# NvTracker's internal working resolution. Must be multiples of 32; these are
# NVIDIA's own defaults and are neither the frame size nor the model input.
_TRACKER_W, _TRACKER_H = 640, 384

_PULL_TIMEOUT_NS = 5 * 1_000_000_000

# The source has to connect and the engines deserialise before the first frame
# appears, which on a Jetson is seconds even with the engines already built.
_OPEN_TIMEOUT_S = 60.0


def _has_property(element, name: str) -> bool:
    """Whether a GStreamer element exposes a property, without raising."""
    return element.find_property(name) is not None


def _import_gst():
    """Import GStreamer and initialise it once for the process."""
    global _gst_ready
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    with _GST_INIT_LOCK:
        if not _gst_ready:
            Gst.init(None)
            _gst_ready = True
    return Gst


class DeepStreamCameraRuntime(CameraRuntime):
    """
    RTSP straight into DeepStream: NVDEC decode, nvinfer, nvtracker.

        nvurisrcbin -> nvstreammux -> nvinfer -> [nvtracker] -> appsink

    No OpenCV and no Python frame. The stream is decoded by the hardware
    decoder into GPU memory and stays there through inference and tracking, so
    nothing crosses to host memory unless something asks for pixels.

    That is the whole reason this is a CameraRuntime rather than a Detector.
    A Detector is handed a frame; here there is deliberately no frame to hand,
    and detection and tracking are elements of the same pipeline that decoded
    it. Nothing in core/model/ changes to accommodate that.

    Which model runs is the nvinfer config in models.yaml `ds_infer_config` —
    PeopleNet, TrafficCamNet, DashCamNet, FaceDetect and any other
    DetectNet_v2 model are a config change with no code change. Which tracker
    runs, and whether it uses a ReID network, is the file named in `tracker:`.

    gi and pyds are imported inside open(), never at module import time, so a
    machine without DeepStream can still import this module and run every other
    runtime.
    """

    def __init__(self, cam: CameraConfig, model: ModelConfig) -> None:
        super().__init__(cam)
        self._model = model
        self._gst = None
        self._pyds = None

        self._pipeline = None
        self._streammux = None
        self._appsink = None
        self._bus = None

        self._size: tuple[int, int] = (0, 0)
        self._linked = threading.Event()
        # Why the pad callback gave up, if it did. Read by open() after the
        # wait times out, because an exception raised in that callback is
        # swallowed by PyGObject rather than propagating.
        self._link_error: str | None = None
        # The detection half: label map, filtering, metadata to rows.
        self._detections = DeepStreamDetections(model, cam.classes)
        self._failed: str | None = None

        self._frame_interval = 1.0 / cam.fps_target if cam.fps_target else 0.0
        self._last_read = 0.0

    @property
    def _tracking(self) -> bool:
        return bool(self._model.use_tracker)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._size

    # ── opening ───────────────────────────────────────────────────────────────

    def open(self) -> None:
        self._gst = _import_gst()
        import pyds
        self._pyds = pyds

        # Before anything is built, so an unusable source fails with its own
        # message rather than as a pipeline that never produces a frame.
        self._check_prerequisites()
        self._source_uri()
        self._detections.load_labels()

        Gst = self._gst
        self._pipeline = self._build()
        self._bus = self._pipeline.get_bus()

        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._check_bus()
            raise RuntimeError("pipeline refused to start (GST_DEBUG=3 for the element)")

        # nvurisrcbin exposes its pad only once it has connected and identified
        # the stream, so the source resolution is not known until then.
        if not self._linked.wait(_OPEN_TIMEOUT_S):
            self._check_bus()
            # The callback records why it gave up, since an exception raised
            # inside it would be swallowed by PyGObject and never reach here.
            raise RuntimeError(
                self._link_error
                or f"no video from {self._cam.source} within {_OPEN_TIMEOUT_S:.0f}s"
            )

        state, _, _ = self._pipeline.get_state(int(_OPEN_TIMEOUT_S * 1e9))
        if state == Gst.StateChangeReturn.FAILURE:
            self._check_bus()
            raise RuntimeError("pipeline failed while reaching PLAYING")

        log.info(
            "camera '%s': DeepStream ready — nvinfer(%s), tracking %s, %d classes",
            self._cam.id, self._model.ds_infer_config,
            f"via nvtracker({self._model.tracker})" if self._tracking else "off",
            self._detections.class_count,
        )

    def _source_uri(self) -> str:
        """
        cameras.yaml `source` as a URI nvurisrcbin can open.

        It takes a URI, not a path or an index, so a video file has to be
        turned into an absolute file:// URL. A USB index cannot be served at
        all: nvurisrcbin has no V4L2 support, and reaching a webcam would mean
        a v4l2src branch converting frames into GPU memory — reintroducing the
        host-side conversion this runtime exists to avoid. Refused with an
        explanation rather than half-supported.
        """
        source = self._cam.source

        if isinstance(source, int):
            raise RuntimeError(
                f"camera '{self._cam.id}': source {source} is a USB device "
                f"index, which the deepstream runtime cannot open — it reads "
                f"URIs through NVDEC. Use runtime: ultralytics for USB "
                f"cameras, or give this camera an rtsp:// source."
            )

        text = str(source)
        if text.startswith(("rtsp://", "rtsps://", "http://", "https://", "file://")):
            return text

        # A bare path — the third form cameras.yaml allows. Config validation
        # already confirmed the file exists.
        return Path(text).resolve().as_uri()

    def _check_prerequisites(self) -> None:
        if not self._model.ds_infer_config:
            raise RuntimeError(
                "runtime is 'deepstream' but ds_infer_config is not set — "
                "nvinfer has no configuration to load"
            )
        if self._tracking and not Path(_LL_LIB).is_file():
            raise FileNotFoundError(
                f"DeepStream's tracker library is missing at {_LL_LIB}"
            )

    # ── pipeline ──────────────────────────────────────────────────────────────

    def _build(self):
        Gst = self._gst
        pipeline = Gst.Pipeline.new(f"ve-{self._cam.id}")

        def make(factory: str):
            el = Gst.ElementFactory.make(factory, f"{factory}-{self._cam.id}")
            if el is None:
                raise RuntimeError(
                    f"GStreamer element '{factory}' could not be created — for "
                    f"nv* elements the DeepStream plugins are not on GST_PLUGIN_PATH"
                )
            pipeline.add(el)
            return el

        source = make("nvurisrcbin")
        streammux = make("nvstreammux")
        pgie = make("nvinfer")
        appsink = make("appsink")
        tracker = make("nvtracker") if self._tracking else None

        source.set_property("uri", self._source_uri())
        # Reconnect rather than ending the stream when the camera drops. Without
        # it a brief network blip terminates the pipeline permanently.
        if source.find_property("rtsp-reconnect-interval"):
            source.set_property("rtsp-reconnect-interval", 10)

        # batch-size 1: one pipeline per camera, matching how every other
        # runtime is arranged. Width and height are set once the source's caps
        # are known, in _on_pad_added.
        streammux.set_property("batch-size", 1)
        streammux.set_property("live-source", 1)
        streammux.set_property("batched-push-timeout", 40000)

        # Order matters: parsing config-file-path resets nvinfer's properties,
        # so the engine override has to follow it to stick. models.yaml `path`
        # stays the single source of truth for which weights run.
        pgie.set_property("config-file-path", self._model.ds_infer_config)
        # models.yaml `path` stays the single source of truth for which weights
        # run, but which property carries it depends on the format: a .engine is
        # a serialized TensorRT plan, an .onnx is a network nvinfer must build
        # from. Passing an .onnx as model-engine-file fails to deserialize.
        if self._model.path:
            suffix = Path(self._model.path).suffix.lower()
            key = "onnx-file" if suffix == ".onnx" else "model-engine-file"
            if _has_property(pgie, key):
                pgie.set_property(key, self._model.path)
            else:
                log.warning("camera '%s': nvinfer has no '%s' property — leaving "
                            "the model to %s", self._cam.id, key,
                            self._model.ds_infer_config)

        if tracker is not None:
            tracker.set_property("ll-lib-file", _LL_LIB)
            # The one line that selects the algorithm — IOU, NvSORT, NvDCF and
            # NvDeepSORT all live in the library above.
            tracker.set_property("ll-config-file", self._model.tracker)
            tracker.set_property("tracker-width", _TRACKER_W)
            tracker.set_property("tracker-height", _TRACKER_H)
            # Scale the crops on the GPU. On Jetson the tracker defaults to the
            # VIC, which cannot handle these formats and quietly raises its
            # minimum object size to 16x16 instead of failing — small or distant
            # people then get detections but never a track_id, with nothing
            # logged to say why. Guarded: dGPU builds have no VIC and no such
            # property.
            if _has_property(tracker, "compute-hw"):
                tracker.set_property("compute-hw", 1)

        # drop=true with a single buffer means the pipeline never blocks on a
        # slow consumer and read() always gets the newest frame rather than
        # working through a backlog.
        appsink.set_property("emit-signals", False)
        appsink.set_property("sync", False)
        appsink.set_property("max-buffers", 1)
        appsink.set_property("drop", True)

        # nvurisrcbin has no pad until it has connected, so this is linked in
        # the callback below rather than here.
        source.connect("pad-added", self._on_pad_added, streammux)

        tail = streammux
        for el in (pgie, tracker, appsink):
            if el is not None:
                tail.link(el)
                tail = el

        self._streammux = streammux
        self._appsink = appsink
        return pipeline

    def _on_pad_added(self, _element, pad, streammux) -> None:
        """
        Link the source once it knows what it is carrying.

        The source resolution arrives with this pad, which is why nvstreammux
        is sized here: it scales every stream to its own width and height, so
        setting it to the camera's native size is what keeps the boxes coming
        back in the coordinate space zones and rules already expect.
        """
        caps = pad.get_current_caps() or pad.query_caps(None)
        name = caps.to_string()
        if not name.startswith("video"):
            return

        structure = caps.get_structure(0)
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not (ok_w and ok_h) or width <= 0 or height <= 0:
            # Without dimensions the muxer keeps its default size and every box
            # would be clamped against 0x0 downstream — detections would vanish
            # with nothing reported. Better to leave _linked unset and let
            # open() time out with a message.
            self._link_error = (f"source caps carried no frame size ({name}) — "
                                f"cannot size nvstreammux")
            log.error("camera '%s': %s", self._cam.id, self._link_error)
            return

        self._size = (width, height)
        streammux.set_property("width", width)
        streammux.set_property("height", height)

        # get_request_pad was deprecated in GStreamer 1.20 for
        # request_pad_simple. Resolved by attribute, and checked for None: a
        # missing method would raise inside this callback, where PyGObject
        # swallows the exception and open() would hang until its timeout with a
        # misleading message.
        request = (getattr(streammux, "request_pad_simple", None)
                   or getattr(streammux, "get_request_pad", None))
        if request is None:
            self._link_error = ("nvstreammux exposes neither request_pad_simple "
                                "nor get_request_pad")
            log.error("camera '%s': %s", self._cam.id, self._link_error)
            return

        sink_pad = request("sink_0")
        if sink_pad is None:
            self._link_error = "could not obtain an nvstreammux sink pad"
            log.error("camera '%s': %s", self._cam.id, self._link_error)
            return

        if pad.link(sink_pad) != self._gst.PadLinkReturn.OK:
            self._link_error = "failed to link the source into nvstreammux"
            log.error("camera '%s': %s", self._cam.id, self._link_error)
            return

        log.info("camera '%s': source linked — %dx%d", self._cam.id, *self._size)
        self._linked.set()

    # ── reading ───────────────────────────────────────────────────────────────

    def read(self) -> tuple[object | None, list[InferenceResult]]:
        """
        Wait out the fps_target interval, then take the newest processed frame.

        Sleeping rather than pulling and discarding is what makes this cheap:
        the appsink drops whatever arrives meanwhile, so nothing is copied or
        converted for a frame that was never going to be used.

        The frame is always None — pixels stay in GPU memory. Rules, zones and
        storage need only the boxes.
        """
        if self._failed:
            raise RuntimeError(f"pipeline stopped and cannot recover — {self._failed}")
        if self._pipeline is None:
            raise RuntimeError("read() before open()")

        if self._frame_interval:
            wait = self._last_read + self._frame_interval - time.time()
            if wait > 0:
                time.sleep(wait)

        self._check_bus()
        sample = self._appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
        if sample is None:
            self._check_bus()
            raise SourceUnavailable("no output from the DeepStream pipeline")

        self._last_read = time.time()
        width, height = self._size
        return None, self._detections.parse(self._pyds, sample, width, height)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _check_bus(self) -> None:
        """
        Turn a pipeline failure into an exception.

        GStreamer reports errors on a bus rather than raising, so without this
        a dead pipeline would surface only as a pull timeout with no cause.
        """
        Gst = self._gst
        while True:
            message = self._bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is None:
                return

            if message.type == Gst.MessageType.EOS:
                self._failed = "the pipeline reached end-of-stream"
                raise SourceUnavailable(self._failed)

            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self._failed = f"{message.src.get_name()}: {err}"
                raise RuntimeError(f"pipeline error from {self._failed} ({debug})")

            err, _ = message.parse_warning()
            log.warning("camera '%s': %s: %s",
                        self._cam.id, message.src.get_name(), err)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            log.info("camera '%s': DeepStream pipeline stopped", self._cam.id)
