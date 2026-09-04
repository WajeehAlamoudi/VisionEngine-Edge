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


def _probe_source(source: str) -> tuple[int, int, float]:
    """
    Read the stream's resolution and frame rate before the pipeline is built.

    nvstreammux validates its width and height during the state change, which
    happens before nvurisrcbin has connected and produced a pad — so the size
    cannot be discovered from the source and applied afterwards. NVIDIA's own
    sample apps set it to a hardcoded value for the same reason.

    It has to be the source's real size rather than anything else: the muxer
    scales every stream to it, so the boxes come back in that space, and the
    zone polygons in cameras.yaml are drawn in source pixels.

    The frame rate matters as much as the size: it is what turns the camera's
    fps_target into a number of frames to drop, and dropping them before the
    decoder is the only place the work is genuinely saved.

    Opened with OpenCV because that code is already here and proven against
    these cameras. It costs one connection and one decoded frame, once, at
    startup — not per frame, which is the cost this runtime exists to avoid.
    """
    import cv2

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {source} to read its resolution")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not (0.0 < fps <= 240.0):
            fps = 0.0        # unreported or nonsense; caller falls back

        # Decode rather than trusting the reported size: a stream with
        # non-standard SPS headers reports zeros until a frame arrives.
        for _ in range(120):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame.shape[1], frame.shape[0], fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > 0 and height > 0:
            return width, height, fps
        raise RuntimeError(f"no frame from {source} — cannot determine its resolution")
    finally:
        cap.release()


def _describe_tracker(path: str) -> str:
    """
    Name the algorithm an nvtracker config actually selects.

    The file decides which of DeepStream's trackers runs, but nothing in its
    name has to say so — every one of them is valid under any filename. Two
    sections settle it: VisualTracker turns on NvDCF's correlation filter, and
    ReID turns on an appearance network. Reported at startup so the log says
    what is running rather than only which file was read.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return "unknown (config unreadable)"

    def flag(name: str) -> int:
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if line.startswith(f"{name}:"):
                value = line.partition(":")[2].strip()
                return int(value) if value.lstrip("-").isdigit() else 0
        return 0

    dcf = flag("visualTrackerType") == 1
    reid = flag("reidType") == 1
    motion = flag("stateEstimatorType") > 0

    if dcf and reid:
        return "NvDCF + ReID (correlation filter and an appearance network)"
    if dcf:
        return "NvDCF (correlation filter, no ReID network)"
    if reid:
        return "NvDeepSORT (appearance network, no correlation filter)"
    if motion:
        return "NvSORT (motion only)"
    return "IOU (overlap only)"


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

    def __init__(self, cam: CameraConfig, model: ModelConfig,
                 with_frames: bool = False) -> None:
        super().__init__(cam)
        self._model = model
        # Pulling pixels back to host memory costs a conversion and a copy per
        # frame — the exact cost this runtime exists to avoid — so it is off
        # unless something actually needs an image. The debug overlay does.
        self._with_frames = with_frames
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

        # How many source frames to keep one of, applied by the decoder. 1 means
        # keep everything. Set once the source's frame rate is known.
        self._drop_interval = 1
        # The remainder the decoder's whole-number interval cannot express,
        # applied by a probe on the decoded pad. Both are set in open().
        self._gate_target = 0      # frames wanted per _gate_source frames
        self._gate_source = 0
        self._gate_credit = 0
        self._last_read = 0.0

    def _plan_rate(self, source_fps: float) -> None:
        """
        Work out how to get from the source's frame rate down to fps_target.

        Two gates, because one cannot do it alone.

        The decoder's drop-frame-interval is a whole number — keep one frame in
        N — and that is not a limitation of the property but of the video: a
        P-frame is a delta from the frames before it, so an arbitrary encoded
        frame cannot be discarded without corrupting the ones that follow. Only
        whole-number decimation is safe before decoding.

        Whatever ratio that leaves is handled by a probe on the decoded pad,
        which can drop any frame it likes. That is after NVDEC but before the
        muxer, so a frame dropped there still costs no scaling, no inference
        and no tracking — all the GPU work worth saving.

        25 fps down to 15 is the case that needs both: 25/15 is not a whole
        number, so the decoder keeps everything and the probe passes 3 of every
        5. At 30 down to 15 the decoder does it alone and the probe passes
        everything.
        """
        target = self._cam.fps_target
        if not target or not source_fps or source_fps <= target:
            self._drop_interval = 1
            self._gate_target = self._gate_source = 0
            return

        # Whole-number part, free because it happens before decoding.
        self._drop_interval = max(1, int(source_fps // target))
        decoded_fps = source_fps / self._drop_interval

        if decoded_fps <= target + 0.01:
            self._gate_target = self._gate_source = 0
            return

        # The remainder, as a ratio the probe can apply exactly.
        self._gate_target = int(round(target * 100))
        self._gate_source = int(round(decoded_fps * 100))
        self._gate_credit = 0

    def _rate_gate(self, _pad, _info) -> object:
        """
        Pass fps_target frames out of every source frame, exactly.

        A running credit rather than a counter, so the kept frames are spread
        evenly instead of arriving in a burst followed by a gap — 3 of every 5
        comes out as keep, drop, keep, drop, keep, which is what a tracker
        wants from its input.
        """
        Gst = self._gst
        self._gate_credit += self._gate_target
        if self._gate_credit >= self._gate_source:
            self._gate_credit -= self._gate_source
            return Gst.PadProbeReturn.OK
        return Gst.PadProbeReturn.DROP

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
        uri = self._source_uri()
        self._detections.load_labels()

        # Before the pipeline exists: nvstreammux needs its output size at
        # build time, not once the source has connected.
        probe_target = uri if not uri.startswith("file://") else str(self._cam.source)
        width, height, source_fps = _probe_source(probe_target)
        self._size = (width, height)
        self._plan_rate(source_fps)
        log.info("camera '%s': source is %dx%d @ %s fps", self._cam.id, width, height,
                 f"{source_fps:.0f}" if source_fps else "unknown")
        if source_fps and self._cam.fps_target:
            kept = source_fps / self._drop_interval
            if self._gate_source:
                kept = kept * self._gate_target / self._gate_source
            log.info("camera '%s': rate gate — decoder keeps 1 in %d, probe passes "
                     "%s, so %.1f fps reach nvinfer (target %d)",
                     self._cam.id, self._drop_interval,
                     f"{self._gate_target}/{self._gate_source}" if self._gate_source else "all",
                     kept, self._cam.fps_target)

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
            "camera '%s': DeepStream ready — %d classes, model %s",
            self._cam.id, self._detections.class_count, self._model.path,
        )
        if self._tracking:
            log.info("camera '%s': tracker is %s  [%s]",
                     self._cam.id, _describe_tracker(self._model.tracker),
                     self._model.tracker)
        else:
            log.info("camera '%s': tracking off (use_tracker: false)", self._cam.id)

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
        # Drop to fps_target here, ahead of the decoder, so the frames that are
        # not wanted cost nothing anywhere downstream.
        if self._drop_interval > 1 and _has_property(source, "drop-frame-interval"):
            source.set_property("drop-frame-interval", self._drop_interval)
            log.info("camera '%s': keeping 1 frame in %d before decode "
                     "(fps_target %d)", self._cam.id, self._drop_interval,
                     self._cam.fps_target)
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
        # Set here, before the state change that validates them. Every source is
        # scaled to this, so it is the camera's own resolution — that keeps the
        # boxes in the coordinate space the zone polygons are drawn in.
        streammux.set_property("width", self._size[0])
        streammux.set_property("height", self._size[1])

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
        if self._with_frames:
            # BGR straight out, so read() hands back what OpenCV expects
            # without another conversion on the Python side.
            appsink.set_property(
                "caps", Gst.Caps.from_string("video/x-raw, format=BGR"))

        # nvurisrcbin has no pad until it has connected, so this is linked in
        # the callback below rather than here.
        source.connect("pad-added", self._on_pad_added, streammux)

        # Frames leave the GPU only on the way to an appsink that was asked
        # for them; otherwise the buffer reaching appsink is still NVMM and
        # carries nothing but metadata.
        outconv = make("nvvideoconvert") if self._with_frames else None
        if outconv is not None and _has_property(outconv, "compute-hw"):
            # The VIC cannot produce BGR — the same limitation that took down
            # an earlier version of this pipeline.
            outconv.set_property("compute-hw", 1)

        tail = streammux
        for el in (pgie, tracker, outconv, appsink):
            if el is not None:
                tail.link(el)
                tail = el

        self._streammux = streammux
        self._appsink = appsink
        return pipeline

    def _on_pad_added(self, _element, pad, streammux) -> None:
        """
        Link the source once it has connected and knows what it is carrying.

        Only linking happens here. The frame size was settled before the
        pipeline was built, because nvstreammux will not leave NULL without it.
        """
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps.to_string().startswith("video"):
            return

        # get_request_pad was deprecated in GStreamer 1.20 for
        # request_pad_simple. Resolved by attribute and checked for None: a
        # missing method would raise inside this callback, where PyGObject
        # swallows the exception, and open() would then wait out its timeout
        # with no idea why.
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

        # Ahead of the link, so a dropped frame never reaches nvstreammux and
        # therefore never reaches nvinfer or the tracker.
        if self._gate_source:
            pad.add_probe(self._gst.PadProbeType.BUFFER, self._rate_gate)

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

        # No wait here. The decoder is already delivering at fps_target, so
        # this blocks on the next frame the pipeline actually produced rather
        # than sleeping through work it has already done.
        self._check_bus()
        sample = self._appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
        if sample is None:
            self._check_bus()
            raise SourceUnavailable("no output from the DeepStream pipeline")

        self._last_read = time.time()
        width, height = self._size
        detections = self._detections.parse(self._pyds, sample, width, height)
        return self._frame_from(sample), detections

    def _frame_from(self, sample):
        """
        The decoded frame, when one was asked for, as a BGR array.

        Returns None otherwise — the buffer is still in GPU memory and there is
        nothing to map. Callers already treat None as "this runtime has no
        pixels", which is the normal case.
        """
        if not self._with_frames:
            return None

        import numpy as np

        buf = sample.get_buffer()
        ok, info = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return None
        try:
            width, height = self._size
            # Copied before unmapping: the array is a view onto memory
            # GStreamer takes back as soon as the buffer is released.
            return np.ndarray(
                shape=(height, width, 3), dtype=np.uint8, buffer=info.data
            ).copy()
        finally:
            buf.unmap(info)

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
