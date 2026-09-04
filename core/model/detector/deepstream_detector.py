from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from core.config import ModelConfig
from ..stable_id import StableIdMap
from ..types import InferenceResult
from .base import Detector

log = logging.getLogger(__name__)

# Gst.init() is global to the process and must run exactly once, however many
# cameras build a pipeline. Guarded rather than left to each instance because
# ModelRegistry creates one detector per camera.
_GST_INIT_LOCK = threading.Lock()
_gst_ready = False

# Shipped with DeepStream; the single library behind IOU, NvSORT, NvDCF and
# NvDeepSORT. Which one runs is decided entirely by the config file in
# models.yaml `tracker:`, never by this path.
_LL_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

# NvTracker's internal working resolution. Must be multiples of 32; these are
# NVIDIA's own defaults and are not the frame size or the model input size.
_TRACKER_W, _TRACKER_H = 640, 384

# What NvDs writes into object_id when an object has not been assigned a track.
_UNTRACKED = (1 << 64) - 1

# How long to wait for a processed buffer before giving up on a frame.
_PULL_TIMEOUT_NS = 5 * 1_000_000_000

# How long to wait for the pipeline to finish reaching PLAYING. Generous: the
# first transition builds or deserialises the TensorRT engines, which on a
# Jetson takes seconds even when they are already on disk.
_START_TIMEOUT_NS = 60 * 1_000_000_000


def _has_property(element, name: str) -> bool:
    """Whether a GStreamer element exposes a property, without raising."""
    return element.find_property(name) is not None


def _import_gst():
    """
    Import GStreamer and initialise it once for the process.

    Returns the Gst module so callers can hold a reference instead of
    re-importing it in every method.
    """
    global _gst_ready
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    with _GST_INIT_LOCK:
        if not _gst_ready:
            Gst.init(None)
            _gst_ready = True
    return Gst


class DeepStreamDetector(Detector):
    """
    Detection, and optionally tracking, through NVIDIA DeepStream.

    The graph is

        appsrc -> videoconvert -> nvvideoconvert -> nvstreammux
               -> nvinfer -> [nvtracker] -> appsink

    nvtracker is present only when models.yaml sets use_tracker, which is what
    makes this the one backend that can assign track_id itself. It is not a
    service or a second model: it is an element in this pipeline, tracking the
    boxes nvinfer produced in the same buffer pass, on the same GPU memory. It
    cannot accept detections from a different detector, which is why it lives
    here rather than behind the Tracker ABC alongside BoxMotTracker.

    Nothing here is bound to one model. nvinfer is driven by the config file
    named in models.yaml `ds_infer_config`, so PeopleNet, TrafficCamNet,
    DashCamNet, FaceDetect and any other DetectNet_v2 model are a config change
    with no code change. Models with a different output head (PeopleNet
    Transformer, PeopleSegNet) need a custom parser library named in that same
    config file.

    gi and pyds are imported inside load(), never at module import time, the
    rule every backend here follows for its own SDK. Building the detector
    registry, running config validation, or running any other backend therefore
    never requires DeepStream to be present. A Raspberry Pi or a Mac imports
    this module perfectly happily; it just must not select it.
    """

    # nvtracker is built only when models.yaml asks for tracking, so whether
    # this is in effect is decided by registry.tracks_internally(cfg).
    tracks_internally = True

    # Never shared, tracking or not. The pipeline negotiates caps once, for the
    # first frame's resolution, so a second camera at another size could not
    # use it — and the push/pull lock would serialise the two regardless.
    shareable = False

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        # SDK modules, captured in load() so no method has to re-import them.
        self._gst = None
        self._pyds = None

        self._pipeline = None
        self._appsrc = None
        self._appsink = None
        self._streammux = None
        self._bus = None

        self._caps_set = False
        # Set once the pipeline reports an error or EOS. A GStreamer pipeline
        # does not recover from either on its own, so every later frame would
        # otherwise fail identically, once per frame, forever.
        self._failed: str | None = None
        self._frame_size: tuple[int, int] = (0, 0)   # (width, height)
        self._ids = StableIdMap()
        # class_id -> name, read from the label file nvinfer is configured with
        self._idx_to_name: dict[int, str] = {}
        # Serialises push/pull. One camera's frames arrive one at a time today,
        # but a pipeline is a single shared resource and interleaving pushes
        # would hand back another frame's detections.
        self._lock = threading.Lock()

    @property
    def _tracking(self) -> bool:
        return bool(self._cfg.use_tracker)

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        self._gst = _import_gst()
        # Imported for its side effect of failing here, at startup, with a
        # clear message — rather than on the first frame in production.
        import pyds
        self._pyds = pyds

        self._check_prerequisites()
        self._idx_to_name = self._load_label_names(self._cfg.ds_infer_config)
        self._pipeline = self._build()
        self._bus = self._pipeline.get_bus()

        # Deliberately NOT started here. appsrc's caps and nvstreammux's width
        # and height come from the first frame, and GStreamer cannot negotiate
        # a pipeline whose source format is still unknown — asking for PLAYING
        # now returns a plain state-change failure with no useful message.
        # _configure_caps starts it once the frame size is known.

        log.info(
            "detector '%s' built — DeepStream nvinfer(%s), tracking %s, %d classes; "
            "starting on first frame",
            self._cfg.id, self._cfg.ds_infer_config,
            f"via nvtracker({self._cfg.tracker})" if self._tracking else "off",
            len(self._idx_to_name),
        )

    def _check_prerequisites(self) -> None:
        """Fail with a specific message before anything is built."""
        if not self._cfg.ds_infer_config:
            raise RuntimeError(
                f"detector '{self._cfg.id}': device is 'deepstream' but "
                f"ds_infer_config is not set — nvinfer has no configuration to "
                f"load. Point it at an nvinfer config file, e.g. "
                f"config/deepstream_infer.txt"
            )
        # The tracker library is only needed when tracking is on; a
        # detection-only pipeline never loads it.
        if self._tracking and not Path(_LL_LIB).is_file():
            raise FileNotFoundError(
                f"detector '{self._cfg.id}': DeepStream's tracker library is "
                f"missing at {_LL_LIB}. This backend needs the DeepStream SDK "
                f"installed; use device: cuda with a .engine instead on a "
                f"machine without it, or set use_tracker: false to run "
                f"detection only."
            )

    def _start(self) -> None:
        """
        Bring the pipeline up. Called once, from _configure_caps, because it
        cannot succeed before the frame format is known.
        """
        Gst = self._gst
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            # The bus usually carries a specific reason; a bare state-change
            # failure says only that something refused, not what.
            self._check_bus()
            raise RuntimeError(
                f"detector '{self._cfg.id}': pipeline refused to start. Run with "
                f"GST_DEBUG=3 to see which element failed."
            )

        # PLAYING is asynchronous for a live source: the call above returns
        # ASYNC and the elements are still negotiating. Waiting here means the
        # first push meets a pipeline that is actually ready, instead of one
        # that silently drops the frame.
        state_change, _, _ = self._pipeline.get_state(_START_TIMEOUT_NS)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._check_bus()
            raise RuntimeError(
                f"detector '{self._cfg.id}': pipeline failed while reaching "
                f"PLAYING. Run with GST_DEBUG=3 for the failing element."
            )

    # ── pipeline construction ─────────────────────────────────────────────────

    def _build(self):
        """Create, configure and link the pipeline. Returns it, not yet playing."""
        pipeline = self._gst.Pipeline.new(f"ve-{self._cfg.id}")
        elements = self._create_elements(pipeline)
        self._configure_elements(elements)
        self._link_elements(elements)

        self._appsrc = elements["appsrc"]
        self._appsink = elements["appsink"]
        self._streammux = elements["streammux"]
        return pipeline

    def _create_elements(self, pipeline) -> dict:
        """
        One entry per element, already added to the pipeline.

        nvtracker is absent when tracking is off — NvDCF is not free, and a
        deployment that only counts detections should not pay for tracking it
        never reads.
        """
        def make(factory: str):
            element = self._gst.ElementFactory.make(factory, f"{factory}-{self._cfg.id}")
            if element is None:
                raise RuntimeError(
                    f"detector '{self._cfg.id}': GStreamer element '{factory}' "
                    f"could not be created. For nv* elements this means the "
                    f"DeepStream plugins are not on GST_PLUGIN_PATH."
                )
            pipeline.add(element)
            return element

        names = ["appsrc", "videoconvert", "nvvideoconvert", "capsfilter",
                 "nvstreammux", "nvinfer", "appsink"]
        if self._tracking:
            names.append("nvtracker")

        elements = {n: make(n) for n in names}
        # Aliases for the two whose factory name says little about their role.
        elements["streammux"] = elements["nvstreammux"]
        elements["pgie"] = elements["nvinfer"]
        return elements

    def _configure_elements(self, el: dict) -> None:
        Gst = self._gst

        el["appsrc"].set_property("format", Gst.Format.TIME)
        el["appsrc"].set_property("is-live", True)
        el["appsrc"].set_property("do-timestamp", True)
        # 0 means unlimited, which is deliberate: flow control here is the
        # synchronous push-then-pull in infer(), not appsrc's own queue. Only
        # one buffer is ever in flight, so the queue cannot grow, and appsrc's
        # default 200KB limit would otherwise block on any frame larger than
        # that — which is every frame above roughly 260x260.
        el["appsrc"].set_property("max-bytes", 0)
        el["appsrc"].set_property("block", True)

        # nvinfer requires NVMM RGBA; this is where the frame reaches the GPU.
        el["capsfilter"].set_property(
            "caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        )

        # Force the conversion onto the GPU. On Jetson nvvideoconvert defaults
        # to the VIC, a fixed-function block that cannot convert RGB or BGR:
        # frames flow for a while and then it fails with "RGB/BGR Format
        # transformation is not supported by VIC use GPU instead", taking the
        # whole pipeline down with an EOS. Frames arrive here as BGR from
        # OpenCV, so the VIC is never the right unit for this stage.
        # 1 = GPU. Guarded because the property does not exist on dGPU builds,
        # where the VIC does not exist either and the default is already GPU.
        if _has_property(el["nvvideoconvert"], "compute-hw"):
            el["nvvideoconvert"].set_property("compute-hw", 1)
        else:
            log.debug("detector '%s': nvvideoconvert has no compute-hw property",
                      self._cfg.id)

        # batch-size stays 1: one pipeline per camera, because this backend is
        # not shareable and ModelRegistry gives each camera its own instance.
        el["streammux"].set_property("batch-size", 1)
        el["streammux"].set_property("live-source", 1)
        el["streammux"].set_property("batched-push-timeout", 40000)

        # Order matters. config-file-path resets nvinfer's properties as it is
        # parsed, so the engine override has to be applied afterwards to stick.
        el["pgie"].set_property("config-file-path", self._cfg.ds_infer_config)
        if self._cfg.path:
            # models.yaml `path` stays the single source of truth for which
            # engine runs, overriding whatever model-engine-file the config
            # names, so the two files can never drift apart.
            el["pgie"].set_property("model-engine-file", self._cfg.path)

        if self._tracking:
            tracker = el["nvtracker"]
            tracker.set_property("ll-lib-file", _LL_LIB)
            # The one line that chooses the algorithm — IOU, NvSORT, NvDCF and
            # NvDeepSORT all live in the library above and are selected purely
            # by which config file is named here.
            tracker.set_property("ll-config-file", self._cfg.tracker)
            tracker.set_property("tracker-width", _TRACKER_W)
            tracker.set_property("tracker-height", _TRACKER_H)

        el["appsink"].set_property("emit-signals", False)
        el["appsink"].set_property("sync", False)
        el["appsink"].set_property("max-buffers", 1)
        el["appsink"].set_property("drop", False)

    def _link_elements(self, el: dict) -> None:
        el["appsrc"].link(el["videoconvert"])
        el["videoconvert"].link(el["nvvideoconvert"])
        el["nvvideoconvert"].link(el["capsfilter"])

        self._link_into_streammux(el["capsfilter"], el["streammux"])

        # nvtracker sits between detection and output when present, receiving
        # the same buffer nvinfer just annotated and adding object_id to it.
        tail = el["streammux"]
        for name in ("pgie", "nvtracker", "appsink"):
            if name in el:
                tail.link(el[name])
                tail = el[name]

    def _link_into_streammux(self, source, streammux) -> None:
        """
        nvstreammux has request pads rather than a static sink pad, so the pad
        has to be asked for.

        get_request_pad was deprecated in GStreamer 1.20 in favour of
        request_pad_simple; which one exists depends on the version DeepStream
        was built against. Resolved by attribute rather than by calling and
        falling back, because a removed method raises AttributeError instead of
        returning None — an `or` would not catch it.
        """
        request_pad = (getattr(streammux, "request_pad_simple", None)
                       or getattr(streammux, "get_request_pad", None))
        if request_pad is None:
            raise RuntimeError(
                f"detector '{self._cfg.id}': nvstreammux exposes neither "
                f"request_pad_simple nor get_request_pad — unexpected "
                f"GStreamer version"
            )
        mux_pad = request_pad("sink_0")
        if mux_pad is None:
            raise RuntimeError(
                f"detector '{self._cfg.id}': could not obtain nvstreammux sink pad"
            )
        if source.get_static_pad("src").link(mux_pad) != self._gst.PadLinkReturn.OK:
            raise RuntimeError(
                f"detector '{self._cfg.id}': failed to link into nvstreammux"
            )

    @staticmethod
    def _read_infer_keys(infer_config: str, *keys: str) -> dict[str, str]:
        """Pull a few top-level keys out of nvinfer's INI-style config."""
        wanted = set(keys)
        found: dict[str, str] = {}
        for line in Path(infer_config).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key in wanted and key not in found:
                found[key] = value.strip()
        return found

    def _load_label_names(self, infer_config: str) -> dict[int, str]:
        """
        Read the class_id -> name map from the label file nvinfer is pointed at.

        The label file is required, and models.yaml `classes` deliberately does
        not stand in for it. `classes` is a filter — it names what this
        deployment wants, in whatever order suits, and may be any subset. A
        model with ten classes narrowed to two would map ids 0 and 1 to those
        two names, so a detection of class 5 would either go unnamed or be
        labelled as the wrong thing. The label file is the model's own ordered
        list, which is what class_id actually indexes into.

        """
        cfg_path = Path(infer_config)
        label_value = self._read_infer_keys(infer_config, "labelfile-path").get("labelfile-path")

        if not label_value:
            raise RuntimeError(
                f"detector '{self._cfg.id}': {infer_config} has no labelfile-path. "
                f"It is required: class_id indexes into the model's own ordered "
                f"class list, and models.yaml classes cannot supply that because "
                f"it is a filter naming only the classes you want. Add a label "
                f"file with one name per line in model order."
            )

        candidate = Path(label_value)
        label_path = candidate if candidate.is_absolute() else cfg_path.parent / candidate
        if not label_path.is_file():
            raise FileNotFoundError(
                f"detector '{self._cfg.id}': {infer_config} names labelfile-path "
                f"'{label_value}' but {label_path} does not exist."
            )

        names = [n.strip() for n in
                 label_path.read_text(encoding="utf-8").splitlines() if n.strip()]
        if not names:
            raise RuntimeError(
                f"detector '{self._cfg.id}': label file {label_path} is empty"
            )

        # classes: may name anything the model produces, in any order and any
        # subset — but a name that appears nowhere in the label file can never
        # match, so the camera filter would silently drop everything.
        unknown = [c for c in self._cfg.classes if c not in set(names)]
        if unknown:
            raise RuntimeError(
                f"detector '{self._cfg.id}': models.yaml classes {unknown} are "
                f"not in the label file {label_path} (which has: "
                f"{', '.join(names)}). Nothing would ever match them."
            )

        log.info("detector '%s': %d class names from %s",
                 self._cfg.id, len(names), label_path)
        return dict(enumerate(names))

    # ── inference ─────────────────────────────────────────────────────────────

    def infer(self, frame, active_classes: list[str]) -> list[InferenceResult]:
        if self._pipeline is None:
            raise RuntimeError(f"detector '{self._cfg.id}': infer() called before load()")

        if self._failed:
            # Same message every frame, but an honest one: the pipeline is gone
            # and will not come back without a restart.
            raise RuntimeError(
                f"detector '{self._cfg.id}': pipeline is stopped and cannot "
                f"recover — {self._failed}. Restart the service."
            )

        active = set(active_classes) if active_classes else None
        height, width = frame.shape[:2]

        with self._lock:
            self._configure_caps(width, height)
            self._check_bus()
            self._push(frame)
            sample = self._pull()
            return self._extract(sample, active, width, height)

    def _push(self, frame) -> None:
        Gst = self._gst
        # ascontiguousarray matters: a cropped or sliced frame is not
        # contiguous, and tobytes() on that would produce a wrong pixel layout.
        buf = Gst.Buffer.new_wrapped(np.ascontiguousarray(frame).tobytes())
        if self._appsrc.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            raise RuntimeError(f"detector '{self._cfg.id}': appsrc rejected a frame")

    def _pull(self):
        sample = self._appsink.emit("try-pull-sample", _PULL_TIMEOUT_NS)
        if sample is None:
            # Check the bus first. A dead pipeline returns here immediately
            # rather than after the timeout, so reporting a timeout would be
            # both wrong and useless — the bus has the real reason.
            self._check_bus()
            self._failed = "the pipeline produced no output and reported no error"
            raise RuntimeError(
                f"detector '{self._cfg.id}': no output from the pipeline "
                f"(waited up to {_PULL_TIMEOUT_NS / 1e9:.0f}s) and the bus is "
                f"silent — appsink returned nothing"
            )
        return sample

    def _configure_caps(self, width: int, height: int) -> None:
        """
        Declare the frame format once, from the first frame that arrives.

        The size is taken from the frame rather than models.yaml input_size:
        input_size is the model's input, which nvinfer scales to internally,
        whereas the pipeline carries frames at capture resolution so that the
        boxes coming back are already in the coordinate space zones and rules
        expect.
        """
        if self._caps_set:
            if (width, height) != self._frame_size:
                raise RuntimeError(
                    f"detector '{self._cfg.id}': frame size changed from "
                    f"{self._frame_size} to {(width, height)} — the pipeline is "
                    f"built for a fixed size"
                )
            return

        self._appsrc.set_property("caps", self._gst.Caps.from_string(
            f"video/x-raw, format=BGR, width={width}, height={height}, framerate=30/1"
        ))
        self._streammux.set_property("width", width)
        self._streammux.set_property("height", height)
        self._frame_size = (width, height)
        self._caps_set = True
        log.info("detector '%s': pipeline configured for %dx%d frames — starting",
                 self._cfg.id, width, height)

        # Only now can the pipeline negotiate: appsrc knows its format and the
        # muxer its output size.
        self._start()

    # ── reading DeepStream metadata ───────────────────────────────────────────

    def _extract(self, sample, active: set[str] | None,
                 width: int, height: int) -> list[InferenceResult]:
        """
        Walk the metadata attached to the processed buffer.

        NvDs metadata is a C linked list, not a Python container, hence the
        explicit .next traversal. The batch holds one frame here, but writing
        it as a real traversal means batching more cameras later needs no
        rewrite.
        """
        pyds = self._pyds
        # hash() on a PyGObject yields the underlying C pointer — the standard
        # pyds idiom for handing a buffer to the metadata API.
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(sample.get_buffer()))
        if batch_meta is None:
            return []

        out: list[InferenceResult] = []
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                result = self._to_result(
                    pyds.NvDsObjectMeta.cast(l_obj.data), active, width, height)
                if result is not None:
                    out.append(result)
                l_obj = l_obj.next
            l_frame = l_frame.next
        return out

    def _to_result(self, obj, active: set[str] | None,
                   width: int, height: int) -> InferenceResult | None:
        """One object to one row, or None if it should be dropped."""
        name = self._class_name(obj)
        if not name:
            return None
        if active is not None and name not in active:
            return None

        confidence = float(obj.confidence)
        if confidence < self._cfg.confidence_threshold:
            return None

        bbox = self._bbox(obj, width, height)
        if bbox is None:
            return None

        raw_id = int(obj.object_id)
        return InferenceResult(
            class_name=name,
            confidence=confidence,
            bbox=bbox,
            # _UNTRACKED covers both tracking being off and an object NvTracker
            # has not yet opened a track for.
            track_id=None if raw_id == _UNTRACKED else self._ids.get(raw_id),
        )

    def _class_name(self, obj) -> str:
        name = self._idx_to_name.get(int(obj.class_id))
        if name is None:
            # No label file, or an id beyond it — nvinfer still attaches the
            # label it used, so fall back to that before discarding the object.
            name = str(obj.obj_label or "").strip()
        return name

    @staticmethod
    def _bbox(obj, width: int, height: int) -> list[float] | None:
        """
        rect_params is already in the pipeline's coordinate space, which is
        capture resolution — no rescaling needed.

        Clamped because NvTracker predicts through occlusion and can carry a
        box past the frame edge; an unclamped box would corrupt zone tests
        downstream. None means it clamped to nothing.
        """
        rect = obj.rect_params
        x1 = max(0.0, float(rect.left))
        y1 = max(0.0, float(rect.top))
        x2 = min(float(width), x1 + float(rect.width))
        y2 = min(float(height), y1 + float(rect.height))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _check_bus(self) -> None:
        """
        Surface a pipeline error as an exception rather than a silent stall.

        GStreamer reports failures on a message bus instead of raising, so
        without this a pipeline that died internally would simply stop
        producing and show up only as the pull timeout.
        """
        Gst = self._gst
        while True:
            message = self._bus.pop_filtered(
                Gst.MessageType.ERROR | Gst.MessageType.WARNING | Gst.MessageType.EOS)
            if message is None:
                return

            if message.type == Gst.MessageType.EOS:
                # End of stream on a live appsrc means an element downstream
                # tore the pipeline down. Nothing will flow again.
                self._failed = "the pipeline reached end-of-stream"
                raise RuntimeError(
                    f"detector '{self._cfg.id}': {self._failed}"
                )

            if message.type == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                self._failed = f"{message.src.get_name()}: {err}"
                raise RuntimeError(
                    f"detector '{self._cfg.id}': pipeline error from "
                    f"{message.src.get_name()}: {err} ({debug})"
                )

            err, _ = message.parse_warning()
            log.warning("detector '%s': %s: %s",
                        self._cfg.id, message.src.get_name(), err)

    def close(self) -> None:
        """Stop the pipeline. Safe to call more than once, and after a failed load."""
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            log.info("detector '%s': pipeline stopped", self._cfg.id)
