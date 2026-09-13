from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from core.buffer import Buffer
from core.collector import Collector
from core.config import load_config
from core.health import HealthReporter
from core.ingest import IngestWorker
from core.model import ModelRegistry
from core.model.detector import build_camera_runtime
from core.notifier import Notifier
from core.pipeline import CameraPipeline
from core.rules import RulesEngine

log = logging.getLogger(__name__)

# How long to wait for the camera loops to finish before giving up on them.
# Generous enough for a read already in flight to return, short enough that a
# stuck camera does not stop the device from exiting.
_SHUTDOWN_TIMEOUT_SECONDS = 10


async def run(config_dir: str) -> None:
    cfg = load_config(config_dir)

    _setup_logging(cfg.device.log_level)

    log.info(
        "VisionEngine Edge — device=%s  environment=%s",
        cfg.device.id, cfg.device.environment,
    )

    if not cfg.enabled_cameras:
        log.error("no enabled cameras in cameras.yaml — nothing to run")
        return

    # ── initialise components ─────────────────────────────────────────────────

    buffer = Buffer(cfg.api.buffer)
    await buffer.start()

    registry = ModelRegistry()
    registry.load_for_cameras(cfg.models, cfg.enabled_cameras)

    rules = RulesEngine(cfg.enabled_rules)
    notifier = Notifier(cfg)
    await notifier.start()

    ingest = IngestWorker(cfg, buffer)
    await ingest.start()

    collector = Collector(cfg)
    await collector.start()

    # ── build pipelines ───────────────────────────────────────────────────────

    # Each camera's video path is chosen by its model's runtime: OpenCV for
    # ultralytics, DeepStream's own for deepstream. CameraPipeline is identical
    # either way - it receives detections and knows nothing about capture.
    pipelines = [
        CameraPipeline(
            cam=cam,
            camera_runtime=build_camera_runtime(
                cam, cfg.models[cam.model_id], registry.get(cam.id)),
            buffer=buffer,
            rules=rules,
            notifier=notifier,
            ingest=ingest,
            device_id=cfg.device.id,
            batch_size=cfg.api.ingest.batch_size,
            collector=collector,
        )
        for cam in cfg.enabled_cameras
    ]

    health = HealthReporter(cfg, pipelines, buffer)
    await health.start()

    log.info("starting %d camera pipeline(s)", len(pipelines))

    # ── run ───────────────────────────────────────────────────────────────────

    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        log.info("shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_signal)
        loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    except NotImplementedError:
        # Windows — asyncio signal handlers not supported; Ctrl+C raises KeyboardInterrupt
        pass

    tasks = [asyncio.create_task(p.run(), name=f"pipeline-{p._cam.id}") for p in pipelines]

    try:
        await _supervise(tasks, stop_event)
    except KeyboardInterrupt:
        log.info("keyboard interrupt — shutting down")

    # ── graceful shutdown ─────────────────────────────────────────────────────

    # Before the cameras, not after. The reporter reads their state, and during
    # the wait below that state is half torn down — it would send heartbeats
    # describing a device that is stopping as though it were running.
    log.info("stopping heartbeat...")
    await health.stop()

    log.info("stopping camera pipelines...")
    for p in pipelines:
        p.stop()

    # Bounded, so one camera that will not come back cannot hold the process
    # open. Everything below still runs, and the buffer is flushed, which
    # matters more than a tidy exit for the pipeline that hung.
    done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_TIMEOUT_SECONDS)
    if pending:
        for task in pending:
            log.warning("%s did not stop within %ds — abandoning it",
                        task.get_name(), _SHUTDOWN_TIMEOUT_SECONDS)
            task.cancel()
        # Awaited, not just requested: cancel() only schedules the CancelledError,
        # and a task torn down while the services below are closing would raise
        # from inside a buffer or client that has already gone.
        await asyncio.gather(*pending, return_exceptions=True)

    # Ahead of the model teardown, not after it. The cameras are down, so every
    # detection that will ever exist is already in the buffer, and this is the
    # last chance to deliver it before the process goes. Releasing GStreamer
    # first would spend that chance waiting on something no row depends on.
    log.info("flushing buffered rows...")
    await ingest.stop()

    # Nothing is mid-inference by now. A backend holding something outside the
    # interpreter (a GStreamer pipeline, a device handle) needs releasing
    # explicitly rather than being left to process exit.
    log.info("releasing models...")
    registry.close()

    log.info("stopping background services...")
    await notifier.stop()
    await buffer.stop()

    log.info("shutdown complete")


async def _supervise(tasks: list[asyncio.Task], stop_event: asyncio.Event) -> None:
    """
    Wait for shutdown, but notice a camera that stops before it arrives.

    Waiting on stop_event alone is how a device ends up running blind: a task
    that ends with an exception simply disappears, the process keeps going, and
    the heartbeat keeps reporting a healthy node with no cameras on it. Every
    ending is logged here, and once the last camera has gone there is nothing
    left to stay up for, so the agent exits and lets the supervisor restart it.

    Nothing is restarted in place. A camera loop already retries the failures it
    can recover from; reaching here means it stopped for a reason that survives
    a retry, and a clean process start clears far more state than re-running the
    same coroutine would.
    """
    stop_waiter = asyncio.create_task(stop_event.wait(), name="stop-signal")
    running = set(tasks)
    try:
        while running:
            done, pending = await asyncio.wait(
                {stop_waiter, *running}, return_when=asyncio.FIRST_COMPLETED)
            if stop_waiter in done:
                return

            running = pending - {stop_waiter}
            for task in done:
                # exception() also marks it retrieved, which is what keeps
                # asyncio from logging it again as never-consumed at exit.
                exc = None if task.cancelled() else task.exception()
                if exc is None:
                    log.warning("%s ended on its own", task.get_name())
                else:
                    log.error("%s ended with an error: %r", task.get_name(), exc,
                              exc_info=exc)
            log.warning("%d camera pipeline(s) still running", len(running))

        log.error("every camera pipeline has stopped — shutting down")
    finally:
        stop_waiter.cancel()


# ── logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    # silence noisy third-party loggers
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VisionEngine Edge Agent")
    parser.add_argument(
        "--config",
        default="config",
        metavar="DIR",
        help="path to config directory (default: config)",
    )
    args = parser.parse_args()

    code = 0
    try:
        asyncio.run(run(args.config))
    except KeyboardInterrupt:
        # Only reachable where asyncio signal handlers are unavailable, and the
        # shutdown inside run() has already happened by the time it lands here.
        pass
    except Exception:
        log.exception("agent exited with an error")
        code = 1

    _exit_now(code)


def _exit_now(code: int) -> None:
    """
    Leave without waiting for threads that are never coming back.

    A camera abandoned at the shutdown timeout is still inside a blocking read
    on a ThreadPoolExecutor worker, and those workers are not daemons:
    concurrent.futures joins every one of them at interpreter exit. So the
    process prints "shutdown complete", returns from asyncio.run, and then hangs
    forever on a join nothing will release — still holding buffer.db and, until
    the reporter was stopped, still telling the backend it was healthy. A second
    agent started against that file then loses every camera to "database is
    locked".

    Skipping the join is safe here and nowhere else: this runs only after the
    buffer has been flushed and closed, so all durable state is already on disk.
    The log streams are flushed by hand because os._exit runs no atexit hooks.
    """
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
