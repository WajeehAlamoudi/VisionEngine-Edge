from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

from core.buffer import Buffer
from core.collector import Collector
from core.config import load_config
from core.health import HealthReporter
from core.ingest import IngestWorker
from core.model import ModelRegistry
from core.notifier import Notifier
from core.pipeline import CameraPipeline
from core.rules import RulesEngine

log = logging.getLogger(__name__)


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

    buffer = Buffer(cfg.device.buffer)
    await buffer.start()

    registry = ModelRegistry()
    registry.load_for_cameras(
        cfg.models,
        needed_ids=[cam.model_id for cam in cfg.enabled_cameras],
    )

    rules = RulesEngine(cfg.enabled_rules)
    notifier = Notifier(cfg)
    await notifier.start()

    ingest = IngestWorker(cfg, buffer)
    await ingest.start()

    collector = Collector(cfg)
    await collector.start()

    # ── build pipelines ───────────────────────────────────────────────────────

    pipelines = [
        CameraPipeline(
            cam=cam,
            runner=registry.get(cam.model_id),
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
        await stop_event.wait()
    except KeyboardInterrupt:
        log.info("keyboard interrupt — shutting down")

    # ── graceful shutdown ─────────────────────────────────────────────────────

    log.info("stopping camera pipelines...")
    for p in pipelines:
        p.stop()
    await asyncio.gather(*tasks, return_exceptions=True)

    log.info("stopping background services...")
    await health.stop()
    await ingest.stop()
    await notifier.stop()
    await buffer.stop()

    log.info("shutdown complete")


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
    asyncio.run(run(args.config))


if __name__ == "__main__":
    main()
