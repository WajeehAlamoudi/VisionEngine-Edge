from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IngestConfig:
    batch_size: int
    flush_interval_seconds: int


@dataclass
class BufferConfig:
    path: str
    max_size_mb: int
    retry_interval_seconds: int
    delete_after_hours: int


@dataclass
class RequestConfig:
    timeout_seconds: int
    max_consecutive_failures: int


@dataclass
class ApiConfig:
    branch_id: str   # which branch this device belongs to — used for alert routing
    key: str
    url: str         # trailing slash stripped at parse time
    ingest: IngestConfig
    buffer: BufferConfig
    request: RequestConfig


def parse(raw: dict) -> ApiConfig:
    ingest = raw.get("ingest", {})
    buf = raw.get("buffer", {})
    request = raw.get("request", {})
    return ApiConfig(
        branch_id=raw["branch_id"],
        key=raw["key"],
        url=raw["url"].rstrip("/"),
        ingest=IngestConfig(
            batch_size=ingest.get("batch_size", 50),
            flush_interval_seconds=ingest.get("flush_interval_seconds", 30),
        ),
        buffer=BufferConfig(
            path=buf.get("path", "./data/buffer.db"),
            max_size_mb=buf.get("max_size_mb", 200),
            retry_interval_seconds=buf.get("retry_interval_seconds", 10),
            delete_after_hours=buf.get("delete_after_hours", 24),
        ),
        request=RequestConfig(
            timeout_seconds=request.get("timeout_seconds", 10),
            max_consecutive_failures=request.get("max_consecutive_failures", 5),
        ),
    )
