from __future__ import annotations
from dataclasses import dataclass


@dataclass
class IngestConfig:
    batch_size: int
    flush_interval_seconds: int


@dataclass
class RequestConfig:
    timeout_seconds: int
    max_consecutive_failures: int


@dataclass
class ApiConfig:
    key: str
    url: str  # trailing slash stripped at parse time
    ingest: IngestConfig
    request: RequestConfig


def parse(raw: dict) -> ApiConfig:
    ingest = raw.get("ingest", {})
    request = raw.get("request", {})
    return ApiConfig(
        key=raw["key"],
        url=raw["url"].rstrip("/"),
        ingest=IngestConfig(
            batch_size=ingest.get("batch_size", 50),
            flush_interval_seconds=ingest.get("flush_interval_seconds", 30),
        ),
        request=RequestConfig(
            timeout_seconds=request.get("timeout_seconds", 10),
            max_consecutive_failures=request.get("max_consecutive_failures", 5),
        ),
    )
