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
    branch_id: str   # which branch this device belongs to — used for alert routing
    key: str
    url: str         # trailing slash stripped at parse time
    ingest: IngestConfig
    request: RequestConfig


def parse(raw: dict) -> ApiConfig:
    ingest = raw.get("ingest", {})
    request = raw.get("request", {})
    return ApiConfig(
        branch_id=raw["branch_id"],
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
