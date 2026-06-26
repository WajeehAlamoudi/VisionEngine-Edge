from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BufferedRow:
    id: int
    table: str
    row: dict
    created_at: float
