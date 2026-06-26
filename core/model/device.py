from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch
        if torch.cuda.is_available():
            log.info("device auto-select: CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            log.info("device auto-select: MPS (Apple Metal)")
            return "mps"
    except ImportError:
        pass

    log.info("device auto-select: CPU")
    return "cpu"
