from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def resolve_accelerator(accelerator: str) -> str:
    """
    Turn models.yaml `accelerator` into the device string torch understands.

    Only "auto" needs resolving; everything else is passed through as written.
    auto is answered by asking torch what is actually present, which is the one
    place in the config where a value is not taken literally — and why a
    TensorRT .engine is not allowed to use it: auto falls back to cpu on a
    machine without CUDA, and an engine cannot load there.

    Named for the config key rather than "device", which in this project means
    the physical edge device (device.yaml) and would read as the wrong thing.
    """
    if accelerator != "auto":
        return accelerator

    try:
        import torch
        if torch.cuda.is_available():
            log.info("accelerator auto-select: CUDA")
            return "cuda"
        if torch.backends.mps.is_available():
            log.info("accelerator auto-select: MPS (Apple Metal)")
            return "mps"
    except ImportError:
        pass

    log.info("accelerator auto-select: CPU")
    return "cpu"
