"""Worker-local VRAM probe + release actuator (WORKER-venv module).

The worker owns its own CUDA context (torch 2.6). ``torch.cuda.memory_reserved()``
is context-LOCAL, so this reports exactly the VRAM THIS process would give back if
asked to release — the same caching-allocator unit the main process reports for
Kokoro (self.ai#74 / self.speak#4). The MAIN process aggregates this number over
localhost into self.speak's single ``held_vram_bytes`` (Phase 2); here we only
report our own slice and free it on demand.
"""

import logging
from typing import Optional

from .engine import ChatterboxEngine

log = logging.getLogger(__name__)


def _reserved_bytes() -> Optional[int]:
    """This process's CUDA caching-allocator reserved bytes.

    * int (incl. 0) when the probe succeeds — 0 on a no-CUDA/CPU worker is a real,
      knowable zero (a CPU process holds no VRAM).
    * ``None`` only when a ``torch.cuda`` call RAISES (driver shadowing / CUDA-init
      failure) — a false 0 there would let the aggregator over-report free VRAM.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_reserved())
    except Exception as e:
        log.warning("chatterbox-vram: reserved-bytes probe raised (%r); reporting null", e)
        return None


def probe() -> dict:
    """The GET /vram-state body: ``{held_bytes: int|null, resident: bool}``.

    ``resident`` is read without constructing the engine singleton (observability
    must not have side effects)."""
    return {
        "held_bytes": _reserved_bytes(),
        "resident": ChatterboxEngine.peek_loaded(),
    }


def unload() -> None:
    """Free this process's Chatterbox VRAM (idempotent). Reads the singleton
    directly and NEVER constructs one — nothing loaded frees nothing, truthfully.
    ``ChatterboxEngine.unload()`` drops the model + ``empty_cache()``/
    ``synchronize()`` and swallows its own raises so the live re-probe is the
    source of truth for freed bytes."""
    inst = ChatterboxEngine._instance
    if inst is None:
        return
    inst.unload()
