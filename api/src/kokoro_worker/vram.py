"""Worker-local VRAM probe + release actuator for the Kokoro worker.

Mirrors ``chatterbox_worker/vram.py``. ``torch.cuda.memory_reserved()`` is
context-LOCAL, so this reports exactly the VRAM THIS process would give back —
and, critically, it does NOT create a CUDA context to answer the question.
``mem_get_info()`` would: it is a runtime API, it costs ~470 MiB the first time
it is called, and putting it on a poll path is how self.sketch burned 386 MiB
sitting idle.
"""

import logging
from typing import Optional

log = logging.getLogger("kokoro_worker.vram")


def _reserved_bytes() -> Optional[int]:
    """This process's CUDA caching-allocator reserved bytes.

    * int (including 0) when the probe succeeds — a real, knowable zero.
    * ``None`` only when a ``torch.cuda`` call RAISES. A false 0 there would let
      the aggregator over-report free VRAM, which over-grants the card.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.memory_reserved())
    except Exception as e:
        log.warning("kokoro-vram: reserved-bytes probe raised (%r); reporting null", e)
        return None


def _peek_loaded() -> bool:
    """Is a Kokoro backend resident? Observability must not have side effects.

    Reads the manager singleton WITHOUT constructing one — constructing it would
    load a model in order to report whether a model is loaded.
    """
    try:
        from ..inference import model_manager

        inst = getattr(model_manager, "_manager_instance", None)
        if inst is None:
            return False
        backend = getattr(inst, "_backend", None)
        return backend is not None and bool(getattr(backend, "is_loaded", False))
    except Exception:
        return False


def probe() -> dict:
    """The GET /vram-state body: ``{held_bytes: int|null, resident: bool}``."""
    return {"held_bytes": _reserved_bytes(), "resident": _peek_loaded()}


def unload() -> None:
    """Free this process's Kokoro VRAM (idempotent).

    Never constructs a manager — nothing loaded frees nothing, truthfully.
    ``gc.collect()`` BEFORE ``empty_cache()`` because nn.Module graphs are
    cyclic: without the collect the modules are unreachable but not yet
    finalised, so the allocator still holds their blocks and empty_cache()
    frees nothing (measured in self.speak!23: 12 MiB -> 12 MiB -> 0 MiB).
    """
    import gc

    try:
        from ..inference import model_manager

        inst = getattr(model_manager, "_manager_instance", None)
        if inst is not None and hasattr(inst, "unload_all"):
            inst.unload_all()
    except Exception as e:
        log.warning("kokoro-vram: unload raised (%r); re-probe is the truth", e)

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as e:
        log.warning("kokoro-vram: empty_cache raised (%r)", e)
