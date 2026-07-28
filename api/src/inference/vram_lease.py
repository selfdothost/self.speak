"""self.speak's side of the cross-service VRAM-lease protocol.

Home of the ``/api/system/*`` capability (cavekit-vram-lease-client): the live
tri-state ``torch.cuda`` VRAM probe (R1), the in-flight-synthesis counter +
bounded drain-wait (R2-AC2), and the single-flight, live-verified release
orchestration (R2). self.speak is a single uvicorn process with module-singleton
state, so all synchronisation here is in-process (asyncio / plain counters on the
single event-loop thread) — there is no supervisord+router split to coordinate.

Import direction is one-directional: ``tts_service`` imports ``track_synthesis``
from here; this module imports only ``torch``, the wire schemas, and (lazily,
inside functions) ``model_manager`` — so there is no import cycle.
"""

import asyncio
import contextlib
import logging
import time
from typing import AsyncIterator

import torch

from ..structures.schemas import VramReleaseResponse

log = logging.getLogger(__name__)


# ─── T-002: live tri-state torch.cuda VRAM probe ─────────────────────────


def _model_resident() -> bool:
    """Best-effort observability flag: is the Kokoro backend loaded right now?

    Reads ``ModelManager._instance`` directly and NEVER constructs a manager
    (a probe must not have side effects). Any failure resolves to ``False`` —
    residency is observability, never part of the byte accounting or a reason to
    500 the state endpoint.
    """
    try:
        from .model_manager import ModelManager

        inst = ModelManager._instance
        if inst is None:
            return False
        backend = getattr(inst, "_backend", None)
        if backend is None:
            return False
        return bool(getattr(backend, "is_loaded", False))
    except Exception:
        return False


def probe_vram_state() -> dict:
    """Compute the tri-state VRAM answer live, on every call (never cached).

    Returns a dict with ``held_vram_bytes``, ``total_capacity_bytes`` (both in
    BYTES), ``gpu_reachable``, ``status``, ``model_resident`` — honouring the
    R1-AC4 tri-state exactly:

    * (a) GPU reachable & probed → both figures are real ints (held near-zero,
      not null, when idle); ``status="ok"``, ``gpu_reachable=True``.
    * (b) GPU present but a ``torch.cuda`` call RAISES (driver shadowing /
      CUDA-init failure) → both figures ``None`` (JSON null, NEVER 0);
      ``status="unreachable"``, ``gpu_reachable=False``.
    * (c) no CUDA / CPU deployment → both figures integer ``0``;
      ``status="no_gpu"``, ``gpu_reachable=False``.

    The ``try/except`` around the CUDA calls is load-bearing, not defensive
    boilerplate: ``mem_get_info()`` can throw on first call under a shadowed
    driver (the self.transcribe ctranslate2-probe failure mode), and that MUST
    resolve to case (b) — a null/unreachable answer — never a crash and never a
    false zero (R1-AC5).
    """
    model_resident = _model_resident()

    # is_available() is itself wrapped: on a genuinely broken CUDA it can be the
    # first call that throws — treat a throw here as case (b), not a crash.
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as e:  # pragma: no cover - defensive, exercised via mocks
        log.warning(
            "vram-lease: torch.cuda.is_available() raised (%r); reporting "
            "unreachable (case b)",
            e,
        )
        return {
            "held_vram_bytes": None,
            "total_capacity_bytes": None,
            "gpu_reachable": False,
            "status": "unreachable",
            "model_resident": model_resident,
        }

    if not cuda_available:
        # Case (c): no CUDA / CPU deployment — a known-zero, no-op consumer.
        return {
            "held_vram_bytes": 0,
            "total_capacity_bytes": 0,
            "gpu_reachable": False,
            "status": "no_gpu",
            "model_resident": model_resident,
        }

    # CUDA reports available → attempt the real probe. A throw is case (b).
    try:
        _free_bytes, total_bytes = torch.cuda.mem_get_info()
        held_bytes = torch.cuda.memory_allocated()
    except Exception as e:
        log.warning(
            "vram-lease: torch.cuda VRAM probe raised (%r); reporting "
            "unreachable (case b) — held/capacity null, never a false 0",
            e,
        )
        return {
            "held_vram_bytes": None,
            "total_capacity_bytes": None,
            "gpu_reachable": False,
            "status": "unreachable",
            "model_resident": model_resident,
        }

    # Case (a): reachable & probed. Real ints, held near-zero when idle.
    return {
        "held_vram_bytes": int(held_bytes),
        "total_capacity_bytes": int(total_bytes),
        "gpu_reachable": True,
        "status": "ok",
        "model_resident": model_resident,
    }


# ─── T-005: in-flight-synthesis counter + bounded wait-for-drain ─────────
#
# This repo had NO request-level in-flight signal (only a chunk-level semaphore
# and an init lock), so the counter is a genuinely new hook in the synthesis hot
# path. It is a plain module-level int guarded by nothing but the single asyncio
# event-loop thread: increments/decrements happen with no ``await`` between the
# read and the write, so they are atomic on the loop. ``wait_for_drain`` polls
# it with a bounded ``asyncio.sleep`` loop (chosen over a module-level
# ``asyncio.Event`` to avoid an Event object binding to one test's event loop and
# raising "bound to a different loop" in the next).

_synthesis_count = 0
_DRAIN_POLL_INTERVAL = 0.02  # seconds


def synthesis_in_flight() -> int:
    """Number of syntheses currently generating (0 == idle)."""
    return _synthesis_count


@contextlib.asynccontextmanager
async def track_synthesis() -> AsyncIterator[None]:
    """Count one in-flight synthesis for the whole ``async with`` body.

    Increments on enter and decrements in ``finally`` (so a raise inside the
    wrapped body — or a GeneratorExit when the stream is closed mid-flight —
    still decrements; the counter never leaks). Used to wrap the synthesis hot
    path in ``TTSService.generate_audio_stream`` (T-005).
    """
    global _synthesis_count
    _synthesis_count += 1
    try:
        yield
    finally:
        _synthesis_count = max(0, _synthesis_count - 1)


async def wait_for_drain(timeout_seconds: float) -> bool:
    """Wait — bounded by ``timeout_seconds`` — for the in-flight counter to hit 0.

    Returns ``True`` if idle now or drained within the deadline (the R2-AC2
    idle-fast-path returns immediately), ``False`` if the deadline passed with a
    synthesis still active. Never blocks the event loop and always respects the
    wall-clock deadline (feeds R2-AC6). A non-positive timeout collapses to a
    single immediate check.
    """
    if _synthesis_count <= 0:
        return True

    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return _synthesis_count <= 0
        await asyncio.sleep(min(_DRAIN_POLL_INTERVAL, remaining))
        if _synthesis_count <= 0:
            return True


# ─── T-006: release orchestration ────────────────────────────────────────
#
# Single-flight is a plain module-level boolean, not an ``asyncio.Lock``: on the
# single event-loop thread the check-and-set below happens with no ``await``
# between, so two coroutines cannot both pass the guard — functionally identical
# to a non-blocking lock acquire, without an asyncio.Lock binding to a stale test
# loop. A single uvicorn process → one process-local guard is sufficient.

_release_in_flight = False


def _unload_backend() -> None:
    """Unload the resident Kokoro backend if (and only if) one exists.

    Reads ``ModelManager._instance`` directly and NEVER constructs a manager to
    unload — a pre-warmup release with no instance frees nothing, truthfully.
    ``ModelManager.unload_all()`` → ``KokoroV1.unload()`` drops the model, clears
    pipelines and (only when CUDA is available) calls
    ``torch.cuda.empty_cache()`` + ``synchronize()``. A raised unload is logged
    and swallowed so the handler still proceeds to the live re-verify — the
    freed figure comes from the measured delta, never from this call returning.
    """
    try:
        from .model_manager import ModelManager

        inst = ModelManager._instance
        if inst is None:
            return
        if getattr(inst, "_backend", None) is None:
            return
        inst.unload_all()
    except Exception as e:
        log.warning(
            "vram-lease: backend unload raised (%r); continuing to live-verify "
            "the freed delta",
            e,
        )


async def handle_vram_release(
    target_bytes: int, timeout_seconds: float
) -> VramReleaseResponse:
    """Answer a release-request: single-flight, drain-wait, live-verified.

    Sequence (cavekit-vram-lease-client R2):

    1. **Single-flight (AC7):** if a release is already in flight, return
       ``busy`` immediately — never run two passes concurrently.
    2. **CPU no-op (AC8):** on a no-CUDA deployment return ``released``/0 — an
       honest empty-but-valid no-op (the zero-byte target is trivially met).
       An ``unreachable`` GPU (case b) → truthful ``partial``/0 (can't measure,
       so can't confirm a free).
    3. **Baseline (AC3):** record the live held figure before freeing.
    4. **Drain-wait (AC2/AC6):** wait — bounded by ``timeout_seconds`` — for
       in-flight synthesis to drain. Idle → returns immediately. Still
       generating at the deadline → ``partial``/0 with NO unload (never yank the
       model from under an active request).
    5. **Free:** unload the Kokoro backend (only if resident).
    6. **Verify live (AC3/AC5):** re-probe held; ``freed = max(0, before -
       after)`` measured from the live delta, never inferred. ``released`` iff
       ``freed >= target_bytes`` else a truthful ``partial``.
    """
    global _release_in_flight

    # 1. Single-flight (AC7). Check-and-set with no await between → race-free.
    if _release_in_flight:
        return VramReleaseResponse(status="busy", freed_bytes=0)
    _release_in_flight = True
    try:
        state = probe_vram_state()
        status = state["status"]

        # 2. AC8: CPU / no-CUDA → honest zero-freed no-op (target-of-nothing met).
        if status == "no_gpu":
            return VramReleaseResponse(status="released", freed_bytes=0)

        # Case (b): GPU present but unprobable — can't measure → can't confirm.
        if status == "unreachable":
            return VramReleaseResponse(status="partial", freed_bytes=0)

        # 3. AC3: live baseline. If unreadable, can't confirm a delta.
        before = state["held_vram_bytes"]
        if before is None:
            return VramReleaseResponse(status="partial", freed_bytes=0)

        # 4. AC2/AC6: drain in-flight synthesis, bounded by the timeout.
        drained = await wait_for_drain(timeout_seconds)
        if not drained:
            # Still generating at the deadline — never unload out from under it.
            return VramReleaseResponse(status="partial", freed_bytes=0)

        # 5. Free (only if a model is actually resident).
        _unload_backend()

        # 6. AC3/AC5: verify live via a fresh probe.
        after_state = probe_vram_state()
        after = after_state["held_vram_bytes"]
        if after is None:
            # Probe went unreachable across the free — can't confirm a delta.
            return VramReleaseResponse(status="partial", freed_bytes=0)

        freed = max(0, int(before) - int(after))
        if freed >= target_bytes:
            return VramReleaseResponse(status="released", freed_bytes=freed)
        # A real but smaller (or zero) free — truthful, never a fabricated success.
        return VramReleaseResponse(status="partial", freed_bytes=freed)
    finally:
        _release_in_flight = False
