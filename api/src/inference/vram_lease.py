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

from ..core.config import settings
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


def _card_occupancy_no_context() -> tuple[int, int]:
    """Whole-card ``(used, total)`` in bytes WITHOUT creating a CUDA context.

    ``torch.cuda.mem_get_info()`` is a runtime API: calling it creates this
    process's CUDA primary context, measured at 386 MiB on the deployed 4090, and
    only process exit ever gives it back. self.speak's main process spends nearly
    all of its life idle — since the Kokoro cutover the model lives in the worker
    (self.speak#5) — so the main process was paying 386 MiB purely to answer "how
    full is the card?" on every poll.

    That is not a rounding error to core's broker. Card memory no consumer claims
    as releasable ``held`` becomes ``unattributed_overhead``, which is subtracted
    from free AND cannot be reclaimed from anyone — it is a double loss. It is
    what denied a Qwen3-Coder-Next load by 85 MiB (self.ai#126).

    NVML is the only context-free source of *used*, and it reports the whole card,
    which is exactly the quantity wanted here. Verified by fd-signature in the
    deployed container: a fresh process sits at 11 ``/dev/nvidia*`` fds after a
    static device query and jumps to 25 the instant ``mem_get_info()`` is called.

    BOTH figures come from NVML on purpose, rather than sourcing the total from
    the cheaper ``get_device_properties().total_memory``. The two do not agree —
    measured on the deployed 4090, properties reports 25512443904 and NVML
    25757220864, a 234 MiB gap (usable global memory vs the full framebuffer).
    Mixing them would put a used and a total from different denominators into the
    same pair, and core derives unattributed overhead by subtraction, so the gap
    would silently become 234 MiB of phantom overhead — the exact class of error
    this function exists to remove.

    Falls back to ``mem_get_info()`` when NVML is unavailable, matching the
    self.sketch precedent: a correct number with a context beats a wrong number
    without one. The fallback is the last resort, never the default, and it is
    self-consistent for the same reason (one call, one denominator).

    Raises on a genuinely unreadable device so the caller resolves to the case-(b)
    unreachable state rather than inventing a figure.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return int(info.used), int(info.total)
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
    except Exception as e:
        log.warning(
            "vram-lease: NVML card read failed (%r); falling back to mem_get_info(), "
            "which COSTS this process a CUDA context (~386 MiB, freed only at exit)",
            e,
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(total_bytes - free_bytes), int(total_bytes)


def probe_vram_state() -> dict:
    """The LOCAL (main-process / Kokoro) VRAM leg — sync, live, never cached.

    This is only this process's own CUDA-context accounting. The authoritative
    wire answer core reads is :func:`probe_vram_state_full`, which aggregates this
    with the sibling Chatterbox worker's slice when Chatterbox is part of the
    deployment (Phase 2). This function is kept sync and unchanged so it stays the
    exact single-engine behaviour on a Kokoro-only deploy and remains the honest
    fallback when the aggregator's worker leg is dark.

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
    boilerplate: the device probe can throw on first call under a shadowed
    driver (the self.transcribe ctranslate2-probe failure mode), and that MUST
    resolve to case (b) — a null/unreachable answer — never a crash and never a
    false zero (R1-AC5).

    Every CUDA call on the happy path is deliberately context-free
    (:func:`_card_occupancy_no_context`, ``memory_reserved()``): an idle main
    process must report the card without occupying it (self.speak#8).
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
            # Card occupancy is UNKNOWN, not zero: we could not read the device
            # at all. A 0 here would tell core the card is empty (self.ai#74).
            "device_used_bytes": None,
            "device_total_bytes": None,
            "gpu_reachable": False,
            "status": "unreachable",
            "model_resident": model_resident,
        }

    if not cuda_available:
        # Case (c): no CUDA / CPU deployment — a known-zero, no-op consumer.
        return {
            "held_vram_bytes": 0,
            "total_capacity_bytes": 0,
            # held 0 is a real, knowable fact (a CPU process holds no VRAM).
            # Card occupancy is NOT: with no CUDA we cannot see the device, so
            # this stays null rather than claiming an empty card (self.ai#74).
            "device_used_bytes": None,
            "device_total_bytes": None,
            "gpu_reachable": False,
            "status": "no_gpu",
            "model_resident": model_resident,
        }

    # CUDA reports available → attempt the real probe. A throw is case (b).
    try:
        device_used_bytes, total_bytes = _card_occupancy_no_context()
        # HELD = memory_reserved(), NOT memory_allocated() (self.ai#74 /
        # self.speak#4). memory_allocated() counts only bytes held by live
        # tensors; memory_reserved() is what the caching allocator has taken from
        # the driver and will not return until empty_cache(). Since our own
        # release path (below) unloads and then calls empty_cache(), reserved is
        # exactly "what we would give back if asked to release" — the unit
        # self.ai's broker sums across consumers to decide what is grantable.
        # Reporting allocated made core believe the card was emptier than it was
        # (over-grant -> OOM), and made a confirmed release look like it freed
        # ~0 bytes while genuinely returning gigabytes, because before/after were
        # measured in a unit blind to what empty_cache() reclaims.
        held_bytes = torch.cuda.memory_reserved()
        # DEVICE occupancy (read above, context-free) is a DIFFERENT quantity: the
        # whole card, every process. Reported separately and NEVER folded into
        # held — core sums held across consumers, so a whole-card figure in that
        # field would double-count every sibling. It is how core accounts for CUDA
        # contexts and non-consumer processes that no consumer can attribute.
    except Exception as e:
        log.warning(
            "vram-lease: torch.cuda VRAM probe raised (%r); reporting "
            "unreachable (case b) — held/capacity null, never a false 0",
            e,
        )
        return {
            "held_vram_bytes": None,
            "total_capacity_bytes": None,
            # Card occupancy is UNKNOWN, not zero: we could not read the device
            # at all. A 0 here would tell core the card is empty (self.ai#74).
            "device_used_bytes": None,
            "device_total_bytes": None,
            "gpu_reachable": False,
            "status": "unreachable",
            "model_resident": model_resident,
        }

    # Case (a): reachable & probed. Real ints, held near-zero when idle.
    return {
        "held_vram_bytes": int(held_bytes),
        "total_capacity_bytes": int(total_bytes),
        "device_used_bytes": device_used_bytes,
        "device_total_bytes": int(total_bytes),
        "gpu_reachable": True,
        "status": "ok",
        "model_resident": model_resident,
    }


# The established case-(b) unreachable shape (held/total/device all null). Reused
# verbatim when an EXPECTED Chatterbox worker is unaccountable, because it is the
# exact contract core already handles as "this consumer is dark, don't count it"
# — introducing a novel held-null-but-total-known state would risk core's parser
# under-counting held against a known total (the self.ai#74 over-grant).
def _unreachable_state(model_resident: bool) -> dict:
    return {
        "held_vram_bytes": None,
        "total_capacity_bytes": None,
        "device_used_bytes": None,
        "device_total_bytes": None,
        "gpu_reachable": False,
        "status": "unreachable",
        "model_resident": model_resident,
    }


async def probe_vram_state_full() -> dict:
    """The AGGREGATED, authoritative wire answer core reads (async).

    Combines the local Kokoro leg (:func:`probe_vram_state`) with the sibling
    Chatterbox worker's own reserved-VRAM slice into self.speak's single
    ``held_vram_bytes`` — the whole point of the per-engine-process model's Phase 2
    (INTEGRATION-PLAN-v2.md §2.4). The two processes own two independent CUDA
    contexts, so neither ``memory_reserved()`` sees the other; the only correct
    total is the localhost sum.

    Aggregation is gated on ``settings.chatterbox_enabled``:

    * **disabled** (default; Kokoro-only deploys and the whole pre-deploy window)
      → return the local leg verbatim. The worker is never queried, so an absent
      or dark worker can never perturb the already-working single-engine lease.
    * **enabled but local status != "ok"** → the local leg already dominates:
      ``no_gpu`` means the whole pod is CPU (the worker holds no VRAM either) and
      ``unreachable`` is already the null answer. Return it unchanged; querying
      the worker cannot add signal.
    * **enabled and local "ok"** → query the worker:
      - worker reports a real ``held_bytes`` int → held becomes
        ``local_held + worker_held``; ``model_resident`` ORs in the worker's
        residency. ``total_capacity_bytes`` / ``device_*`` are whole-card figures
        from the local NVML read, which spans BOTH contexts by virtue of being a
        card-wide figure, so they are left exactly as the local leg reported —
        never summed again.
      - worker unreachable, or reports ``held_bytes: null`` (its own probe raised)
        → **collapse to the case-(b) unreachable shape**. We know Chatterbox is
        part of this footprint but cannot read its slice; reporting only Kokoro's
        held would under-report the total and let core over-grant (self.ai#74).
        Unknown-and-say-so beats a confident under-count.
    """
    local = probe_vram_state()

    # Which sibling workers are expected to hold part of this footprint. Under
    # the P3 cutover the KOKORO worker is one of them, and it is the one that
    # matters most: when it is enabled the local leg holds ~nothing, because main
    # deliberately never allocates. Reporting only the local leg would then claim
    # self.speak holds almost no VRAM while a worker holds a whole model.
    probes = []
    if settings.chatterbox_enabled:
        from . import chatterbox_client

        probes.append(("chatterbox", chatterbox_client.probe_state))
    if settings.kokoro_worker_enabled:
        from . import kokoro_client

        probes.append(("kokoro", kokoro_client.probe_state))

    if not probes:
        return local
    if local["status"] != "ok":
        return local

    aggregated = dict(local)
    held = int(local["held_vram_bytes"])
    resident = bool(local["model_resident"])

    for name, probe in probes:
        worker = await probe()
        if worker is None or worker.get("held_bytes") is None:
            # Expected worker, unaccountable slice → dark. Preserve the residency
            # signal we do know; hold/capacity go null (never a false 0).
            # Under-reporting held is what lets core over-grant (self.ai#74), so
            # unknown-and-say-so beats a confident under-count.
            log.warning(
                "vram-lease: %s worker unaccountable; collapsing to unreachable",
                name,
            )
            return _unreachable_state(resident)
        held += int(worker["held_bytes"])
        resident = resident or bool(worker.get("resident", False))

    aggregated["held_vram_bytes"] = held
    aggregated["model_resident"] = resident
    return aggregated


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


async def _release_chatterbox_worker(
    target_bytes: int, timeout_seconds: float, force: bool = False
) -> None:
    """Best-effort: ask the sibling Chatterbox worker to unload + empty_cache
    toward ``target_bytes`` (INTEGRATION-PLAN-v2.md §1.4). Swallows every failure
    — a down/absent/CPU-only worker (or one with no httpx path) must never fail the
    Kokoro release. Phase-1 scope: this drives the worker's own free; the
    cross-process held/freed AGGREGATION and the async tri-state probe are Phase 2.
    """
    try:
        from . import chatterbox_client

        result = await chatterbox_client.release(target_bytes, timeout_seconds, force)
        if result is not None:
            log.info(
                "vram-lease: chatterbox worker release -> %s (freed=%s bytes)",
                result.get("status"),
                result.get("freed_bytes"),
            )
    except Exception as e:
        log.warning(
            "vram-lease: chatterbox worker release raised (%r); continuing to "
            "live-verify the Kokoro-side delta",
            e,
        )


async def _release_kokoro_worker(
    target_bytes: int, timeout_seconds: float, force: bool = False
) -> None:
    """Best-effort: ask the sibling Kokoro worker to unload toward
    ``target_bytes``. Swallows every failure — a worker that is down, absent, or
    mid-respawn after a cooperative step-aside must never fail the release.

    On a FORCED release the worker may answer and then exit, returning its CUDA
    primary context (~470 MiB). That is the only mechanism that can return a
    context at all, since no torch call frees one.
    """
    try:
        from . import kokoro_client

        result = await kokoro_client.release(target_bytes, timeout_seconds, force)
        if result is not None:
            log.info(
                "vram-lease: kokoro worker release -> %s (freed=%s bytes, exiting=%s)",
                result.get("status"),
                result.get("freed_bytes"),
                result.get("exiting", False),
            )
    except Exception as e:
        log.warning("vram-lease: kokoro worker release raised (%r); continuing", e)


async def _freed_since(before: int):
    """Live-measured bytes freed so far, or ``None`` if unmeasurable.

    Goes through the aggregated probe, so it spans every CUDA context in this
    deployment rather than just the main process's.
    """
    state = await probe_vram_state_full()
    after = state.get("held_vram_bytes")
    if after is None:
        return None
    return max(0, int(before) - int(after))


async def _escalating_worker_release(
    before: int, target_bytes: int, timeout_seconds: float, force: bool = False
) -> None:
    """Ask engines to yield in increasing order of what their yielding COSTS,
    and stop the moment the target is met (self.speak#5 P4).

    Ordering alone is not the point -- P3 already called these in order, but
    called them BOTH, unconditionally, each toward the full target. That
    cold-started ordinary TTS even when the optional engine's yield would have
    been sufficient on its own. Escalation is what makes the ordering mean
    anything.

    The cost gradient, cheapest first:

    1. ``_unload_backend()`` (already done by the caller) -- free under the
       cutover, since main holds no model to drop.
    2. **Chatterbox.** Optional engine. A step-aside costs a slow voice clone
       on the next request. Latency-tolerant by nature.
    3. **Kokoro.** The PRIMARY engine, fronting assistant devices. A step-aside
       costs ordinary TTS a CUDA init plus a model load on the next request.
       Never disturbed unless step 2 was genuinely not enough.

    Each step is measured live rather than assumed: a worker's own report of
    what it freed is not the same as the card actually having that much more
    free, and the whole lease protocol exists because those diverge.

    **Unmeasurable is treated as NOT met**, so we escalate. The asymmetry is
    deliberate: under-delivering VRAM to a priority-10 consumer that asked for
    it means an OOM in the inference brain everything else depends on, while an
    unnecessary escalation costs one cold start. Guessing "probably fine" on the
    expensive side of that trade is how an over-grant happens (self.ai#74).
    """
    engines = int(bool(settings.chatterbox_enabled)) + int(
        bool(settings.kokoro_worker_enabled)
    )
    if engines == 0:
        return

    if engines == 1:
        # Nothing to ORDER, so nothing to escalate: there is exactly one engine
        # and it is going to be asked either way. Skipping the probes here is not
        # an optimisation, it is the absence of a decision -- and it keeps the
        # single-engine path (every deploy today) making exactly the same calls
        # it made before escalation existed, so measuring costs nothing where it
        # can change nothing.
        if settings.chatterbox_enabled:
            await _release_chatterbox_worker(target_bytes, timeout_seconds, force)
        else:
            await _release_kokoro_worker(target_bytes, timeout_seconds, force)
        return

    freed = await _freed_since(before)
    if freed is not None and freed >= target_bytes:
        log.info(
            "vram-lease: target met by the local unload (%s >= %s bytes); "
            "no worker was disturbed",
            freed,
            target_bytes,
        )
        return

    if settings.chatterbox_enabled:
        remaining = target_bytes if freed is None else max(0, target_bytes - freed)
        await _release_chatterbox_worker(remaining, timeout_seconds, force)
        freed = await _freed_since(before)
        if freed is not None and freed >= target_bytes:
            log.info(
                "vram-lease: target met after chatterbox yielded (%s >= %s bytes); "
                "the Kokoro worker was NOT asked to step aside",
                freed,
                target_bytes,
            )
            return

    if settings.kokoro_worker_enabled:
        remaining = target_bytes if freed is None else max(0, target_bytes - freed)
        log.warning(
            "vram-lease: escalating to the Kokoro worker for %s more bytes "
            "(freed so far: %s) -- this costs ordinary TTS a cold start",
            remaining,
            "unmeasurable" if freed is None else freed,
        )
        await _release_kokoro_worker(remaining, timeout_seconds, force)


async def handle_vram_release(
    target_bytes: int, timeout_seconds: float, force: bool = False
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
       model from under an active request). **SKIPPED entirely when ``force``.**
    5. **Free, ESCALATING (#5 P4):** unload the local backend, then ask sibling
       workers to yield in increasing order of what their yielding COSTS --
       chatterbox (a slow voice clone) before Kokoro (a cold start on the
       assistant-device path) -- stopping as soon as the target is met, measured
       live. An engine that did not need to be disturbed is not disturbed.
    6. **Verify live (AC3/AC5):** re-probe held; ``freed = max(0, before -
       after)`` measured from the live delta, never inferred. ``released`` iff
       ``freed >= target_bytes`` else a truthful ``partial``.

    ``force`` (the admin 'Unload All Models' e-stop, "stop now short of pulling
    the plug"): skip the step-4 drain-wait and unload immediately even if a
    synthesis is mid-flight (that request gets a truncated/cut stream — acceptable
    for an e-stop), and forward force to the Chatterbox worker so it does the same.
    Default False keeps the routine, polite, priority-driven broker path exactly
    as it was.
    """
    global _release_in_flight

    # 1. Single-flight (AC7). Check-and-set with no await between → race-free.
    if _release_in_flight:
        return VramReleaseResponse(status="busy", freed_bytes=0)
    _release_in_flight = True
    try:
        # Aggregated baseline: includes the Chatterbox worker's held slice when
        # enabled, so ``freed`` below reflects a worker-side free too (Phase 2).
        state = await probe_vram_state_full()
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

        # 4. AC2/AC6: drain in-flight synthesis, bounded by the timeout — UNLESS
        # this is a force e-stop, which unloads now regardless of in-flight work.
        if not force:
            drained = await wait_for_drain(timeout_seconds)
            if not drained:
                # Still generating at the deadline — never yank a cooperative
                # release out from under it. (A force e-stop skips this entirely.)
                return VramReleaseResponse(status="partial", freed_bytes=0)

        # 5. Free (only if a model is actually resident).
        _unload_backend()
        await _escalating_worker_release(before, target_bytes, timeout_seconds, force)

        # 6. AC3/AC5: verify live via a fresh aggregated probe.
        after_state = await probe_vram_state_full()
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
