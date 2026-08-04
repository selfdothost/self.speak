"""System control-plane endpoints — the VRAM-lease protocol surface.

Introduces the ``/api/system/*`` namespace to self.speak for the first time
(cavekit-vram-lease-client). Before this, the only control endpoint was the
unticketed ``GET /api/voices`` (``routers/control.py``); these two routes are the
first ticket-gated control endpoints on this backend:

* ``GET  /api/system/vram-state``  — R1, ticket scope ``system:read``
* ``POST /api/system/vram-release`` — R2, ticket scope ``system:write``

Both are the authoritative wire contract core's transport parses; the actual
introspection / orchestration lives in ``inference/vram_lease.py``.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..core.auth import SCOPE_SYSTEM_READ, SCOPE_SYSTEM_WRITE, require_scope
from ..inference.vram_lease import handle_vram_release, probe_vram_state_full
from ..structures.schemas import (
    VramReleaseRequest,
    VramReleaseResponse,
    VramStateResponse,
)

router = APIRouter(tags=["system"])
log = logging.getLogger(__name__)


@router.get("/api/system/vram-state")
async def vram_state(
    _auth=Depends(require_scope(SCOPE_SYSTEM_READ)),
) -> VramStateResponse:
    """Report currently-held VRAM and total addressable capacity, in bytes
    (cavekit-vram-lease-client R1 — T-007).

    Computes the figures live on every call via ``probe_vram_state_full()`` (the
    aggregated Kokoro + Chatterbox-worker probe; never cached — R1-AC2) and
    marshals the tri-state result straight through into
    ``VramStateResponse``. **Always returns HTTP 200** (R1-AC3): when the GPU is
    unreachable (case b) or absent (case c) the null/zero figures plus the
    ``status`` / ``gpu_reachable`` fields carry the signal — an HTTP error would
    hide the very state core needs to read. Even an unexpected probe exception is
    caught here and surfaced as a 200 ``unreachable`` body with null figures,
    never a 500 that would mask the state.
    """
    try:
        probe = await probe_vram_state_full()
    except Exception as e:
        # A probe should never raise (its CUDA calls are already guarded, and the
        # worker leg swallows its own errors), but if it somehow does, still answer
        # 200 with a truthful unreachable body —
        # null figures, never a false 0, never a 5xx (R1-AC3/AC5).
        log.warning(
            "vram-lease: probe_vram_state raised unexpectedly (%r); returning "
            "200 unreachable",
            e,
        )
        probe = {
            "held_vram_bytes": None,
            "total_capacity_bytes": None,
            # Unknown, not empty — see VramStateResponse (self.ai#74).
            "device_used_bytes": None,
            "device_total_bytes": None,
            "gpu_reachable": False,
            "status": "unreachable",
            "model_resident": False,
        }
    return VramStateResponse(**probe)


@router.post("/api/system/vram-release")
async def vram_release(
    req: VramReleaseRequest,
    _auth=Depends(require_scope(SCOPE_SYSTEM_WRITE)),
) -> VramReleaseResponse:
    """Answer a release-request by draining in-flight synthesis (bounded by the
    timeout) then unloading the Kokoro backend, reporting the live-measured freed
    amount (cavekit-vram-lease-client R2 — T-008).

    ``async`` because it awaits the drain. Delegates the whole sequence
    (single-flight guard, drain-wait, live-verified free, CPU no-op) to
    ``handle_vram_release()``. A ``"busy"`` result (another release already in
    flight, R2-AC7) is surfaced as **HTTP 409** so a racing caller gets the
    unambiguous signal core's transport reads as "not now", rather than a
    misleading 200-with-zero.
    """
    result = await handle_vram_release(req.target_bytes, req.timeout_seconds, req.force)
    if result.status == "busy":
        raise HTTPException(
            status_code=409,
            detail="A VRAM release-request is already in flight on this instance",
        )
    return result
