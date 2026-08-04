"""Chatterbox WORKER process — private, localhost-only control API (127.0.0.1:8881).

The sibling of the main uvicorn inside the ONE self-speak-gpu container
(INTEGRATION-PLAN-v2.md §1.2 / §1.4). Runs under ``/app/.venv-chatterbox`` (torch
2.6). It is:

  * localhost-ONLY (binds 127.0.0.1) — never core-facing, never reachable off-box,
    so it carries NO ticket auth and NO SERVICE_AUTH_SECRET. The MAIN process is the
    sole ticket-validating surface; it proxies already-authorised work inward.
  * the owner of the Chatterbox model (lazy load) and its CUDA context.
  * a RAW-waveform producer — /synth and /clone return bare float32 PCM +
    ``X-Sample-Rate``; the MAIN process does all mp3/opus/wav encoding.

Endpoints (all consumed only by the main process's ``inference/chatterbox_client``):
  GET  /health        liveness for the launcher + main's availability check
  POST /synth         {text, audio_prompt_path?, exaggeration?, cfg_weight?} → PCM
  POST /clone         same as /synth (clone == synth-with-reference for Chatterbox)
  GET  /vram-state    {held_bytes: int|null, resident: bool}
  POST /vram-release  {target_bytes, timeout_seconds} → {status, freed_bytes}

Launched by ``docker/scripts/entrypoint.sh`` as
``/app/.venv-chatterbox/bin/python -m api.src.chatterbox_worker.main`` in a bounded
respawn loop; a crash never takes the pod down (only main is on the liveness path).
"""

import asyncio
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import vram
from .engine import ChatterboxEngine

logging.basicConfig(level=os.environ.get("CHATTERBOX_LOG_LEVEL", "INFO"))
log = logging.getLogger("chatterbox_worker")

# Bind is localhost-only by default and MUST stay that way (never core-facing).
HOST = os.environ.get("CHATTERBOX_WORKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("CHATTERBOX_WORKER_PORT", "8881"))

app = FastAPI(title="chatterbox-worker", docs_url=None, redoc_url=None)

# Serialise GPU work: one model, one time-slice — never run two generations (or a
# generation and an unload) at once. Also gives /vram-release an in-worker
# drain-wait (§2.6 v0 serialisation policy), keeping /health responsive because
# the blocking torch call runs in a threadpool, not on the event loop.
_synth_lock = asyncio.Lock()


class SynthRequest(BaseModel):
    text: str
    audio_prompt_path: str | None = None
    exaggeration: float | None = None
    cfg_weight: float | None = None


class BlendReference(BaseModel):
    audio_prompt_path: str
    weight: float = 1.0


class BlendRequest(BaseModel):
    text: str
    references: list[BlendReference]
    exaggeration: float | None = None
    cfg_weight: float | None = None


class ReleaseRequest(BaseModel):
    target_bytes: int
    timeout_seconds: float
    force: bool = False  # e-stop: skip the drain-wait, unload the model NOW


def _run_generate(req: SynthRequest) -> tuple[bytes, int]:
    """Blocking synthesis (runs in a threadpool). Returns (raw float32 PCM, sr)."""
    controls = {}
    if req.exaggeration is not None:
        controls["exaggeration"] = req.exaggeration
    if req.cfg_weight is not None:
        controls["cfg_weight"] = req.cfg_weight
    audio, sr = ChatterboxEngine.instance().generate(
        req.text, audio_prompt_path=req.audio_prompt_path, **controls
    )
    pcm = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    return pcm, sr


def _run_blend(req: BlendRequest) -> tuple[bytes, int]:
    """Blocking multi-reference blend (runs in a threadpool)."""
    refs = [(r.audio_prompt_path, r.weight) for r in req.references]
    audio, sr = ChatterboxEngine.instance().generate_blend(
        req.text,
        refs,
        exaggeration=req.exaggeration,
        cfg_weight=req.cfg_weight,
    )
    pcm = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    return pcm, sr


async def _synthesize(req: SynthRequest) -> Response:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    async with _synth_lock:
        try:
            pcm, sr = await run_in_threadpool(_run_generate, req)
        except Exception as e:
            log.exception("chatterbox synthesis failed")
            raise HTTPException(status_code=500, detail=f"synthesis failed: {e}") from e
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/synth")
async def synth(req: SynthRequest) -> Response:
    return await _synthesize(req)


@app.post("/clone")
async def clone(req: SynthRequest) -> Response:
    # Clone == synth-with-reference for Chatterbox; identical worker path.
    return await _synthesize(req)


@app.post("/blend")
async def blend(req: BlendRequest) -> Response:
    # Weighted multi-reference blend — interpolates the clips' speaker embeddings
    # into a NEW voice (see ChatterboxEngine.generate_blend). One reference in the
    # list degrades to a plain clone.
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    if not req.references:
        raise HTTPException(status_code=400, detail="no reference clips")
    async with _synth_lock:
        try:
            pcm, sr = await run_in_threadpool(_run_blend, req)
        except Exception as e:
            log.exception("chatterbox blend failed")
            raise HTTPException(status_code=500, detail=f"blend failed: {e}") from e
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(sr)},
    )


@app.get("/vram-state")
async def vram_state():
    return vram.probe()


@app.post("/vram-release")
async def vram_release(req: ReleaseRequest):
    # Cooperative (default): drain-wait up to timeout_seconds for an in-flight
    # synthesis to release the lock; if it doesn't, do NOT yank the model out from
    # under it. FORCE (e-stop): skip the lock entirely and unload NOW — an
    # in-flight generate gets cut, which is the whole point of "stop now".
    acquired = False
    if not req.force:
        try:
            await asyncio.wait_for(
                _synth_lock.acquire(), timeout=max(0.0, req.timeout_seconds)
            )
            acquired = True
        except asyncio.TimeoutError:
            return {"status": "partial", "freed_bytes": 0}
    try:
        before = vram._reserved_bytes()
        vram.unload()
        after = vram._reserved_bytes()
        if before is None or after is None:
            # Couldn't measure across the free — can't confirm a delta.
            return {"status": "partial", "freed_bytes": 0}
        freed = max(0, int(before) - int(after))
        status = "released" if freed >= req.target_bytes else "partial"
        return {"status": status, "freed_bytes": freed}
    finally:
        if acquired:
            _synth_lock.release()


if __name__ == "__main__":
    import uvicorn

    log.info("chatterbox worker starting on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
