"""Kokoro WORKER process — private, localhost-only API (127.0.0.1:8882).

Sibling of the chatterbox worker and shaped like it deliberately: localhost-only
bind, no ticket auth (the MAIN process is the sole ticket-validating surface and
proxies already-authorised work inward), lazy model load, and a cooperative
VRAM-release path that can step aside by exiting.

**P2 SCOPE: this worker exists and serves. Nothing calls it yet.** Main still
runs Kokoro in-process; the cutover is P3. Until then launching this would make
VRAM strictly WORSE — two CUDA primary contexts (~470 MiB each) instead of one —
which is why the entrypoint launch is opt-in and default OFF. The win only
arrives when main stops importing Kokoro.

Endpoints:
  GET  /health        liveness for the launcher
  POST /generate      framed audio stream (see ``codec``) — the P1 wire format
  GET  /vram-state    {held_bytes: int|null, resident: bool}
  POST /vram-release  {target_bytes, timeout_seconds, force} -> {status, freed_bytes}

Three things the P1 review said P2 had to decide, decided here:

1.  **The blocking generator is off the event loop** via ``bridge``. An
    ``async def`` around a synchronous ``for result in pipeline(...)`` still
    blocks; ``run_in_threadpool`` cannot help a generator. Without this the
    broker's ``/vram-state`` probe times out during every long utterance and
    reads self.speak as dark.

2.  **Generation is serialised, and that is not a free choice.** One model on
    one GPU cannot run two generations at once. The cost is real — a second
    caller waits for the first — but the alternative is not concurrency, it is
    corruption or an OOM. What matters is that the wait is BOUNDED, which is
    what (3) provides.

3.  **The stream has a whole-stream deadline, not just a per-read timeout** —
    implemented in ``stream``, which is where the bounds and the framing live.
    This is the one that would have been a production incident; the reasoning is
    written out there.

Framing and bounding are deliberately NOT in this module. They have nothing to
do with HTTP, and keeping them in ``stream`` means the part with the failure
modes is testable without FastAPI, torch, or a GPU. This file is wiring.
"""

import asyncio
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import codec, reclaim, stream, vram

logging.basicConfig(level=os.environ.get("KOKORO_WORKER_LOG_LEVEL", "INFO"))
log = logging.getLogger("kokoro_worker")

# Localhost-only by default and MUST stay that way (never core-facing).
HOST = os.environ.get("KOKORO_WORKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("KOKORO_WORKER_PORT", "8882"))

app = FastAPI(title="kokoro-worker", docs_url=None, redoc_url=None)

# One model, one GPU, one generation at a time. Bounded by the deadline above.
_generate_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    text: str
    voice: str
    # The voice tensor path, RESOLVED BY MAIN. Not an optimisation: for a
    # combined voice ("af_jadzia+af_jessica") main blends the tensors itself and
    # writes the result to a temp file, so the name alone cannot reproduce it.
    # And KokoroV1.generate treats a bare string as a PATH, not a name, so a
    # name-only request fails for EVERY voice, not just combined ones.
    #
    # Safe to pass a path across this boundary because the worker is a sibling
    # PROCESS in the same container, not a separate pod -- same filesystem, same
    # /tmp. If that ever stops being true, this field has to become the tensor
    # itself rather than a reference to one.
    voice_path: str | None = None
    speed: float = 1.0
    lang_code: str | None = None
    return_timestamps: bool = False


class PhonemeRequest(BaseModel):
    phonemes: str
    voice: str
    # Resolved by main, same as /generate — see GenerateRequest.voice_path.
    voice_path: str | None = None
    speed: float = 1.0
    lang_code: str = "a"


class ReleaseRequest(BaseModel):
    target_bytes: int
    timeout_seconds: float
    force: bool = False


async def _engine():
    """Resolve THIS process's in-process Kokoro, loading it on first use.

    Two things this must get right, both of which broke the first enablement:

    1.  ``force_in_process_backend()`` -- the worker shares
        KOKORO_WORKER_ENABLED with main, so without declaring its role its own
        ModelManager would select KokoroWorkerBackend and proxy to ITSELF.
    2.  ``ensure_loaded()`` -- nothing else initialises the backend in this
        process. Without it every request died on "Backend not initialized",
        which is exactly what happened in production.

    Still lazy: the import and the load both happen on the first /generate, so
    a worker that never serves never initialises CUDA.
    """
    from ..inference import model_manager

    model_manager.force_in_process_backend()
    manager = await model_manager.get_manager()
    await manager.ensure_loaded()
    return manager


async def _frames(req: GenerateRequest):
    """Thin adapter: resolve the engine, then delegate to ``stream``.

    The framing and the bounds live in ``stream`` on purpose — they are where
    the failure modes are, and they must be testable without FastAPI.
    """
    manager = await _engine()
    # Pass (name, path) when main resolved a path, which is the form the backend
    # expects; a bare string would be read as a filesystem path.
    voice = (req.voice, req.voice_path) if req.voice_path else req.voice
    async for frame in stream.encode_stream(
        manager,
        text=req.text,
        voice=voice,
        speed=req.speed,
        lang_code=req.lang_code,
        return_timestamps=req.return_timestamps,
    ):
        yield frame


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(req: GenerateRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="empty text")
    if not req.voice:
        raise HTTPException(status_code=400, detail="no voice")

    async def _body():
        # The lock is taken INSIDE the body, not around it: acquiring before
        # returning the StreamingResponse would serialise header delivery too,
        # so a queued caller would not even get a response until the one ahead
        # finished. It is released when the generator closes, including on
        # client disconnect (Starlette closes the body generator).
        async with _generate_lock:
            async for frame in _frames(req):
                yield frame

    return StreamingResponse(_body(), media_type="application/octet-stream")


@app.post("/generate-from-phonemes")
async def generate_from_phonemes(req: PhonemeRequest):
    """Single-shot phoneme synthesis (self.speak#7).

    Returns RAW float32 PCM rather than P1 frames: this path produces one
    result, not a stream, so framing would buy nothing. Same shape as the
    chatterbox worker's /synth.

    Shares ``_generate_lock`` with /generate — one model, one GPU, one
    generation at a time — and runs the blocking pipeline call in a threadpool
    so /health and /vram-state stay answerable. (A single call CAN use
    run_in_threadpool; only a generator cannot, which is why /generate needs
    the bridge.)
    """
    if not req.phonemes or not req.phonemes.strip():
        raise HTTPException(status_code=400, detail="empty phonemes")
    if not req.voice_path:
        raise HTTPException(status_code=400, detail="no voice_path")

    manager = await _engine()
    backend = manager.get_backend()
    if backend is None or not hasattr(backend, "generate_from_phonemes"):
        raise HTTPException(status_code=503, detail="backend cannot do phonemes")

    async with _generate_lock:
        try:
            audio = await backend.generate_from_phonemes(
                req.phonemes, req.voice_path, req.speed, req.lang_code
            )
        except Exception as e:
            log.exception("kokoro-worker: phoneme generation failed")
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    pcm = np.ascontiguousarray(audio, dtype=np.float32).tobytes()
    return Response(
        content=pcm,
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(codec.DEFAULT_SAMPLE_RATE)},
    )


@app.get("/vram-state")
async def vram_state():
    return vram.probe()


@app.post("/vram-release")
async def vram_release(req: ReleaseRequest):
    # Same contract as the chatterbox worker: cooperative by default (drain-wait
    # for an in-flight generation), force = e-stop that skips the wait. The
    # drain-wait is only meaningful because the whole-stream deadline guarantees
    # the lock is released in bounded time.
    acquired = False
    if not req.force:
        try:
            await asyncio.wait_for(
                _generate_lock.acquire(), timeout=max(0.0, req.timeout_seconds)
            )
            acquired = True
        except asyncio.TimeoutError:
            return {"status": "partial", "freed_bytes": 0}
    try:
        before = vram._reserved_bytes()
        vram.unload()
        after = vram._reserved_bytes()
        if before is None or after is None:
            return {"status": "partial", "freed_bytes": 0}
        freed = max(0, int(before) - int(after))
        if freed >= req.target_bytes:
            return {"status": "released", "freed_bytes": freed}

        # What is left is this process's CUDA PRIMARY CONTEXT (~470 MiB on the
        # 4090). No torch call frees it; only exiting does. On a FORCED release
        # we step aside and the entrypoint respawn loop brings us back.
        result = {"status": "partial", "freed_bytes": freed}
        if req.force and reclaim.RELEASE_MAY_EXIT and reclaim.context_held():
            if reclaim.schedule_exit(
                f"forced release asked for {int(req.target_bytes)} B, unloading "
                f"freed {freed} B; the remainder is this process's CUDA context"
            ):
                result["exiting"] = True
        return result
    finally:
        if acquired:
            _generate_lock.release()


if __name__ == "__main__":
    import uvicorn

    log.info("kokoro worker starting on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
