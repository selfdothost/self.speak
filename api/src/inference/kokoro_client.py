"""Main-process backend that runs Kokoro in the WORKER, not in this process.

This is the P3 cutover (self.speak#5). It presents the ``ModelBackend`` surface
``TTSService`` already expects, but every generation crosses a localhost HTTP
boundary in the P1 frame format instead of touching torch here.

**The point is what this file does NOT do: allocate on the GPU.** A CUDA primary
context is ~470 MiB on the 4090, created by a process's first real allocation and
released only by that process exiting. Main is the pod's sole liveness path, so
it can never exit, so its context was permanently unreclaimable — priority 5
holding VRAM against a priority-10 consumer with no lever. Moving generation out
means main never creates one at all, and the 470 MiB disappears rather than
merely becoming reclaimable.

So this module must stay torch-free on every path that matters. It imports
``numpy`` (for the decoded PCM) and ``httpx``, and nothing that allocates.

Failure taxonomy, kept deliberately narrow and distinct:

* ``KokoroWorkerUnavailable`` — transport: unreachable, timed out, non-200.
  Retryable in principle; the worker may simply be respawning after a
  cooperative VRAM step-aside.
* ``KokoroWorkerProtocolError`` (from ``codec``) — the peer is not speaking this
  protocol. Retrying will not help.

The two must stay distinguishable. Collapsing them is how a dead worker ends up
masquerading as a content error.
"""

import asyncio
import logging
import os
from typing import AsyncGenerator, Optional

import httpx
import numpy as np

from ..core.config import settings
from ..kokoro_worker import codec
from .base import AudioChunk, BaseModelBackend

log = logging.getLogger(__name__)

# Connect is short so a dead or still-binding worker fails fast rather than
# hanging a request. The read budget is per-socket-read, not whole-stream: the
# worker owns the whole-stream deadline (kokoro_worker/stream.py), because it is
# the side that can actually stop generating.
_CONNECT_TIMEOUT = 2.0
_STREAM_READ_TIMEOUT = 120.0
_CONTROL_READ_TIMEOUT = 5.0

# One retry across a worker respawn. NOT general-purpose resilience: this exists
# because a forced VRAM release DELIBERATELY exits the worker (kokoro_worker/
# reclaim.py) to return its ~470 MiB CUDA primary context. The entrypoint
# respawn loop brings it straight back, but any request landing in that window
# would otherwise get a 5xx -- on the path that serves assistant devices, for a
# reclamation the system asked for on purpose. In-process Kokoro had no such
# window, so without this the cutover is an availability regression.
_RESPAWN_RETRIES = int(os.environ.get("KOKORO_CLIENT_RESPAWN_RETRIES", "1"))
# reclaim.py waits ~1.5s before exiting, then the loop restarts and uvicorn
# binds. Long enough to clear a respawn, short enough not to stack latency onto
# a genuinely dead worker.
_RESPAWN_BACKOFF_SECONDS = float(
    os.environ.get("KOKORO_CLIENT_RESPAWN_BACKOFF", "2.0")
)


def _make_client(timeout: httpx.Timeout) -> httpx.AsyncClient:
    """Construct the HTTP client for a worker call.

    A seam, deliberately. Both bugs found in this epic lived exactly where the
    tests stopped, and the client<->worker boundary had NO test crossing it at
    all -- which is how a voice argument that breaks every synthesis survived a
    green suite. This lets a test drive the real client against the real worker
    app over an in-process ASGI transport, so the seam is exercised rather than
    assumed on both sides.
    """
    return httpx.AsyncClient(timeout=timeout)


class KokoroWorkerUnavailable(RuntimeError):
    """The Kokoro worker was unreachable, timed out, or returned non-200."""


def _url(path: str) -> str:
    return settings.kokoro_worker_url.rstrip("/") + path


class KokoroWorkerBackend(BaseModelBackend):
    """A Kokoro backend whose model lives in another process.

    ``streams_text`` is the capability ``TTSService`` branches on. It used to
    branch on ``isinstance(backend, KokoroV1)``, which silently means "is the
    concrete in-process class" — a worker-backed backend would have fallen
    through to the legacy tokens-in/one-blob-out path and produced wrong audio
    rather than an error. The question being asked is about the CONTRACT (takes
    text, yields chunks), so it is now asked directly.
    """

    streams_text = True

    def __init__(self):
        super().__init__()
        self._device = "cuda" if settings.use_gpu else "cpu"
        # Mirrors the worker's residency; refreshed by probes, never by loading
        # anything here.
        self._resident = False

    # -- lifecycle -----------------------------------------------------------

    async def load_model(self, path: str) -> None:
        """No-op: the worker lazy-loads on its first /generate.

        Deliberately not a pre-warm. Reaching across to force a load would put
        model-loading latency on main's startup path for a model main does not
        own, and would defeat the worker's own lazy-load-after-step-aside
        behaviour.
        """
        log.info("kokoro-client: load_model is a no-op; the worker loads lazily")

    def unload(self) -> None:
        """Ask the worker to drop its model. Best-effort and non-fatal.

        Synchronous by interface contract, so this cannot await. The real
        reclamation path is the VRAM lease (``vram_lease``), which calls the
        worker's ``/vram-release`` directly and can also make it step aside.
        """
        self._resident = False

    @property
    def is_loaded(self) -> bool:
        return self._resident

    @property
    def device(self) -> str:
        return self._device

    # -- generation ----------------------------------------------------------

    async def generate(
        self,
        text: str,
        voice,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        return_timestamps: bool = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Stream audio from the worker, decoding P1 frames into AudioChunks.

        ``voice`` arrives as either a name or a ``(name, path)`` tuple — the
        in-process backend accepted both, so this does too rather than pushing
        the difference onto callers.

        **The PATH must cross the boundary, not just the name.** TTSService
        resolves it first, and for a combined voice (``af_jadzia+af_jessica``)
        that means blending the tensors and writing a temp file — a name alone
        cannot reproduce it. Worse, ``KokoroV1.generate`` reads a bare string as
        a filesystem PATH rather than a voice name, so dropping the path breaks
        EVERY voice, not only combined ones.
        """
        if isinstance(voice, (tuple, list)):
            voice_name, voice_path = voice[0], voice[1]
        else:
            voice_name, voice_path = voice, None

        payload = {
            "text": text,
            "voice": voice_name,
            "voice_path": voice_path,
            "speed": speed,
            "lang_code": lang_code,
            "return_timestamps": bool(return_timestamps),
        }

        # Retry only while NOTHING has been yielded. Once a chunk has reached the
        # caller the stream cannot be rewound -- replaying from the start would
        # duplicate audio, which is worse than the error.
        attempts = 1 + max(0, _RESPAWN_RETRIES)
        for attempt in range(attempts):
            delivered = False
            try:
                async for chunk in self._stream_once(payload):
                    delivered = True
                    yield chunk
                return
            except KokoroWorkerUnavailable as e:
                if delivered or attempt == attempts - 1:
                    raise
                log.warning(
                    "kokoro-client: worker unavailable (%s); retrying once in %.1fs "
                    "-- it may be respawning after a forced VRAM release",
                    e,
                    _RESPAWN_BACKOFF_SECONDS,
                )
                await asyncio.sleep(_RESPAWN_BACKOFF_SECONDS)

    async def _stream_once(self, payload: dict) -> AsyncGenerator[AudioChunk, None]:
        """One attempt at the worker. Raises; never retries."""
        decoder = codec.FrameDecoder()
        timeout = httpx.Timeout(
            _STREAM_READ_TIMEOUT, connect=_CONNECT_TIMEOUT, read=_STREAM_READ_TIMEOUT
        )

        try:
            async with _make_client(timeout) as client:
                async with client.stream("POST", _url("/generate"), json=payload) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread())[:512]
                        raise KokoroWorkerUnavailable(
                            f"kokoro worker returned {resp.status_code}: {body!r}"
                        )
                    self._resident = True

                    async for raw in resp.aiter_bytes():
                        for frame in decoder.feed(raw):
                            if isinstance(frame, codec.ErrorFrame):
                                # The worker told us it failed. That is a real
                                # answer, not a transport fault, and it must not
                                # be reported as one.
                                raise RuntimeError(
                                    f"kokoro worker: {frame.error_class}: {frame.message}"
                                )
                            if isinstance(frame, codec.EndFrame):
                                return
                            yield AudioChunk(
                                frame.audio,
                                word_timestamps=frame.word_timestamps,
                            )
        except (httpx.HTTPError, httpx.StreamError) as e:
            # Covers mid-stream deaths too: a hard-killed producer surfaces as
            # RemoteProtocolError out of aiter_bytes(), long before finish()
            # would ever run. Without this branch the caller would see a raw
            # httpx type and could not tell it from a protocol violation.
            raise KokoroWorkerUnavailable(f"kokoro worker stream failed: {e}") from e
        finally:
            # NOT decoder.finish(). A caller that stops early -- an assistant
            # device hanging up mid-utterance, which is normal and frequent --
            # closes this generator, and finish() would then raise "stream
            # truncated" during unwinding for a perfectly benign hangup. The
            # END-frame `return` above is what confirms a complete stream.
            if not decoder.ended:
                log.debug("kokoro-client: stream closed before END (caller stopped early)")


    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice_path: str,
        speed: float = 1.0,
        lang_code: str = "a",
    ):
        """Phoneme synthesis via the worker (self.speak#7).

        Raw float32 PCM rather than P1 frames: one result, not a stream, so
        framing would buy nothing. Mirrors the chatterbox worker's /synth.
        """
        payload = {
            "phonemes": phonemes,
            "voice": "",
            "voice_path": voice_path,
            "speed": speed,
            "lang_code": lang_code,
        }
        timeout = httpx.Timeout(
            _STREAM_READ_TIMEOUT, connect=_CONNECT_TIMEOUT, read=_STREAM_READ_TIMEOUT
        )
        try:
            async with _make_client(timeout) as client:
                resp = await client.post(_url("/generate-from-phonemes"), json=payload)
                if resp.status_code != 200:
                    raise KokoroWorkerUnavailable(
                        f"kokoro worker returned {resp.status_code}: {resp.content[:512]!r}"
                    )
                self._resident = True
                # bytearray so the array is writable -- the caller may scale it.
                return np.frombuffer(bytearray(resp.content), dtype="<f4")
        except (httpx.HTTPError, httpx.StreamError) as e:
            raise KokoroWorkerUnavailable(
                f"kokoro worker phoneme request failed: {e}"
            ) from e


async def probe_state() -> Optional[dict]:
    """GET /vram-state. Returns ``None`` when the worker is unreachable.

    ``None`` is not zero. A worker that cannot be reached holds an UNKNOWN
    amount, and reporting 0 would let the aggregator over-report free VRAM,
    which over-grants the card.
    """
    try:
        async with _make_client(
            httpx.Timeout(_CONTROL_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)
        ) as client:
            resp = await client.get(_url("/vram-state"))
            if resp.status_code != 200:
                return None
            return resp.json()
    except Exception as e:  # noqa: BLE001 - unreachable is an answer, not a crash
        log.debug("kokoro-client: vram-state probe failed (%r)", e)
        return None


async def release(target_bytes: int, timeout_seconds: float, force: bool = False):
    """POST /vram-release. Returns ``None`` when the worker is unreachable."""
    try:
        async with _make_client(
            httpx.Timeout(
                max(_CONTROL_READ_TIMEOUT, timeout_seconds + 5.0),
                connect=_CONNECT_TIMEOUT,
            )
        ) as client:
            resp = await client.post(
                _url("/vram-release"),
                json={
                    "target_bytes": int(target_bytes),
                    "timeout_seconds": float(timeout_seconds),
                    "force": bool(force),
                },
            )
            if resp.status_code != 200:
                return None
            return resp.json()
    except Exception as e:  # noqa: BLE001
        log.warning("kokoro-client: vram-release failed (%r)", e)
        return None
