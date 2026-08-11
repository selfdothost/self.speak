"""Turn an engine's audio chunks into a bounded stream of P1 frames.

Deliberately free of FastAPI: framing and bounding a stream has nothing to do
with HTTP, and keeping them apart means this — the part with the failure modes —
is testable without a web framework, a GPU, or torch. ``main`` is then a thin
wiring layer.

The bounds here are the ones the P1 review said P2 had to add, and one of them
prevents a specific production incident rather than a hypothetical:

A worker that wedges into emitting one small frame every ~60s never trips a
per-read timeout, so the stream never ends. Main wraps every synthesis in
``track_synthesis()``, so ``_synthesis_count`` stays above zero indefinitely.
Every subsequent cooperative ``/vram-release`` then hits ``wait_for_drain()``,
times out, and returns ``partial``/``freed_bytes=0`` with NO unload. self.speak
becomes permanently unreclaimable to the shared-4090 broker until a force e-stop
or a pod restart — precisely the starvation the lease protocol exists to
prevent. A whole-stream deadline turns "permanent" into "bounded and loud".
"""

import logging
import os
import time

from . import codec
from .bridge import BridgeTimeout, GeneratorBridge

log = logging.getLogger("kokoro_worker.stream")

# Generous enough that no legitimate synthesis approaches them: a 450-token
# chunk is a handful of frames and a few seconds. A guard that fires on ordinary
# traffic is worse than no guard, because it trains people to raise it.
STREAM_DEADLINE_SECONDS = float(os.environ.get("KOKORO_STREAM_DEADLINE_SECONDS", "600"))
STREAM_MAX_FRAMES = int(os.environ.get("KOKORO_STREAM_MAX_FRAMES", "10000"))
STREAM_MAX_BYTES = int(os.environ.get("KOKORO_STREAM_MAX_BYTES", str(512 << 20)))
# Per-item timeout inside the bridge — distinct from the whole-stream deadline.
ITEM_TIMEOUT_SECONDS = float(os.environ.get("KOKORO_ITEM_TIMEOUT_SECONDS", "120"))


async def encode_stream(
    manager,
    *,
    text: str,
    voice: str,
    speed: float = 1.0,
    lang_code: str | None = None,
    return_timestamps: bool = False,
    deadline_seconds: float | None = None,
    max_frames: int | None = None,
    max_bytes: int | None = None,
    item_timeout: float | None = None,
):
    """Yield encoded frames for one generation. ALWAYS terminates the stream.

    Every exit path emits either END or ERROR. A body that merely stops is the
    one outcome a consumer cannot interpret — indistinguishable from a hard kill
    — so it must never be how this returns. That is the same failure shape as
    self.speak#6, where a stream that ended quietly read as a complete one.
    """
    deadline = STREAM_DEADLINE_SECONDS if deadline_seconds is None else deadline_seconds
    cap_frames = STREAM_MAX_FRAMES if max_frames is None else max_frames
    cap_bytes = STREAM_MAX_BYTES if max_bytes is None else max_bytes
    per_item = ITEM_TIMEOUT_SECONDS if item_timeout is None else item_timeout

    started = time.monotonic()
    seq = 0
    total_bytes = 0
    bridge = None

    try:
        def _factory():
            return manager.generate(
                text,
                voice,
                speed=speed,
                lang_code=lang_code,
                return_timestamps=return_timestamps,
            )

        bridge = GeneratorBridge(_factory, item_timeout=per_item).start()

        async for chunk in bridge.__aiter__():
            elapsed = time.monotonic() - started
            if elapsed > deadline:
                raise TimeoutError(
                    f"stream exceeded the {deadline}s whole-stream deadline "
                    f"after {seq} frame(s)"
                )
            if seq >= cap_frames:
                raise RuntimeError(f"stream exceeded {cap_frames} frames")
            if total_bytes > cap_bytes:
                raise RuntimeError(f"stream exceeded {cap_bytes} bytes")

            audio = getattr(chunk, "audio", None)
            if audio is None or len(audio) == 0:
                # The engine yields empty chunks; skipping one is correct. Unlike
                # the old in-process path this costs no silence, because seq and
                # the END count still describe the stream honestly.
                continue

            frame = codec.encode_audio_frame(
                audio, seq=seq, word_timestamps=getattr(chunk, "word_timestamps", None)
            )
            total_bytes += len(frame)
            seq += 1
            yield frame

        yield codec.encode_end_frame(seq)

    except BridgeTimeout as e:
        log.warning("kokoro-stream: %s", e)
        yield codec.encode_error_frame("TimeoutError", str(e), {"frames": seq})
    except Exception as e:  # noqa: BLE001 - the client must learn what happened
        log.exception("kokoro-stream: generation failed")
        yield codec.encode_error_frame(type(e).__name__, str(e), {"frames": seq})
    finally:
        if bridge is not None:
            bridge.stop()
