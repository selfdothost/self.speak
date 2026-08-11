"""Client retry across a worker respawn (self.speak#5).

A forced VRAM release DELIBERATELY exits the Kokoro worker to return its
~470 MiB CUDA primary context — that is the only mechanism that can return a
context at all. The entrypoint respawn loop brings it right back, but a request
landing in that window would otherwise get a 5xx, on the path that serves
assistant devices, for a reclamation the system asked for on purpose.

In-process Kokoro had no such window, so without this the cutover is an
availability regression rather than a pure win.

The interesting half is what must NOT be retried.
"""

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from api.src.inference import kokoro_client
from api.src.inference.base import AudioChunk


def _chunk(v=0.2):
    return AudioChunk(np.full(8, v, dtype=np.float32), word_timestamps=None)


def _streamer(*attempts):
    """Fake _stream_once: each entry is either a list of chunks or an exception."""
    calls = {"n": 0}

    async def _fake(self, payload):
        i = calls["n"]
        calls["n"] += 1
        outcome = attempts[i] if i < len(attempts) else attempts[-1]
        if isinstance(outcome, BaseException):
            raise outcome
        for c in outcome:
            yield c

    return _fake, calls


@pytest.fixture(autouse=True)
def _no_sleep():
    """Never actually wait the backoff in tests."""
    with patch.object(kokoro_client.asyncio, "sleep", AsyncMock()):
        yield


async def _drain(backend):
    return [c async for c in backend.generate("hi", ("af_heart", "/v.pt"))]


@pytest.mark.asyncio
async def test_a_respawning_worker_is_retried_and_succeeds():
    """The window a forced release opens on purpose."""
    fake, calls = _streamer(
        kokoro_client.KokoroWorkerUnavailable("connection refused"),
        [_chunk(0.5)],
    )
    with patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", fake):
        got = await _drain(kokoro_client.KokoroWorkerBackend())

    assert len(got) == 1
    assert calls["n"] == 2, "did not retry"


@pytest.mark.asyncio
async def test_a_genuinely_dead_worker_still_raises():
    """Retry must not turn a dead worker into a hang or a silent empty stream."""
    fake, calls = _streamer(
        kokoro_client.KokoroWorkerUnavailable("refused"),
        kokoro_client.KokoroWorkerUnavailable("refused again"),
    )
    with patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", fake):
        with pytest.raises(kokoro_client.KokoroWorkerUnavailable):
            await _drain(kokoro_client.KokoroWorkerBackend())

    assert calls["n"] == 2, "retried more or fewer times than configured"


@pytest.mark.asyncio
async def test_a_stream_that_already_delivered_audio_is_NOT_retried():
    """The important half.

    Once a chunk has reached the caller the stream cannot be rewound. Replaying
    from the start would duplicate audio in the middle of an utterance, which is
    worse than surfacing the error — and it is exactly the kind of
    plausible-looking retry that produces garbled speech rather than a failure
    anyone notices.
    """

    async def _fake(self, payload):
        yield _chunk(0.1)
        raise kokoro_client.KokoroWorkerUnavailable("died mid-stream")

    with patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", _fake):
        backend = kokoro_client.KokoroWorkerBackend()
        got = []
        with pytest.raises(kokoro_client.KokoroWorkerUnavailable):
            async for c in backend.generate("hi", ("af_heart", "/v.pt")):
                got.append(c)

    assert len(got) == 1, "the delivered chunk should still have reached the caller"


@pytest.mark.asyncio
async def test_a_worker_side_error_is_NOT_retried():
    """An ERROR frame is a real answer, not a transport fault.

    Retrying a CUDA OOM just burns the window twice and delays the error.
    """
    fake, calls = _streamer(RuntimeError("kokoro worker: RuntimeError: CUDA OOM"))
    with patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", fake):
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await _drain(kokoro_client.KokoroWorkerBackend())

    assert calls["n"] == 1, "a worker-side error must not be retried"


@pytest.mark.asyncio
async def test_a_protocol_error_is_NOT_retried():
    """The peer is not speaking this protocol; retrying cannot help."""
    from api.src.kokoro_worker.codec import KokoroWorkerProtocolError

    fake, calls = _streamer(KokoroWorkerProtocolError("bad frame sync"))
    with patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", fake):
        with pytest.raises(KokoroWorkerProtocolError):
            await _drain(kokoro_client.KokoroWorkerBackend())

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_retries_can_be_disabled():
    """A knob, so an operator can turn this off without a redeploy of logic."""
    fake, calls = _streamer(kokoro_client.KokoroWorkerUnavailable("refused"))
    with (
        patch.object(kokoro_client, "_RESPAWN_RETRIES", 0),
        patch.object(kokoro_client.KokoroWorkerBackend, "_stream_once", fake),
    ):
        with pytest.raises(kokoro_client.KokoroWorkerUnavailable):
            await _drain(kokoro_client.KokoroWorkerBackend())

    assert calls["n"] == 1
