"""The client<->worker seam, exercised end to end (self.speak#5).

**This file exists because of a bug that a green suite could not have caught.**

P2 tested the worker. P3 tested the client. Neither crossed between them, and
the very first thing that boundary does is interpret the voice argument — which
it got wrong in a way that would have failed EVERY synthesis the moment the
cutover was enabled. Eight P3 tests passed while it sat there.

So these tests drive the REAL client against the REAL worker app over an
in-process ASGI transport. Everything between them is genuine: pydantic request
validation, HTTP, chunked transfer, the P1 frame encoder and decoder, the
stream bounds and the terminator. Only the engine is faked, because it is the
one piece that needs a GPU.
"""

import numpy as np
import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

from api.src.core.config import settings  # noqa: E402
from api.src.inference import kokoro_client  # noqa: E402
from api.src.inference.base import AudioChunk  # noqa: E402
from api.src.structures.schemas import WordTimestamp  # noqa: E402


class _Chunk:
    def __init__(self, audio, word_timestamps=None):
        self.audio = audio
        self.word_timestamps = word_timestamps


class _RecordingManager:
    """Fake engine that records exactly what the worker asked it to generate."""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []

    def generate(self, text, voice, **kwargs):
        self.calls.append({"text": text, "voice": voice, **kwargs})

        async def _gen():
            for c in self._chunks:
                yield c

        return _gen()


@pytest.fixture
def wired(monkeypatch):
    """Point the real client at the real worker app, in process.

    Returns a helper that installs a fake engine and yields the manager so a
    test can assert on what actually crossed the wire.
    """
    from api.src.kokoro_worker import main as worker

    def _install(chunks_or_manager):
        manager = (
            chunks_or_manager
            if hasattr(chunks_or_manager, "generate")
            else _RecordingManager(chunks_or_manager)
        )

        async def _engine():
            return manager

        monkeypatch.setattr(worker, "_engine", _engine)

        def _make_client(timeout):
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=worker.app),
                base_url=settings.kokoro_worker_url,
                timeout=timeout,
            )

        monkeypatch.setattr(kokoro_client, "_make_client", _make_client)
        return manager

    return _install


def _audio(n=64, value=0.3):
    return np.full(n, value, dtype=np.float32)


# --- the bug this file was written for --------------------------------------


@pytest.mark.asyncio
async def test_the_resolved_voice_path_crosses_the_boundary(wired):
    """The regression.

    TTSService calls generate(text, (name, path)). If only the name crosses,
    the worker hands a bare string to KokoroV1.generate -- which reads a bare
    string as a filesystem PATH, not a name -- and every synthesis fails.
    """
    manager = wired([_Chunk(_audio())])
    backend = kokoro_client.KokoroWorkerBackend()

    got = [c async for c in backend.generate("hello", ("af_heart", "/voices/af_heart.pt"))]

    assert got, "no audio crossed the boundary"
    assert manager.calls[0]["voice"] == ("af_heart", "/voices/af_heart.pt")


@pytest.mark.asyncio
async def test_a_combined_voice_keeps_its_blended_tensor_path(wired):
    """Main blends the tensors and writes a temp file; the name cannot rebuild it."""
    manager = wired([_Chunk(_audio())])
    backend = kokoro_client.KokoroWorkerBackend()
    blended = "/tmp/af_jadzia+af_jessica.pt"

    [c async for c in backend.generate("hi", ("af_jadzia+af_jessica", blended))]

    assert manager.calls[0]["voice"] == ("af_jadzia+af_jessica", blended)


@pytest.mark.asyncio
async def test_a_bare_voice_name_still_works(wired):
    """Callers that pass a plain name must not be broken by the tuple handling."""
    manager = wired([_Chunk(_audio())])
    backend = kokoro_client.KokoroWorkerBackend()

    [c async for c in backend.generate("hi", "af_heart")]

    assert manager.calls[0]["voice"] == "af_heart"


# --- the round trip ---------------------------------------------------------


@pytest.mark.asyncio
async def test_audio_survives_the_boundary_bit_for_bit(wired):
    """Acceptance criterion: output unchanged across the IPC boundary."""
    original = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    wired([_Chunk(original.copy())])
    backend = kokoro_client.KokoroWorkerBackend()

    got = [c async for c in backend.generate("hi", ("af_heart", "/v.pt"))]

    np.testing.assert_array_equal(np.concatenate([c.audio for c in got]), original)


@pytest.mark.asyncio
async def test_word_timestamps_survive_the_boundary(wired):
    """Acceptance criterion: word timestamps identical across the boundary."""
    ts = [
        WordTimestamp(word="hello", start_time=0.0, end_time=0.42),
        WordTimestamp(word="there", start_time=0.42, end_time=0.90),
    ]
    wired([_Chunk(_audio(), word_timestamps=ts)])
    backend = kokoro_client.KokoroWorkerBackend()

    got = [
        c
        async for c in backend.generate(
            "hello there", ("af_heart", "/v.pt"), return_timestamps=True
        )
    ]

    out = got[0].word_timestamps
    assert [(t.word, t.start_time, t.end_time) for t in out] == [
        (t.word, t.start_time, t.end_time) for t in ts
    ]


@pytest.mark.asyncio
async def test_multiple_chunks_arrive_in_order(wired):
    wired([_Chunk(np.full(32, i / 10, dtype=np.float32)) for i in range(1, 6)])
    backend = kokoro_client.KokoroWorkerBackend()

    got = [c async for c in backend.generate("hi", ("af_heart", "/v.pt"))]

    assert len(got) == 5
    assert [round(float(c.audio[0]), 2) for c in got] == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.asyncio
async def test_returned_chunks_are_AudioChunks_the_parent_can_mutate(wired):
    """TTSService multiplies audio in place and mutates timestamp attributes."""
    ts = [WordTimestamp(word="x", start_time=0.1, end_time=0.2)]
    wired([_Chunk(_audio(), word_timestamps=ts)])
    backend = kokoro_client.KokoroWorkerBackend()

    got = [
        c
        async for c in backend.generate(
            "x", ("af_heart", "/v.pt"), return_timestamps=True
        )
    ]

    chunk = got[0]
    assert isinstance(chunk, AudioChunk)
    chunk.audio *= 1.5  # must not raise on a read-only array
    chunk.word_timestamps[0].start_time += 1.0  # must not raise on a dict


# --- failures across the boundary -------------------------------------------


@pytest.mark.asyncio
async def test_a_worker_side_failure_reaches_the_caller(wired):
    """An ERROR frame is a real answer and must not be silently swallowed.

    The whole reason self.speak#6 existed is that a failed generation looked
    like a complete one. Across a process boundary that is even easier to get
    wrong, so it is pinned here.
    """

    class _Exploding:
        def generate(self, text, voice, **kwargs):
            async def _gen():
                yield _Chunk(_audio())
                raise RuntimeError("CUDA OOM")

            return _gen()

    wired(_Exploding())
    backend = kokoro_client.KokoroWorkerBackend()

    with pytest.raises(RuntimeError, match="CUDA OOM"):
        [c async for c in backend.generate("hi", ("af_heart", "/v.pt"))]


@pytest.mark.asyncio
async def test_empty_text_is_rejected_with_a_clean_error(wired):
    """A 400 from the worker must surface as unavailable, not a protocol error."""
    wired([_Chunk(_audio())])
    backend = kokoro_client.KokoroWorkerBackend()

    with pytest.raises(kokoro_client.KokoroWorkerUnavailable):
        [c async for c in backend.generate("   ", ("af_heart", "/v.pt"))]
