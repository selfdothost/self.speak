"""self.speak#6 — a synthesis that loses audio must not end cleanly.

Before this, a generation failure mid-stream produced HTTP 200 with a valid,
playable, silently SHORTER audio file. The chain was: model_manager flattened
the error to RuntimeError, _process_chunk logged it without re-raising (and it
is an async generator, so returning cleanly ended the consumer's `async for`
normally), chunk_index still advanced so finalization ran, and a correct
container trailer was written over incomplete audio.

These tests pin both halves of the fix: drops are RECORDED, and a recorded drop
prevents the stream from finalizing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from api.src.services.tts_service import TTSService, _note_dropped_audio


@pytest.fixture
def service():
    """A TTSService with managers stubbed out — no model, no voices, no GPU.

    model_manager is a MagicMock with only `ensure_loaded` made async, NOT a
    bare AsyncMock: on an AsyncMock every child is async too, so the real code's
    synchronous `get_backend()` would hand back an un-awaited coroutine and warn
    instead of a backend.
    """
    svc = TTSService("test_output")
    svc.model_manager = MagicMock()
    svc.model_manager.ensure_loaded = AsyncMock()
    svc.model_manager.get_backend.return_value = MagicMock()
    svc._voice_manager = MagicMock()
    return svc


def _smart_split_yielding(*chunks):
    """Stand in for smart_split, yielding (chunk_text, tokens, pause) triples."""

    async def _fake(text, **kwargs):
        for chunk_text in chunks:
            yield chunk_text, [1, 2, 3], None

    return _fake


def _process_chunk_that_drops(*, failing_texts):
    """A _process_chunk that produces no audio for `failing_texts`.

    Mirrors what the real one does on failure: record the drop, yield nothing,
    and return normally — which is precisely what used to be invisible.
    """

    async def _fake(self, chunk_text, tokens, *args, failures=None, **kwargs):
        if kwargs.get("is_last"):
            yield MagicMock(output=b"TRAILER", audio=np.zeros(1, dtype=np.int16))
            return
        if chunk_text in failing_texts:
            _note_dropped_audio(failures, f"Failed to process tokens: boom [{chunk_text}]")
            return
        yield MagicMock(
            output=b"AUDIO", audio=np.zeros(240, dtype=np.int16), word_timestamps=None
        )

    return _fake


async def _drain(service, text, sink=None):
    """Collect everything the stream yields for `text` into `sink`.

    The sink is passed IN rather than returned, because the interesting cases
    raise part-way through and a returned list would be lost — which would make
    "no trailer was emitted" pass vacuously.
    """
    out = sink if sink is not None else []
    writer = MagicMock()
    with patch.object(
        TTSService, "_get_voices_path", AsyncMock(return_value=("af", "/v.pt"))
    ):
        async for chunk in service._generate_audio_stream_impl(
            text, "af_voice", writer, output_format="mp3"
        ):
            out.append(chunk)
    return out


@pytest.mark.asyncio
async def test_dropped_chunk_aborts_the_stream(service):
    """The regression: a failed chunk must not yield a finalized stream."""
    with (
        patch(
            "api.src.services.tts_service.smart_split",
            _smart_split_yielding("good one", "bad one"),
        ),
        patch.object(
            TTSService, "_process_chunk", _process_chunk_that_drops(failing_texts={"bad one"})
        ),
    ):
        with pytest.raises(RuntimeError, match="Synthesis incomplete"):
            await _drain(service, "good one bad one")


@pytest.mark.asyncio
async def test_no_trailer_is_written_when_audio_was_lost(service):
    """The abort must happen BEFORE finalization.

    A trailer written over short audio is exactly the playable-but-wrong file
    #6 was about, so it is not enough that the stream raises — it must raise
    without ever emitting one.
    """
    emitted = []
    with (
        patch(
            "api.src.services.tts_service.smart_split",
            _smart_split_yielding("good one", "bad one"),
        ),
        patch.object(
            TTSService, "_process_chunk", _process_chunk_that_drops(failing_texts={"bad one"})
        ),
    ):
        with pytest.raises(RuntimeError):
            await _drain(service, "good one bad one", sink=emitted)

    assert emitted, "expected the good chunk to still have been delivered"
    assert all(getattr(c, "output", None) != b"TRAILER" for c in emitted)


@pytest.mark.asyncio
async def test_clean_synthesis_still_finalizes(service):
    """The other half: nothing dropped must still finish normally.

    Guards against the fix turning every synthesis into an abort.
    """
    with (
        patch(
            "api.src.services.tts_service.smart_split",
            _smart_split_yielding("good one", "also good"),
        ),
        patch.object(TTSService, "_process_chunk", _process_chunk_that_drops(failing_texts=set())),
    ):
        out = await _drain(service, "good one also good")

    assert out, "a clean synthesis yielded nothing"
    assert any(getattr(c, "output", None) == b"TRAILER" for c in out), (
        "a clean synthesis must still be finalized"
    )


def test_note_dropped_audio_tolerates_no_collector():
    """`failures=None` keeps the old log-and-continue behaviour."""
    _note_dropped_audio(None, "nothing collects this")


def test_note_dropped_audio_records():
    failures = []
    _note_dropped_audio(failures, "first")
    _note_dropped_audio(failures, "second")
    assert failures == ["first", "second"]
