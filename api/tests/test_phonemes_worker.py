"""Phoneme synthesis survives the worker boundary (self.speak#7).

The route died under the P3 cutover because `TTSService` reached into
`backend._get_pipeline(...).generate_from_tokens(...)` — a Kokoro internal,
which cannot cross a process boundary. The fix is not a special case for the
worker; it is making phoneme synthesis part of the backend CONTRACT so the
caller stops caring where the model lives.

Same integration shape as `test_kokoro_worker_integration`: the real client
against the real worker app over an in-process ASGI transport, because the last
two bugs in this epic both lived in seams that only one side's tests covered.
"""

import numpy as np
import pytest

pytest.importorskip("httpx")
import httpx  # noqa: E402

from api.src.core.config import settings  # noqa: E402
from api.src.inference import kokoro_client  # noqa: E402


class _FakeBackend:
    """Stands in for KokoroV1 inside the worker."""

    def __init__(self, audio=None, error=None):
        self._audio = audio
        self._error = error
        self.calls = []

    async def generate_from_phonemes(self, phonemes, voice_path, speed=1.0, lang_code="a"):
        self.calls.append((phonemes, voice_path, speed, lang_code))
        if self._error:
            raise self._error
        return self._audio


@pytest.fixture
def wired(monkeypatch):
    from api.src.kokoro_worker import main as worker

    def _install(backend):
        class _Manager:
            def get_backend(self):
                return backend

        async def _engine():
            return _Manager()

        monkeypatch.setattr(worker, "_engine", _engine)

        def _make_client(timeout):
            return httpx.AsyncClient(
                transport=httpx.ASGITransport(app=worker.app),
                base_url=settings.kokoro_worker_url,
                timeout=timeout,
            )

        monkeypatch.setattr(kokoro_client, "_make_client", _make_client)
        return backend

    return _install


@pytest.mark.asyncio
async def test_phonemes_round_trip_across_the_boundary(wired):
    """The regression: this route must work with the model in the worker."""
    original = np.linspace(-0.5, 0.5, 2048, dtype=np.float32)
    backend = wired(_FakeBackend(audio=original.copy()))
    client_backend = kokoro_client.KokoroWorkerBackend()

    got = await client_backend.generate_from_phonemes("hh ax l ow", "/v.pt", 1.0, "a")

    np.testing.assert_array_equal(got, original)
    assert backend.calls == [("hh ax l ow", "/v.pt", 1.0, "a")]


@pytest.mark.asyncio
async def test_speed_and_lang_code_survive(wired):
    backend = wired(_FakeBackend(audio=np.zeros(8, dtype=np.float32)))
    client_backend = kokoro_client.KokoroWorkerBackend()

    await client_backend.generate_from_phonemes("p", "/v.pt", 1.5, "b")

    assert backend.calls[0][2] == 1.5
    assert backend.calls[0][3] == "b"


@pytest.mark.asyncio
async def test_returned_audio_is_writable(wired):
    """Callers scale audio in place; frombuffer over bytes is read-only."""
    wired(_FakeBackend(audio=np.ones(16, dtype=np.float32)))
    client_backend = kokoro_client.KokoroWorkerBackend()

    got = await client_backend.generate_from_phonemes("p", "/v.pt")
    got *= 2.0  # must not raise


@pytest.mark.asyncio
async def test_a_backend_failure_reaches_the_caller(wired):
    """Never a silent empty result — that is the self.speak#6 failure shape."""
    wired(_FakeBackend(error=RuntimeError("CUDA OOM")))
    client_backend = kokoro_client.KokoroWorkerBackend()

    with pytest.raises(kokoro_client.KokoroWorkerUnavailable, match="500"):
        await client_backend.generate_from_phonemes("p", "/v.pt")


@pytest.mark.asyncio
async def test_empty_phonemes_are_rejected(wired):
    wired(_FakeBackend(audio=np.zeros(4, dtype=np.float32)))
    client_backend = kokoro_client.KokoroWorkerBackend()

    with pytest.raises(kokoro_client.KokoroWorkerUnavailable, match="400"):
        await client_backend.generate_from_phonemes("   ", "/v.pt")


@pytest.mark.asyncio
async def test_a_missing_voice_path_is_rejected(wired):
    """The path is what makes a combined voice reproducible (self.speak!31)."""
    wired(_FakeBackend(audio=np.zeros(4, dtype=np.float32)))
    client_backend = kokoro_client.KokoroWorkerBackend()

    with pytest.raises(kokoro_client.KokoroWorkerUnavailable, match="400"):
        await client_backend.generate_from_phonemes("p", "")


def test_both_backends_declare_the_capability():
    """A contract only one side implements is not a contract."""
    from api.src.inference.kokoro_client import KokoroWorkerBackend
    from api.src.inference.kokoro_v1 import KokoroV1

    assert hasattr(KokoroV1, "generate_from_phonemes")
    assert hasattr(KokoroWorkerBackend, "generate_from_phonemes")
