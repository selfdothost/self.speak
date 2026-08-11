"""P3 cutover tests (self.speak#5).

The acceptance criterion for P3 is a NEGATIVE: main must not hold a CUDA
context. The decisive check is `nvidia-smi --query-compute-apps` on a live pod
and cannot be asserted here — self.sketch is the cautionary tale, where unit
tests and import-cost measurements all passed while 386 MiB sat untouched.

So these tests pin the things that are checkable in process, and that are the
mechanisms by which the negative holds or fails:

  * main does not IMPORT kokoro when the worker owns the model
  * the backend is selected by capability, not by concrete class
  * a worker-held model still shows up in the aggregated VRAM answer
  * unreachable is reported as unknown, never as zero
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.core.config import settings


@pytest.fixture
def worker_mode():
    """Turn on the cutover for the duration of a test."""
    before = settings.kokoro_worker_enabled
    settings.kokoro_worker_enabled = True
    yield
    settings.kokoro_worker_enabled = before


# --- the point of the whole exercise ---------------------------------------


@pytest.mark.asyncio
async def test_worker_mode_does_not_import_kokoro(worker_mode):
    """main must never import kokoro when the worker owns the model.

    Not a style preference. main is the pod's sole liveness path, so it can
    never exit, so any CUDA primary context it creates (~470 MiB on the 4090) is
    unreclaimable forever. Never importing the thing that would allocate is the
    cheapest way to be certain it does not.
    """
    from api.src.inference.model_manager import ModelManager

    # Snapshot and RESTORE the module objects rather than just deleting them.
    # Deleting outright poisons the whole session: a later re-import builds a
    # NEW class object, so other modules' references and any
    # patch("...kokoro_v1.KPipeline") target a different class than the code
    # under test uses. That broke four unrelated tests the first time round.
    # Restoring the identical objects keeps identity intact.
    saved = {m: sys.modules[m] for m in list(sys.modules) if "kokoro_v1" in m}
    for m in saved:
        del sys.modules[m]
    try:
        manager = ModelManager()
        await manager.initialize()

        assert not any("kokoro_v1" in m for m in sys.modules), (
            "main imported kokoro_v1 despite the worker owning the model"
        )
    finally:
        sys.modules.update(saved)


@pytest.mark.asyncio
async def test_worker_mode_selects_the_worker_backend(worker_mode):
    from api.src.inference.kokoro_client import KokoroWorkerBackend
    from api.src.inference.model_manager import ModelManager

    manager = ModelManager()
    await manager.initialize()
    assert isinstance(manager.get_backend(), KokoroWorkerBackend)


@pytest.mark.asyncio
async def test_the_worker_backend_declares_the_streaming_capability():
    """TTSService branches on this instead of `isinstance(backend, KokoroV1)`.

    With the old isinstance check a worker-backed backend fell through to the
    legacy tokens-in/one-blob-out branch -- wrong audio rather than an error.
    """
    from api.src.inference.kokoro_client import KokoroWorkerBackend

    assert KokoroWorkerBackend.streams_text is True


def test_the_in_process_backend_declares_it_too():
    """Both sides must answer the same question, or the branch is a lie."""
    from api.src.inference.kokoro_v1 import KokoroV1

    assert KokoroV1.streams_text is True


# --- VRAM accounting --------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_held_vram_is_aggregated(worker_mode):
    """A worker-held model must appear in self.speak's single held figure.

    Under the cutover the LOCAL leg holds ~nothing by design, so reporting only
    the local leg would tell the broker self.speak holds almost no VRAM while a
    whole model sits in the worker -- an over-grant of the shared 4090.
    """
    from api.src.inference import vram_lease

    local = {
        "status": "ok",
        "held_vram_bytes": 0,
        "model_resident": False,
        "total_capacity_bytes": 24 << 30,
    }
    with (
        patch.object(vram_lease, "probe_vram_state", return_value=local),
        patch(
            "api.src.inference.kokoro_client.probe_state",
            AsyncMock(return_value={"held_bytes": 3_000_000_000, "resident": True}),
        ),
    ):
        got = await vram_lease.probe_vram_state_full()

    assert got["held_vram_bytes"] == 3_000_000_000
    assert got["model_resident"] is True


@pytest.mark.asyncio
async def test_an_unreachable_worker_is_unknown_not_zero(worker_mode):
    """Reporting 0 for an unreadable worker over-grants the card (self.ai#74)."""
    from api.src.inference import vram_lease

    local = {
        "status": "ok",
        "held_vram_bytes": 0,
        "model_resident": False,
        "total_capacity_bytes": 24 << 30,
    }
    with (
        patch.object(vram_lease, "probe_vram_state", return_value=local),
        patch(
            "api.src.inference.kokoro_client.probe_state", AsyncMock(return_value=None)
        ),
    ):
        got = await vram_lease.probe_vram_state_full()

    assert got["held_vram_bytes"] is None, "an unreachable worker reported as a real 0"


@pytest.mark.asyncio
async def test_disabled_cutover_does_not_query_the_worker():
    """The pre-deploy window must be untouched: no probe, no perturbation."""
    from api.src.inference import vram_lease

    local = {
        "status": "ok",
        "held_vram_bytes": 123,
        "model_resident": True,
        "total_capacity_bytes": 24 << 30,
    }
    probe = AsyncMock(return_value=None)
    with (
        patch.object(vram_lease, "probe_vram_state", return_value=local),
        patch("api.src.inference.kokoro_client.probe_state", probe),
    ):
        got = await vram_lease.probe_vram_state_full()

    assert got["held_vram_bytes"] == 123
    probe.assert_not_called()


# --- the scope limit, made explicit ----------------------------------------


@pytest.mark.asyncio
async def test_the_phoneme_path_works_through_the_worker(worker_mode):
    """REPLACES an earlier test that asserted this path REFUSES.

    Under P3 it did refuse -- deliberately, because TTSService reached into
    backend._get_pipeline(), a Kokoro internal with no worker endpoint, and
    falling through would have produced wrong audio instead of an error.

    self.speak#7 made phoneme synthesis part of the backend CONTRACT instead, so
    the refusal is gone and the route works wherever the model lives. The old
    test failing is the correct outcome of that change, not a regression -- it
    was pinning behaviour we intended to remove.
    """
    import numpy as np

    from api.src.services.tts_service import TTSService

    svc = TTSService("out")

    class _Workerish:
        streams_text = True  # no _get_pipeline: the model is in another process

        async def generate_from_phonemes(self, phonemes, voice_path, speed=1.0, lang_code="a"):
            return np.full(64, 0.25, dtype=np.float32)

    # MagicMock, NOT AsyncMock: on an AsyncMock every child is async too, so the
    # production code's synchronous get_backend() would receive an un-awaited
    # coroutine instead of the backend.
    svc.model_manager = MagicMock()
    svc.model_manager.get_backend.return_value = _Workerish()

    with patch.object(
        TTSService, "_get_voices_path", AsyncMock(return_value=("af", "/v.pt"))
    ):
        audio, elapsed = await svc.generate_from_phonemes("hh ax l ow", "af_heart")

    assert len(audio) == 64
    assert elapsed >= 0


@pytest.mark.asyncio
async def test_a_backend_without_the_capability_still_fails_loudly(worker_mode):
    """Silently degrading is the failure mode worth guarding against.

    A backend that cannot do phonemes must raise, not fall through to a path
    that returns plausible-but-wrong audio.
    """
    from api.src.services.tts_service import TTSService

    svc = TTSService("out")

    class _Incapable:
        streams_text = True  # and no generate_from_phonemes

    svc.model_manager = MagicMock()
    svc.model_manager.get_backend.return_value = _Incapable()

    with patch.object(
        TTSService, "_get_voices_path", AsyncMock(return_value=("af", "/v.pt"))
    ):
        with pytest.raises(ValueError, match="only supported"):
            await svc.generate_from_phonemes("hh ax l ow", "af_heart")
