"""P4 release-ordering tests (self.speak#5).

P3 already called the workers in order -- but called them BOTH, unconditionally,
each toward the full target. So ordinary TTS got cold-started even when the
optional engine's yield would have covered the request on its own. Escalation is
what makes the ordering mean anything, and these tests pin the behaviour that
distinguishes the two: **an engine that did not need to be disturbed is not
disturbed.**
"""

from unittest.mock import AsyncMock, patch

import pytest

from api.src.core.config import settings
from api.src.inference import vram_lease


@pytest.fixture
def both_engines():
    """Both workers part of the footprint, as on the deployed GPU pod."""
    before = (settings.chatterbox_enabled, settings.kokoro_worker_enabled)
    settings.chatterbox_enabled = True
    settings.kokoro_worker_enabled = True
    yield
    settings.chatterbox_enabled, settings.kokoro_worker_enabled = before


def _freed_sequence(*values):
    """Stand in for the live aggregated probe, returning each value in turn.

    Values are bytes-freed-so-far; ``None`` means unmeasurable.
    """
    seq = list(values)

    async def _fake(before):
        return seq.pop(0) if seq else seq_last[0]

    seq_last = [values[-1]]
    return _fake


GB = 1 << 30


@pytest.mark.asyncio
async def test_kokoro_is_untouched_when_the_local_unload_sufficed(both_engines):
    """Nothing is disturbed if main's own unload already met the target."""
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(4 * GB)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)

    cb.assert_not_called()
    kk.assert_not_called()


@pytest.mark.asyncio
async def test_kokoro_is_untouched_when_chatterbox_sufficed(both_engines):
    """The whole point of P4.

    Chatterbox yields, the target is met, and the PRIMARY engine serving
    assistant devices is left alone. Under P3 it would have been asked anyway
    and paid a CUDA init plus a model load on the next request for nothing.
    """
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(0, 3 * GB)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)

    cb.assert_called_once()
    kk.assert_not_called()


@pytest.mark.asyncio
async def test_kokoro_is_asked_when_chatterbox_was_not_enough(both_engines):
    """Escalation must actually happen, or a priority-10 consumer starves."""
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(0, 1 * GB, 1 * GB)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 5 * GB, 5.0)

    cb.assert_called_once()
    kk.assert_called_once()


@pytest.mark.asyncio
async def test_escalation_asks_only_for_the_REMAINING_deficit(both_engines):
    """Asking each engine for the full target would over-free.

    Chatterbox already gave 1 GiB toward a 5 GiB ask, so Kokoro should be asked
    for 4, not 5 -- otherwise engines yield more than the broker requested and
    the extra cold starts buy nothing.
    """
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(0, 1 * GB, 1 * GB)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 5 * GB, 5.0)

    assert kk.call_args.args[0] == 4 * GB
    assert cb.call_args.args[0] == 5 * GB  # nothing freed yet at that point


@pytest.mark.asyncio
async def test_ordering_is_chatterbox_before_kokoro(both_engines):
    """Cheapest-to-yield first, and the order must be observable, not implied."""
    calls = []
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(0, 0, 0)),
        patch.object(
            vram_lease,
            "_release_chatterbox_worker",
            AsyncMock(side_effect=lambda *a, **k: calls.append("chatterbox")),
        ),
        patch.object(
            vram_lease,
            "_release_kokoro_worker",
            AsyncMock(side_effect=lambda *a, **k: calls.append("kokoro")),
        ),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 9 * GB, 5.0)

    assert calls == ["chatterbox", "kokoro"]


@pytest.mark.asyncio
async def test_unmeasurable_escalates_rather_than_assuming_success(both_engines):
    """Unknown is NOT met.

    Under-delivering VRAM to a priority-10 consumer means an OOM in the
    inference brain everything depends on; an unnecessary escalation costs one
    cold start. The asymmetry decides the tie.
    """
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(None, None, None)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)

    cb.assert_called_once()
    kk.assert_called_once()
    # Unmeasurable → ask for the whole target, never a discounted remainder.
    assert kk.call_args.args[0] == 2 * GB


@pytest.mark.asyncio
async def test_force_is_forwarded_to_both(both_engines):
    """force is what authorises a worker to EXIT and return its CUDA context."""
    cb, kk = AsyncMock(), AsyncMock()
    with (
        patch.object(vram_lease, "_freed_since", _freed_sequence(0, 0, 0)),
        patch.object(vram_lease, "_release_chatterbox_worker", cb),
        patch.object(vram_lease, "_release_kokoro_worker", kk),
    ):
        await vram_lease._escalating_worker_release(10 * GB, 9 * GB, 5.0, True)

    assert cb.call_args.args[2] is True
    assert kk.call_args.args[2] is True


@pytest.mark.asyncio
async def test_a_single_engine_is_asked_directly_with_no_probing():
    """One engine means no ordering decision, so nothing to measure.

    This is not an optimisation -- it is the absence of a decision. It also
    keeps the single-engine path (every deploy today) making exactly the same
    calls it made before escalation existed. Getting this wrong broke five
    existing tests with `coroutine raised StopIteration`, because they mock the
    probe with a finite side_effect list and the extra probes exhausted it.
    """
    before_flags = (settings.chatterbox_enabled, settings.kokoro_worker_enabled)
    settings.chatterbox_enabled = True
    settings.kokoro_worker_enabled = False
    probe, cb, kk = AsyncMock(), AsyncMock(), AsyncMock()
    try:
        with (
            patch.object(vram_lease, "_freed_since", probe),
            patch.object(vram_lease, "_release_chatterbox_worker", cb),
            patch.object(vram_lease, "_release_kokoro_worker", kk),
        ):
            await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)
    finally:
        settings.chatterbox_enabled, settings.kokoro_worker_enabled = before_flags

    probe.assert_not_called()
    cb.assert_called_once()
    assert cb.call_args.args[0] == 2 * GB  # the full target, not a remainder
    kk.assert_not_called()


@pytest.mark.asyncio
async def test_no_engines_enabled_does_nothing_at_all():
    """A Kokoro-in-process deploy must be untouched by any of this."""
    before_flags = (settings.chatterbox_enabled, settings.kokoro_worker_enabled)
    settings.chatterbox_enabled = False
    settings.kokoro_worker_enabled = False
    probe, cb, kk = AsyncMock(), AsyncMock(), AsyncMock()
    try:
        with (
            patch.object(vram_lease, "_freed_since", probe),
            patch.object(vram_lease, "_release_chatterbox_worker", cb),
            patch.object(vram_lease, "_release_kokoro_worker", kk),
        ):
            await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)
    finally:
        settings.chatterbox_enabled, settings.kokoro_worker_enabled = before_flags

    probe.assert_not_called()
    cb.assert_not_called()
    kk.assert_not_called()


@pytest.mark.asyncio
async def test_a_disabled_engine_is_skipped_without_being_probed():
    """A Kokoro-only deploy must never call the chatterbox leg, and vice versa."""
    before = (settings.chatterbox_enabled, settings.kokoro_worker_enabled)
    settings.chatterbox_enabled = False
    settings.kokoro_worker_enabled = True
    cb, kk = AsyncMock(), AsyncMock()
    try:
        with (
            patch.object(vram_lease, "_freed_since", _freed_sequence(0, 0)),
            patch.object(vram_lease, "_release_chatterbox_worker", cb),
            patch.object(vram_lease, "_release_kokoro_worker", kk),
        ):
            await vram_lease._escalating_worker_release(10 * GB, 2 * GB, 5.0)
    finally:
        settings.chatterbox_enabled, settings.kokoro_worker_enabled = before

    cb.assert_not_called()
    kk.assert_called_once()
