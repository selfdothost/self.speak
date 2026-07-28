"""Regression: the serving path lazily reloads the backend after a VRAM-lease
release unloaded it — instead of 500'ing "Backend not initialized" forever.

This is the bug the GPU deployment's live verification caught (self.ai!204): a
vram-release unloads Kokoro to free VRAM for a co-resident service, and every
synthesis afterwards failed permanently until pod restart. The vram-lease unit
tests mocked the unload, so serve-after-release was never exercised. These tests
exercise the real ModelManager reload path with a stubbed KokoroV1 backend.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.src.inference.model_manager import ModelManager


def _fake_backend():
    """A KokoroV1 stand-in: async load_model, sync unload (matches the real API)."""
    inst = MagicMock()
    inst.load_model = AsyncMock()
    inst.unload = MagicMock()
    return inst


@pytest.mark.asyncio
async def test_ensure_loaded_reloads_after_release_unload():
    """Cold -> load; then unload (a lease release) -> next request reloads."""
    mgr = ModelManager()
    with patch(
        "api.src.inference.model_manager.KokoroV1", side_effect=lambda: _fake_backend()
    ) as K:
        # Cold start: no backend -> ensure_loaded brings it up.
        assert mgr._backend is None
        await mgr.ensure_loaded()
        assert mgr._backend is not None
        assert K.call_count == 1
        mgr.get_backend()  # does not raise

        # A VRAM-lease release unloads the model to free the card.
        mgr.unload_all()
        assert mgr._backend is None
        with pytest.raises(RuntimeError, match="Backend not initialized"):
            mgr.get_backend()

        # The next synthesis reloads transparently (the actual bug: this used to
        # stay unloaded and 500 forever).
        await mgr.ensure_loaded()
        assert mgr._backend is not None
        assert K.call_count == 2
        mgr.get_backend()  # serves again — no restart needed


@pytest.mark.asyncio
async def test_ensure_loaded_is_noop_when_resident():
    """No redundant reloads while the backend is already resident."""
    mgr = ModelManager()
    with patch(
        "api.src.inference.model_manager.KokoroV1", side_effect=lambda: _fake_backend()
    ) as K:
        await mgr.ensure_loaded()
        assert K.call_count == 1
        await mgr.ensure_loaded()
        await mgr.ensure_loaded()
        assert K.call_count == 1  # still just the one load


@pytest.mark.asyncio
async def test_concurrent_ensure_loaded_reloads_exactly_once():
    """A burst of requests arriving after a release triggers ONE reload, not N
    (the reload lock + double-check)."""
    mgr = ModelManager()
    with patch(
        "api.src.inference.model_manager.KokoroV1", side_effect=lambda: _fake_backend()
    ) as K:
        await asyncio.gather(*[mgr.ensure_loaded() for _ in range(8)])
        assert K.call_count == 1
        assert mgr._backend is not None
