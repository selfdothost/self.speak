"""Unload must gc.collect() BEFORE empty_cache().

`torch.cuda.empty_cache()` returns only blocks the caching allocator already
considers free. An nn.Module graph is full of reference cycles, so dropping the
last name does NOT collect it — CPython's refcounting cannot break a cycle, and
the cyclic collector has to run first.

Get the order wrong (or omit gc entirely) and the cache is emptied while every
tensor is still alive, so nothing returns to the driver. That is not
theoretical: self.speak's chatterbox worker sat at 212 MiB reserved with
`resident: false` for 41 hours, and answered `freed_bytes: 0` to a VRAM-lease
release that had asked for memory. Demonstrated on the deployed torch build:

    drop + empty_cache   reserved 12 MiB -> 12 MiB   (nothing freed)
    gc   + empty_cache   reserved 12 MiB ->  0 MiB

These tests pin the ORDER, not the mechanism — they assert gc.collect() is
called and that it happens before empty_cache(), because a future refactor that
"tidies" the call into the wrong position reintroduces a silent leak that only
shows up as a broker release returning zero.
"""

from unittest.mock import MagicMock, patch

import pytest


def _order_probe():
    """Records the order of gc.collect / empty_cache calls."""
    calls = []
    return calls


class TestChatterboxWorkerUnload:
    def test_collects_before_emptying_the_cache(self):
        from api.src.chatterbox_worker.engine import ChatterboxEngine

        eng = ChatterboxEngine.__new__(ChatterboxEngine)
        eng._model = MagicMock()
        eng._sr = 24000

        calls = _order_probe()
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.empty_cache.side_effect = lambda: calls.append("empty_cache")
        fake_torch.cuda.synchronize.side_effect = lambda: calls.append("synchronize")

        with patch.dict("sys.modules", {"torch": fake_torch}), patch(
            "api.src.chatterbox_worker.engine.gc.collect",
            side_effect=lambda: calls.append("gc"),
        ):
            eng.unload()

        assert "gc" in calls, "unload() must gc.collect(); empty_cache() alone frees nothing cyclic"
        assert calls.index("gc") < calls.index("empty_cache"), (
            f"gc.collect() must run BEFORE empty_cache(); got {calls}"
        )
        assert eng._model is None

    def test_survives_a_torch_failure(self):
        """A failed unload is measured by the before/after delta, never raised —
        the release path must still answer the broker."""
        from api.src.chatterbox_worker.engine import ChatterboxEngine

        eng = ChatterboxEngine.__new__(ChatterboxEngine)
        eng._model = MagicMock()
        eng._sr = 24000

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.side_effect = RuntimeError("driver gone")

        with patch.dict("sys.modules", {"torch": fake_torch}):
            eng.unload()  # must not raise

        assert eng._model is None


class TestKokoroUnload:
    def test_collects_before_emptying_the_cache(self):
        pytest.importorskip("api.src.inference.kokoro_v1")
        from api.src.inference import kokoro_v1

        be = kokoro_v1.KokoroV1.__new__(kokoro_v1.KokoroV1)
        be._model = MagicMock()
        be._pipelines = {"a": MagicMock(), "e": MagicMock()}

        calls = _order_probe()
        with patch.object(kokoro_v1.torch.cuda, "is_available", return_value=True), patch.object(
            kokoro_v1.torch.cuda, "empty_cache", side_effect=lambda: calls.append("empty_cache")
        ), patch.object(
            kokoro_v1.torch.cuda, "synchronize", side_effect=lambda: calls.append("sync")
        ), patch.object(
            kokoro_v1.gc, "collect", side_effect=lambda: calls.append("gc")
        ):
            be.unload()

        assert "gc" in calls
        assert calls.index("gc") < calls.index("empty_cache"), (
            f"gc.collect() must run BEFORE empty_cache(); got {calls}"
        )
        assert be._model is None
        assert be._pipelines == {}
