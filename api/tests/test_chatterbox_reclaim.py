"""Stepping aside: the chatterbox worker returning its CUDA primary context.

`empty_cache()` returns the caching allocator's blocks and never the context
(~470 MiB on the deployed 4090). Only process exit does. So a forced release
that unloading cannot satisfy has one honest answer left.

These tests pin the decisions rather than the mechanism, because the action is
irreversible: exiting on a guess costs a live worker, and the entrypoint
respawn loop makes that cheap but not free.
"""

import sys
import types

import pytest

from api.src.chatterbox_worker import reclaim


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(reclaim, "_exit_scheduled", False)


def _torch(reserved=0, raises=False, initialized=True):
    """`initialized` is settable and deliberately IGNORED by context_held()."""

    class _Cuda:
        def memory_reserved(self, *a):
            if raises:
                raise RuntimeError("driver gone")
            return reserved

        def is_initialized(self):
            return initialized

    m = types.ModuleType("torch")
    m.cuda = _Cuda()
    return m


class TestContextHeld:
    def test_true_once_the_process_has_allocated(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", _torch(reserved=470 * 1024 * 1024))
        assert reclaim.context_held() is True

    def test_startup_alone_is_not_a_context(self, monkeypatch):
        """is_initialized() is True after a mere device-properties query. Using
        it as this test cost self.sketch hours of restarting every 15 minutes to
        reclaim memory it did not hold. Not repeated here."""
        monkeypatch.setitem(sys.modules, "torch", _torch(reserved=0, initialized=True))
        assert reclaim.context_held() is False

    def test_a_raising_probe_is_false(self, monkeypatch):
        """Unknown must never authorise an exit."""
        monkeypatch.setitem(sys.modules, "torch", _torch(raises=True))
        assert reclaim.context_held() is False

    def test_missing_torch_is_false(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        assert reclaim.context_held() is False


class TestScheduleExit:
    def test_is_idempotent(self, monkeypatch):
        monkeypatch.setattr(reclaim.os, "_exit", lambda code: None)
        assert reclaim.schedule_exit("first") is True
        assert reclaim.schedule_exit("second") is False

    def test_exits_immediately_without_an_event_loop(self, monkeypatch):
        """The grace period exists to flush the HTTP response. With no loop, a
        dropped response beats a context held forever against a consumer that
        asked for it."""
        seen = {}
        monkeypatch.setattr(reclaim.os, "_exit", lambda code: seen.setdefault("code", code))

        def _no_loop():
            raise RuntimeError("no running event loop")

        monkeypatch.setattr(reclaim.asyncio, "get_event_loop", _no_loop)
        reclaim.schedule_exit("no loop")
        assert seen["code"] == 0
