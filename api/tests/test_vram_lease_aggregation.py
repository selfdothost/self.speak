"""Phase-2 cross-process VRAM aggregation tests (INTEGRATION-PLAN-v2.md §2.4).

self.speak runs Kokoro (main process) and Chatterbox (sibling worker) in two
independent CUDA contexts. Neither ``memory_reserved()`` sees the other, so the
only correct ``held_vram_bytes`` is the localhost SUM. These tests pin the
tri-state that keeps that sum honest — and, critically, the self.ai#74 guard:
an EXPECTED-but-unaccountable worker collapses ``held`` to null/``unreachable``
rather than under-reporting it (under-reporting = core over-grants = OOM).

The sibling worker leg is patched (no live worker in CI); the local leg reuses
the same ``torch.cuda`` mocking as ``test_vram_lease.py``. ``chatterbox_enabled``
defaults False, so ``probe_vram_state_full`` reduces to the local leg unless a
test flips it on — which is exactly why the existing single-engine release tests
keep passing unchanged.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import api.src.inference.chatterbox_client as chatterbox_client
import api.src.inference.vram_lease as vram_lease
from api.src.inference.vram_lease import handle_vram_release, probe_vram_state_full

_GB = 1024 * 1024 * 1024


def _local(status, held, resident=True):
    """A local-leg probe result shaped exactly like probe_vram_state() returns."""
    real = status == "ok"
    return {
        "held_vram_bytes": held,
        "total_capacity_bytes": 24 * _GB if real else None,
        "device_used_bytes": 6 * _GB if real else None,
        "device_total_bytes": 24 * _GB if real else None,
        "gpu_reachable": real,
        "status": status,
        "model_resident": resident,
    }


class TestProbeAggregation:
    async def test_disabled_returns_local_verbatim_worker_never_queried(self):
        """chatterbox_enabled=False → local leg only, worker never touched. This
        is what preserves the working single-engine lease pre-deploy."""
        probe_state = AsyncMock()
        with patch.object(vram_lease.settings, "chatterbox_enabled", False), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 5 * _GB)), \
             patch.object(chatterbox_client, "probe_state", probe_state):
            state = await probe_vram_state_full()
        probe_state.assert_not_awaited()
        assert state["held_vram_bytes"] == 5 * _GB
        assert state["status"] == "ok"

    async def test_enabled_sums_worker_held(self):
        """Enabled + both legs known → held is the localhost sum (the point)."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 5 * _GB)), \
             patch.object(chatterbox_client, "probe_state",
                          AsyncMock(return_value={"held_bytes": 3 * _GB, "resident": True})):
            state = await probe_vram_state_full()
        assert state["held_vram_bytes"] == 8 * _GB
        assert state["status"] == "ok"
        assert state["gpu_reachable"] is True
        # Whole-card figures come from the local mem_get_info() (spans BOTH
        # contexts already) and must NOT be summed again.
        assert state["total_capacity_bytes"] == 24 * _GB
        assert state["device_total_bytes"] == 24 * _GB

    async def test_enabled_worker_unreachable_collapses_to_unreachable(self):
        """self.ai#74 guard: worker unreachable → held null / unreachable, NOT a
        Kokoro-only under-count (which would let core over-grant)."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 5 * _GB)), \
             patch.object(chatterbox_client, "probe_state", AsyncMock(return_value=None)):
            state = await probe_vram_state_full()
        assert state["held_vram_bytes"] is None
        assert state["total_capacity_bytes"] is None
        assert state["device_used_bytes"] is None
        assert state["gpu_reachable"] is False
        assert state["status"] == "unreachable"

    async def test_enabled_worker_reports_null_held_collapses(self):
        """Worker reachable but its OWN cuda probe raised (held_bytes:null) → same
        collapse. A null slice is unknown, never a contributing 0."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 5 * _GB)), \
             patch.object(chatterbox_client, "probe_state",
                          AsyncMock(return_value={"held_bytes": None, "resident": True})):
            state = await probe_vram_state_full()
        assert state["held_vram_bytes"] is None
        assert state["status"] == "unreachable"

    async def test_enabled_worker_zero_held_is_a_real_contributing_zero(self):
        """A worker on GPU with nothing loaded reports held 0 — a KNOWN zero that
        sums cleanly (distinct from null). held stays the Kokoro figure."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 5 * _GB)), \
             patch.object(chatterbox_client, "probe_state",
                          AsyncMock(return_value={"held_bytes": 0, "resident": False})):
            state = await probe_vram_state_full()
        assert state["held_vram_bytes"] == 5 * _GB
        assert state["status"] == "ok"

    async def test_enabled_residency_ors_worker_in(self):
        """model_resident ORs both engines: worker-resident shows even when Kokoro
        is unloaded (observability, never byte accounting)."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state",
                          return_value=_local("ok", 1 * _GB, resident=False)), \
             patch.object(chatterbox_client, "probe_state",
                          AsyncMock(return_value={"held_bytes": 2 * _GB, "resident": True})):
            state = await probe_vram_state_full()
        assert state["model_resident"] is True
        assert state["held_vram_bytes"] == 3 * _GB

    async def test_enabled_local_no_gpu_returns_local_worker_not_queried(self):
        """Local no_gpu means the whole pod is CPU — the worker holds no VRAM
        either. Return the honest local 0; do not query the worker."""
        probe_state = AsyncMock()
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("no_gpu", 0)), \
             patch.object(chatterbox_client, "probe_state", probe_state):
            state = await probe_vram_state_full()
        probe_state.assert_not_awaited()
        assert state["held_vram_bytes"] == 0
        assert state["status"] == "no_gpu"

    async def test_enabled_local_unreachable_returns_local_worker_not_queried(self):
        """Local case (b) already dominates with a null answer; the worker can add
        no signal, so skip it."""
        probe_state = AsyncMock()
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("unreachable", None)), \
             patch.object(chatterbox_client, "probe_state", probe_state):
            state = await probe_vram_state_full()
        probe_state.assert_not_awaited()
        assert state["held_vram_bytes"] is None
        assert state["status"] == "unreachable"


class TestReleaseReflectsWorkerFree:
    async def test_freed_delta_spans_both_contexts(self):
        """The whole Phase-2 payoff: a worker-side free shows up in freed_bytes.
        before = 4G(kokoro)+6G(worker)=10G; after = 2G+0 = 2G → freed 8G ->
        released. The delta could NOT see the worker's 6G in Phase 1."""
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_local("ok", 4 * _GB), _local("ok", 2 * _GB)]), \
             patch.object(chatterbox_client, "probe_state",
                          AsyncMock(side_effect=[{"held_bytes": 6 * _GB, "resident": True},
                                                 {"held_bytes": 0, "resident": False}])), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=True)), \
             patch.object(vram_lease, "_unload_backend", MagicMock()), \
             patch.object(chatterbox_client, "release", AsyncMock(return_value={"status": "released", "freed_bytes": 6 * _GB})):
            resp = await handle_vram_release(5 * _GB, 30.0)
        assert resp.status == "released"
        assert resp.freed_bytes == 8 * _GB

    async def test_worker_dark_during_release_is_truthful_partial(self):
        """Enabled but worker unreachable at baseline → aggregated status is
        unreachable → partial/0, never a fabricated free and never an unload."""
        unload = MagicMock()
        with patch.object(vram_lease.settings, "chatterbox_enabled", True), \
             patch.object(vram_lease, "probe_vram_state", return_value=_local("ok", 4 * _GB)), \
             patch.object(chatterbox_client, "probe_state", AsyncMock(return_value=None)), \
             patch.object(vram_lease, "_unload_backend", unload):
            resp = await handle_vram_release(5 * _GB, 30.0)
        unload.assert_not_called()
        assert resp.status == "partial"
        assert resp.freed_bytes == 0

    async def test_worker_release_leg_skipped_when_disabled(self):
        """chatterbox_enabled=False → the worker-release leg is never invoked
        (worker not part of footprint)."""
        release = AsyncMock()
        with patch.object(vram_lease.settings, "chatterbox_enabled", False), \
             patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_local("ok", 10 * _GB), _local("ok", 2 * _GB)]), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=True)), \
             patch.object(vram_lease, "_unload_backend", MagicMock()), \
             patch.object(chatterbox_client, "release", release):
            resp = await handle_vram_release(5 * _GB, 30.0)
        release.assert_not_awaited()
        assert resp.status == "released"
        assert resp.freed_bytes == 8 * _GB
