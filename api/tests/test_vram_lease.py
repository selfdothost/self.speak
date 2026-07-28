"""Tests for self.speak's VRAM-lease client (cavekit-vram-lease-client).

Covers all 14 acceptance criteria (R1-AC1..AC6, R2-AC1..AC8) across the live
tri-state probe (T-002), the wire models (T-003/T-004), the in-flight counter +
drain-wait (T-005), the release orchestration (T-006), and the two endpoints
(T-007/T-008).

Conventions mirror this repo's ``test_ticket_auth.py``:
  * ``TestClient(app)`` is NEVER entered as a ``with`` context manager, so
    FastAPI's lifespan (which loads the real Kokoro model) never runs. A ticket
    that clears auth still reaches the (patched) handler.
  * ``mint_test_ticket(scope=...)`` from ``conftest.py`` mints service tickets;
    ``SERVICE_AUTH_SECRET`` is already ``setdefault``-ed there.

CI runs this under ``.[test,cpu]`` (CPU torch → ``torch.cuda.is_available()`` is
False), so the case-(c) probe and the CPU no-op release path run UNMOCKED and
live; the GPU cases (a)/(b) run via patched ``torch.cuda``.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import api.src.inference.vram_lease as vram_lease
from api.src.inference.vram_lease import (
    handle_vram_release,
    probe_vram_state,
    synthesis_in_flight,
    track_synthesis,
    wait_for_drain,
)
from api.src.main import app
from api.src.structures.schemas import (
    VramReleaseRequest,
    VramReleaseResponse,
    VramStateResponse,
)
from api.tests.conftest import mint_test_ticket

_GB = 1024 * 1024 * 1024


# ─── T-002: live tri-state torch.cuda VRAM probe ─────────────────────────


class TestProbeVramState:
    def test_case_a_gpu_reachable_returns_real_ints(self):
        """(a) GPU reachable & probed → real ints, status='ok' (R1-AC4)."""
        with patch.object(vram_lease.torch.cuda, "is_available", return_value=True), \
             patch.object(vram_lease.torch.cuda, "mem_get_info", return_value=(2 * _GB, 24 * _GB)), \
             patch.object(vram_lease.torch.cuda, "memory_allocated", return_value=5 * _GB):
            state = probe_vram_state()
        assert state["status"] == "ok"
        assert state["gpu_reachable"] is True
        assert isinstance(state["held_vram_bytes"], int)
        assert isinstance(state["total_capacity_bytes"], int)
        assert state["held_vram_bytes"] == 5 * _GB
        assert state["total_capacity_bytes"] == 24 * _GB

    def test_case_a_idle_held_is_near_zero_int_not_null(self):
        """(a) idle → held is a real (near-zero) int, NOT null (R1-AC4)."""
        with patch.object(vram_lease.torch.cuda, "is_available", return_value=True), \
             patch.object(vram_lease.torch.cuda, "mem_get_info", return_value=(24 * _GB, 24 * _GB)), \
             patch.object(vram_lease.torch.cuda, "memory_allocated", return_value=0):
            state = probe_vram_state()
        assert state["status"] == "ok"
        assert state["held_vram_bytes"] == 0
        assert state["held_vram_bytes"] is not None

    def test_case_b_probe_raises_returns_null_not_zero(self):
        """(b) mem_get_info raises → both null, status='unreachable', NEVER 0
        (R1-AC4/AC5). The try/except is load-bearing — a throw must not crash."""
        with patch.object(vram_lease.torch.cuda, "is_available", return_value=True), \
             patch.object(vram_lease.torch.cuda, "mem_get_info", side_effect=RuntimeError("CUDA init failed")), \
             patch.object(vram_lease.torch.cuda, "memory_allocated", return_value=0):
            state = probe_vram_state()
        assert state["status"] == "unreachable"
        assert state["gpu_reachable"] is False
        assert state["held_vram_bytes"] is None
        assert state["total_capacity_bytes"] is None
        # The false-zero R1-AC5 explicitly bans.
        assert state["held_vram_bytes"] != 0

    def test_case_c_no_cuda_returns_integer_zero(self):
        """(c) no CUDA → integer 0, status='no_gpu' (R1-AC4). Runs UNMOCKED and
        live in CI (CPU torch), so it must also pass without any patch below."""
        with patch.object(vram_lease.torch.cuda, "is_available", return_value=False):
            state = probe_vram_state()
        assert state["status"] == "no_gpu"
        assert state["gpu_reachable"] is False
        assert state["held_vram_bytes"] == 0
        assert state["total_capacity_bytes"] == 0

    def test_case_c_unmocked_cpu_is_no_gpu(self):
        """Live CI gate: on CPU torch the unmocked probe is case (c)."""
        state = probe_vram_state()
        # Under .[test,cpu] this is no_gpu; on a real GPU box it may be ok — in
        # either case held is a concrete int (never null), never a crash.
        assert state["status"] in ("no_gpu", "ok")
        if state["status"] == "no_gpu":
            assert state["held_vram_bytes"] == 0
            assert state["total_capacity_bytes"] == 0

    def test_liveness_never_cached_reflects_changing_allocation(self):
        """R1-AC2: two calls with a changed allocation return the changed value —
        never a cached figure."""
        with patch.object(vram_lease.torch.cuda, "is_available", return_value=True), \
             patch.object(vram_lease.torch.cuda, "mem_get_info", return_value=(0, 24 * _GB)), \
             patch.object(vram_lease.torch.cuda, "memory_allocated", side_effect=[3 * _GB, 7 * _GB]):
            first = probe_vram_state()
            second = probe_vram_state()
        assert first["held_vram_bytes"] == 3 * _GB
        assert second["held_vram_bytes"] == 7 * _GB


# ─── T-003 / T-004: wire models ──────────────────────────────────────────


class TestWireModels:
    def test_vram_state_null_serialises_to_json_null_not_zero(self):
        """R1-AC4/AC5/AC6: an unknown held serialises to JSON null, never 0."""
        model = VramStateResponse(
            held_vram_bytes=None,
            total_capacity_bytes=None,
            gpu_reachable=False,
            status="unreachable",
            model_resident=False,
        )
        payload = json.loads(model.model_dump_json())
        assert payload["held_vram_bytes"] is None
        assert payload["total_capacity_bytes"] is None
        assert payload["held_vram_bytes"] != 0

    def test_vram_state_int_fields_stay_int(self):
        """R1-AC4/AC6: reachable figures serialise as JSON ints in bytes."""
        model = VramStateResponse(
            held_vram_bytes=5 * _GB,
            total_capacity_bytes=24 * _GB,
            gpu_reachable=True,
            status="ok",
            model_resident=True,
        )
        payload = json.loads(model.model_dump_json())
        assert payload["held_vram_bytes"] == 5 * _GB
        assert isinstance(payload["held_vram_bytes"], int)
        assert isinstance(payload["total_capacity_bytes"], int)
        # Field names are exactly what core's transport parses (R1-AC6).
        assert set(["held_vram_bytes", "total_capacity_bytes"]).issubset(payload)

    def test_release_request_amount_only_no_mechanism(self):
        """R2-AC1: request is {target_bytes, timeout_seconds} — no mechanism."""
        req = VramReleaseRequest(**{"target_bytes": 5_000_000_000, "timeout_seconds": 30.0})
        assert req.target_bytes == 5_000_000_000
        assert req.timeout_seconds == 30.0
        assert "mechanism" not in req.model_dump()

    def test_release_response_freed_bytes_never_boolean_on_wire(self):
        """R2-AC4: freed_bytes reaches the wire as a JSON integer, never a bool
        (core rejects a boolean freed_bytes)."""
        model = VramReleaseResponse(status="released", freed_bytes=8 * _GB)
        payload = json.loads(model.model_dump_json())
        freed = payload["freed_bytes"]
        # Exactly core's _map_response acceptance test.
        assert isinstance(freed, int) and not isinstance(freed, bool)
        assert freed == 8 * _GB

    def test_release_response_boolean_freed_bytes_does_not_survive(self):
        """R2-AC4: a boolean freed_bytes is either rejected or coerced away so a
        JSON boolean never reaches the wire."""
        try:
            model = VramReleaseResponse(status="released", freed_bytes=True)
        except Exception:
            return  # rejected outright — acceptable
        payload = json.loads(model.model_dump_json())
        assert not isinstance(payload["freed_bytes"], bool)

    def test_release_response_status_vocabulary_enforced(self):
        """R2-AC4: a status outside the recognised vocabulary is rejected."""
        with pytest.raises(Exception):
            VramReleaseResponse(status="denied", freed_bytes=0)


# ─── T-005: in-flight counter + bounded drain-wait ───────────────────────


class TestDrainWait:
    async def test_idle_returns_true_immediately(self):
        """R2-AC2 idle-fast-path: counter 0 → True with no wait."""
        assert synthesis_in_flight() == 0
        assert await wait_for_drain(5.0) is True

    async def test_times_out_while_synthesis_held_open(self):
        """R2-AC2/AC6: held past the deadline → False, bounded by timeout."""
        async with track_synthesis():
            assert synthesis_in_flight() == 1
            result = await wait_for_drain(0.05)
        assert result is False

    async def test_returns_true_when_released_before_deadline(self):
        """R2-AC2: a synthesis that finishes within the window → True."""

        async def worker():
            async with track_synthesis():
                await asyncio.sleep(0.05)

        task = asyncio.create_task(worker())
        await asyncio.sleep(0.01)  # let the worker enter track_synthesis
        assert synthesis_in_flight() == 1
        result = await wait_for_drain(2.0)
        assert result is True
        await task
        assert synthesis_in_flight() == 0

    async def test_counter_decremented_when_wrapped_body_raises(self):
        """The counter must not leak when the wrapped body raises."""
        assert synthesis_in_flight() == 0
        with pytest.raises(ValueError):
            async with track_synthesis():
                assert synthesis_in_flight() == 1
                raise ValueError("boom")
        assert synthesis_in_flight() == 0


# ─── T-006: release orchestration ────────────────────────────────────────


def _state(status, held):
    return {
        "held_vram_bytes": held,
        "total_capacity_bytes": 24 * _GB if held is not None else None,
        "gpu_reachable": status == "ok",
        "status": status,
        "model_resident": True,
    }


class TestHandleVramRelease:
    async def test_freed_is_live_delta_not_unload_return(self):
        """R2-AC3: an unload that 'succeeds' but frees nothing (before==after)
        reports freed 0 — the figure is the live delta, never the unload return."""
        unload = MagicMock()
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("ok", 10 * _GB), _state("ok", 10 * _GB)]), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=True)), \
             patch.object(vram_lease, "_unload_backend", unload):
            resp = await handle_vram_release(5 * _GB, 30.0)
        unload.assert_called_once()
        assert resp.freed_bytes == 0
        assert resp.status == "partial"

    async def test_still_generating_at_deadline_is_partial_no_unload(self):
        """R2-AC2/AC6: drain deadline hit while generating → partial/0, NO unload,
        returns within the budget."""
        unload = MagicMock()
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("ok", 10 * _GB)]), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=False)), \
             patch.object(vram_lease, "_unload_backend", unload):
            resp = await handle_vram_release(5 * _GB, 30.0)
        unload.assert_not_called()
        assert resp.status == "partial"
        assert resp.freed_bytes == 0

    async def test_smaller_than_target_is_truthful_partial(self):
        """R2-AC5: a real but smaller free → partial with the real figure, never
        a fabricated success."""
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("ok", 10 * _GB), _state("ok", 8 * _GB)]), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=True)), \
             patch.object(vram_lease, "_unload_backend", MagicMock()):
            resp = await handle_vram_release(5 * _GB, 30.0)
        assert resp.status == "partial"
        assert resp.freed_bytes == 2 * _GB

    async def test_target_met_is_released_with_real_figure(self):
        """R2-AC4 happy path: freed >= target → released with the live figure."""
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("ok", 10 * _GB), _state("ok", 2 * _GB)]), \
             patch.object(vram_lease, "wait_for_drain", AsyncMock(return_value=True)), \
             patch.object(vram_lease, "_unload_backend", MagicMock()):
            resp = await handle_vram_release(5 * _GB, 30.0)
        assert resp.status == "released"
        assert resp.freed_bytes == 8 * _GB

    async def test_single_flight_second_call_is_busy(self):
        """R2-AC7: a second concurrent release → busy, no work done."""
        vram_lease._release_in_flight = True
        try:
            resp = await handle_vram_release(5 * _GB, 30.0)
        finally:
            vram_lease._release_in_flight = False
        assert resp.status == "busy"
        assert resp.freed_bytes == 0

    async def test_cpu_no_gpu_is_honest_released_zero_no_unload(self):
        """R2-AC8: CPU/no-CUDA → released/0 no-op, never an error, no unload."""
        unload = MagicMock()
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("no_gpu", 0)]), \
             patch.object(vram_lease, "_unload_backend", unload):
            resp = await handle_vram_release(5 * _GB, 30.0)
        unload.assert_not_called()
        assert resp.status == "released"
        assert resp.freed_bytes == 0

    async def test_unreachable_gpu_is_truthful_partial_zero(self):
        """Case (b) at release time: can't measure → partial/0, never fabricated."""
        with patch.object(vram_lease, "probe_vram_state",
                          side_effect=[_state("unreachable", None)]), \
             patch.object(vram_lease, "_unload_backend", MagicMock()) as unload:
            resp = await handle_vram_release(5 * _GB, 30.0)
        unload.assert_not_called()
        assert resp.status == "partial"
        assert resp.freed_bytes == 0

    async def test_cpu_release_runs_live_unmocked(self):
        """R2-AC8 live CI gate: on CPU torch, an unmocked release is a clean
        no-op (released/0), never an error."""
        resp = await handle_vram_release(5 * _GB, 1.0)
        # released/0 on CPU (case c); on a real GPU box it may free a real amount.
        assert resp.status in ("released", "partial")
        assert isinstance(resp.freed_bytes, int) and not isinstance(resp.freed_bytes, bool)


# ─── T-007 / T-008: endpoints ────────────────────────────────────────────


@pytest.fixture
def client():
    """Unauthenticated TestClient — no lifespan, no default ticket."""
    return TestClient(app)


def _read_hdr():
    return {"X-Selfai-Ticket": mint_test_ticket(scope="system:read")}


def _write_hdr():
    return {"X-Selfai-Ticket": mint_test_ticket(scope="system:write")}


class TestVramStateEndpoint:
    def test_case_a_returns_200_with_byte_fields(self, client):
        """R1-AC1/AC3: system:read ticket → 200 with int byte figures."""
        with patch("api.src.routers.system.probe_vram_state",
                   return_value=_state("ok", 5 * _GB)):
            resp = client.get("/api/system/vram-state", headers=_read_hdr())
        assert resp.status_code == 200
        body = resp.json()
        assert body["held_vram_bytes"] == 5 * _GB
        assert body["total_capacity_bytes"] == 24 * _GB
        assert body["status"] == "ok"

    def test_case_b_returns_200_null_not_5xx(self, client):
        """R1-AC3/AC4/AC5: unreachable → 200 with null held, never a 5xx, never 0."""
        with patch("api.src.routers.system.probe_vram_state",
                   return_value=_state("unreachable", None)):
            resp = client.get("/api/system/vram-state", headers=_read_hdr())
        assert resp.status_code == 200
        body = resp.json()
        assert body["held_vram_bytes"] is None
        assert body["status"] == "unreachable"

    def test_case_c_returns_200_integer_zero(self, client):
        """R1-AC4: no-CUDA → 200 with integer-0 held and status no_gpu."""
        with patch("api.src.routers.system.probe_vram_state",
                   return_value=_state("no_gpu", 0)):
            resp = client.get("/api/system/vram-state", headers=_read_hdr())
        assert resp.status_code == 200
        body = resp.json()
        assert body["held_vram_bytes"] == 0
        assert body["status"] == "no_gpu"

    def test_probe_exception_still_200_unreachable(self, client):
        """R1-AC3: even an unexpected probe raise → 200 unreachable, not 500."""
        with patch("api.src.routers.system.probe_vram_state",
                   side_effect=RuntimeError("unexpected")):
            resp = client.get("/api/system/vram-state", headers=_read_hdr())
        assert resp.status_code == 200
        assert resp.json()["status"] == "unreachable"

    def test_no_ticket_is_401(self, client):
        resp = client.get("/api/system/vram-state")
        assert resp.status_code == 401

    def test_wrong_scope_is_403(self, client):
        headers = {"X-Selfai-Ticket": mint_test_ticket(scope="audio:synthesize")}
        resp = client.get("/api/system/vram-state", headers=headers)
        assert resp.status_code == 403


class TestVramReleaseEndpoint:
    def test_valid_write_ticket_returns_freed_response(self, client):
        """R2-AC1: system:write ticket + body → 200 with the freed response."""
        released = VramReleaseResponse(status="released", freed_bytes=8 * _GB)
        with patch("api.src.routers.system.handle_vram_release",
                   AsyncMock(return_value=released)):
            resp = client.post(
                "/api/system/vram-release",
                json={"target_bytes": 5 * _GB, "timeout_seconds": 30.0},
                headers=_write_hdr(),
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "released"
        assert body["freed_bytes"] == 8 * _GB

    def test_busy_result_maps_to_409(self, client):
        """R2-AC7: a busy handler result → HTTP 409 for the racing caller."""
        busy = VramReleaseResponse(status="busy", freed_bytes=0)
        with patch("api.src.routers.system.handle_vram_release",
                   AsyncMock(return_value=busy)):
            resp = client.post(
                "/api/system/vram-release",
                json={"target_bytes": 5 * _GB, "timeout_seconds": 30.0},
                headers=_write_hdr(),
            )
        assert resp.status_code == 409

    def test_no_ticket_is_401(self, client):
        resp = client.post(
            "/api/system/vram-release",
            json={"target_bytes": 5 * _GB, "timeout_seconds": 30.0},
        )
        assert resp.status_code == 401

    def test_wrong_scope_is_403(self, client):
        headers = {"X-Selfai-Ticket": mint_test_ticket(scope="system:read")}
        resp = client.post(
            "/api/system/vram-release",
            json={"target_bytes": 5 * _GB, "timeout_seconds": 30.0},
            headers=headers,
        )
        assert resp.status_code == 403

    def test_missing_target_bytes_is_422(self, client):
        """R2-AC1: a body missing target_bytes fails pydantic validation."""
        resp = client.post(
            "/api/system/vram-release",
            json={"timeout_seconds": 30.0},
            headers=_write_hdr(),
        )
        assert resp.status_code == 422
