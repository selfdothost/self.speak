"""Debug router surface (self.speak#2).

`/debug/session_pools` read `manager._session_pools` off ModelManager. That
attribute does not exist on this fork's single-backend ModelManager (which has
`_config`, `_reload_lock`, `_device`, `_backend`), so the endpoint could only
ever raise AttributeError. It was a leftover from upstream kokoro-fastapi's
multi-backend ONNX architecture, which was never ported.

It was REMOVED rather than reimplemented, for three reasons:
  * its subject -- ONNX CPU/GPU session pools -- does not exist here at all, so
    there is nothing to report against the current shape;
  * the useful part of what it was for (is a model resident, what is on the
    card) already exists as GET /api/system/vram-state, which is auth-gated
    behind require_scope(SCOPE_SYSTEM_READ);
  * this router carries NO auth dependency, so removing a route shrinks an
    ungated surface rather than growing one.

TestClient(app) is never entered as a `with` block, so FastAPI's lifespan (real
Kokoro load) never runs -- same convention as test_chatterbox_clone.py.
"""

from fastapi.testclient import TestClient

from api.src.main import app


class TestDeadEndpointIsGone:
    def test_session_pools_is_not_routed(self):
        """404, not 500. Pre-removal this raised AttributeError and surfaced as
        a server error -- which is why 'it 500s' was the symptom in #2."""
        client = TestClient(app)

        resp = client.get("/debug/session_pools")

        assert resp.status_code == 404

    def test_no_route_references_session_pools(self):
        """Guards the whole app, not just the one path -- so a copy of this
        endpoint reappearing under another name still fails this test."""
        paths = [getattr(r, "path", "") for r in app.routes]

        assert not [p for p in paths if "session_pool" in p]


class TestSurvivingDebugRoutesStillWork:
    """The removal must not take the rest of the router with it."""

    def test_threads_still_responds(self):
        client = TestClient(app)

        resp = client.get("/debug/threads")

        assert resp.status_code == 200
        # Keys read off the handler itself. The first draft of this test
        # asserted "thread_count", which this endpoint has never returned --
        # a test asserting a key that does not exist fails for the wrong reason
        # and teaches you nothing about the removal.
        body = resp.json()
        for key in ("total_threads", "active_threads", "thread_names", "memory_mb"):
            assert key in body

    def test_storage_still_responds(self):
        client = TestClient(app)

        resp = client.get("/debug/storage")

        assert resp.status_code == 200
        assert "storage_info" in resp.json()

    def test_system_still_responds(self):
        """This one matters most for the removal: it is the other GPUtil user in
        the file, so it proves the import cleanup did not break it."""
        client = TestClient(app)

        resp = client.get("/debug/system")

        assert resp.status_code == 200
        body = resp.json()
        for key in ("cpu", "memory", "process", "network", "gpu"):
            assert key in body
