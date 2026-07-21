"""
self.ai#25: tests for the service-ticket auth layer on POST /v1/audio/speech
(api/src/core/auth.py), ported from self.llamolotl's api/tests/test_auth.py
(self.llamolotl#12) per context/kits/cavekit-service-mesh-ticket-auth.md
(self.ai repo).

Uses a plain, unauthenticated TestClient(app) (not the module-level `client`
in test_openai_endpoints.py, which pre-attaches a valid ticket via
mint_test_ticket() so the rest of that file's TTS-behavior tests aren't
blocked by this dependency) -- each test here controls exactly what ticket,
if any, is sent.

TestClient(app) is never entered as a `with` context manager, so FastAPI's
lifespan (which loads the real Kokoro model) never runs -- consistent with
self.llamolotl's own test convention. Because of that, a ticket that clears
the auth layer still fails downstream (TTSService not initialized), so the
"correct scope is accepted" cases below assert "not 401/403", not "200" --
proof the auth layer let the request through, not proof synthesis works
end-to-end (that's test_openai_endpoints.py's job, via mocked services).
"""

import time

import jwt
import pytest
from fastapi.testclient import TestClient

from api.src.main import app
from api.tests.conftest import (
    TEST_SERVICE_AUTH_AUDIENCE,
    TEST_SERVICE_AUTH_SECRET,
    mint_test_ticket,
)


def _post_speech(client, headers=None):
    return client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hello", "voice": "alloy", "stream": False},
        headers=headers or {},
    )


@pytest.fixture
def auth_client():
    """Unauthenticated TestClient -- no default ticket attached."""
    return TestClient(app)


class TestMissingOrMalformedTicket:
    def test_no_ticket_header_is_rejected(self, auth_client):
        resp = _post_speech(auth_client)
        assert resp.status_code == 401
        assert "X-Selfai-Ticket" in resp.json()["detail"]

    def test_malformed_ticket_is_rejected(self, auth_client):
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": "not-a-jwt"})
        assert resp.status_code == 401
        assert "Invalid service ticket" in resp.json()["detail"]

    def test_empty_ticket_header_is_rejected(self, auth_client):
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": ""})
        assert resp.status_code == 401


class TestExpiryAndAudience:
    def test_expired_ticket_is_rejected(self, auth_client):
        now = int(time.time())
        expired = jwt.encode(
            {
                "iss": "self.ai",
                "aud": TEST_SERVICE_AUTH_AUDIENCE,
                "scope": "audio:synthesize",
                "iat": now - 600,
                "exp": now - 60,
            },
            TEST_SERVICE_AUTH_SECRET,
            algorithm="HS256",
        )
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": expired})
        assert resp.status_code == 401
        assert "expired" in resp.json()["detail"].lower()

    def test_wrong_audience_is_rejected(self, auth_client):
        ticket = mint_test_ticket(audience="self.transcribe")
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": ticket})
        assert resp.status_code == 401
        assert "audience" in resp.json()["detail"].lower()

    def test_wrong_signing_secret_is_rejected(self, auth_client):
        ticket = mint_test_ticket(secret="not-the-real-secret")
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": ticket})
        assert resp.status_code == 401
        assert "invalid" in resp.json()["detail"].lower()


class TestScopeEnforcement:
    def test_correct_scope_passes_auth_layer(self, auth_client):
        ticket = mint_test_ticket(scope="audio:synthesize")
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": ticket})
        assert resp.status_code not in (401, 403)

    def test_missing_scope_is_rejected_with_403(self, auth_client):
        ticket = mint_test_ticket(scope="some:other:scope")
        resp = _post_speech(auth_client, {"X-Selfai-Ticket": ticket})
        assert resp.status_code == 403
        assert "audio:synthesize" in resp.json()["detail"]


class TestNoSecretConfigured:
    def test_missing_server_secret_fails_closed_with_503(self, auth_client):
        """If SERVICE_AUTH_SECRET isn't set on this node at all, every
        ticket check must fail closed (503), never silently accept."""
        import api.src.core.auth as auth_module

        original_secret = auth_module.SERVICE_AUTH_SECRET
        auth_module.SERVICE_AUTH_SECRET = ""
        try:
            ticket = mint_test_ticket()
            resp = _post_speech(auth_client, {"X-Selfai-Ticket": ticket})
            assert resp.status_code == 503
        finally:
            auth_module.SERVICE_AUTH_SECRET = original_secret


class TestHealthAndDiscoveryStayOpen:
    """self.speak's read-only discovery endpoints (and /health) are
    deliberately left unticketed -- see the kit's R2 section."""

    def test_health_requires_no_ticket(self, auth_client):
        resp = auth_client.get("/health")
        assert resp.status_code == 200

    def test_voices_requires_no_ticket(self, auth_client):
        resp = auth_client.get("/v1/audio/voices")
        # Unticketed by design; whatever status TTSService init produces
        # here (200 with an empty/real list, or a 500 since lifespan never
        # ran in this harness) must not be an auth rejection.
        assert resp.status_code not in (401, 403)
