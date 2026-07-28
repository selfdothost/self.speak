"""
Ticket-based auth for self.speak's TTS endpoint (self.ai#25).

self.ai mints short-lived, scoped JWTs before calling into this service
(self.ai#25's proposal: self.ai is the yard's internal ticket-granting
service). This module is the validating side: the mutating endpoint below
requires a valid ticket in the `X-Selfai-Ticket` header — signature,
audience, expiry, and scope are all checked.

Ported verbatim from self.llamolotl's api/auth.py (self.llamolotl#12,
merged 2026-07-09, the first leg of this mesh) per
context/kits/cavekit-service-mesh-ticket-auth.md (self.ai repo) — same JWT
shape, same fail-closed behavior, same SERVICE_AUTH_SECRET env var (shared
via the selfai-service-auth ExternalSecret, already consumed by
self.llamolotl and now this service too). Only the scope taxonomy and
default audience are specific to self.speak.

Deliberately a plain os.environ-based module rather than folded into
api/src/core/config.py's pydantic Settings — this keeps the validating side
byte-for-byte comparable across all three backends now on this pattern
(self.llamolotl, self.transcribe, self.speak), which matters more here than
fitting this repo's otherwise-pydantic config convention.

Scope taxonomy (mirror of self.ai's minting side in
api/selfai_ui/utils/service_auth.py / routers/audio.py — keep both lists in
sync):
  audio:synthesize  - POST /v1/audio/speech
  system:read       - GET  /api/system/vram-state
  system:write      - POST /api/system/vram-release

The two `system:*` scopes are the VRAM-lease control plane
(context/kits/cavekit-vram-lease-client.md — R1 VRAM state reporting,
R2 release-request handling). Names are reused verbatim from self.llamolotl's
taxonomy so one taxonomy covers both control planes; core already mints this
pair for the self.llamolotl audience, so its minting side
(api/selfai_ui/utils/service_auth.py) only needs to ADD audience `self.speak`
to the same two scopes — an explicit two-sided coordination. The scope strings
here MUST match core's minted strings byte-for-byte; keep both lists in sync.

self.speak's synthesis surface has exactly one meaningful mutating endpoint,
so that slice of the taxonomy is deliberately a single scope rather than an
unused hierarchy (see the kit's R2 section, which also leaves the read-only
/v1/audio/voices and /v1/models discovery endpoints unticketed for now — same
posture as self.llamolotl's /health). /health stays unticketed too.

NetworkPolicy-level pod-to-pod restriction is a complementary defense-in-
depth layer, explicitly out of scope here (see self.llamolotl#12).
"""

import logging
import os
from typing import Callable, List, Optional

import jwt
from fastapi import Header, HTTPException, status

log = logging.getLogger(__name__)

# Shared HMAC secret with self.ai. Empty by default so a misconfigured
# deployment fails closed (every ticket check 503s) instead of silently
# accepting unsigned requests.
SERVICE_AUTH_SECRET = os.environ.get("SERVICE_AUTH_SECRET", "")
# This service's own audience value — tickets minted for any other
# audience are rejected even if otherwise well-formed and correctly signed.
SERVICE_AUTH_AUDIENCE = os.environ.get("SERVICE_AUTH_AUDIENCE", "self.speak")
SERVICE_AUTH_ALGORITHM = "HS256"

TICKET_HEADER = "X-Selfai-Ticket"

# Named scope constants so route decorators reference these rather than bare
# strings (reduces typo-drift against the wire scope core mints). These are the
# canonical wire strings — see the module docstring's scope taxonomy.
SCOPE_AUDIO_SYNTHESIZE = "audio:synthesize"
SCOPE_SYSTEM_READ = "system:read"
SCOPE_SYSTEM_WRITE = "system:write"


class TicketError(HTTPException):
    """A ticket validation failure. Carries a plain-English detail message
    (never the raw exception) so we don't leak signing internals on the wire."""

    def __init__(self, detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
        super().__init__(status_code=status_code, detail=detail)


def _decode_ticket(token: str) -> dict:
    if not SERVICE_AUTH_SECRET:
        log.error(
            "SERVICE_AUTH_SECRET is not configured — rejecting all service "
            "tickets until it is set."
        )
        raise TicketError(
            "Service auth is not configured on this node",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    try:
        claims = jwt.decode(
            token,
            SERVICE_AUTH_SECRET,
            algorithms=[SERVICE_AUTH_ALGORITHM],
            audience=SERVICE_AUTH_AUDIENCE,
        )
    except jwt.ExpiredSignatureError:
        raise TicketError("Service ticket expired")
    except jwt.InvalidAudienceError:
        raise TicketError("Service ticket audience mismatch")
    except jwt.InvalidTokenError as e:
        log.warning("Rejected malformed service ticket: %s", e)
        raise TicketError("Invalid service ticket")

    return claims


def _scopes_from_claims(claims: dict) -> List[str]:
    scope = claims.get("scope", "")
    if isinstance(scope, str):
        return scope.split()
    if isinstance(scope, (list, tuple)):
        return list(scope)
    return []


def require_scope(required_scope: str) -> Callable[..., dict]:
    """FastAPI dependency factory.

    Usage: `Depends(require_scope("audio:synthesize"))` on a route.
    Validates the `X-Selfai-Ticket` header: present, correctly signed, not
    expired, audience == this service, and `required_scope` is among the
    ticket's granted scopes. Returns the decoded claims (available to the
    route via the dependency's return value, though most routes don't need
    it).
    """

    def _dependency(
        x_selfai_ticket: Optional[str] = Header(default=None, alias=TICKET_HEADER),
    ) -> dict:
        if not x_selfai_ticket:
            raise TicketError(f"Missing {TICKET_HEADER} header")

        claims = _decode_ticket(x_selfai_ticket)

        granted = _scopes_from_claims(claims)
        if required_scope not in granted:
            raise TicketError(
                f"Ticket does not grant required scope '{required_scope}'",
                status.HTTP_403_FORBIDDEN,
            )

        return claims

    return _dependency
