"""Main-venv httpx client to the Chatterbox WORKER process.

This is the MAIN process's (torch 2.8) only handle on Chatterbox: a thin httpx
client that proxies to the sibling worker's localhost control API on
``settings.chatterbox_control_url`` (default ``http://127.0.0.1:8881``, INTERNAL —
core never talks to the worker). It replaces INTEGRATION-PLAN v1's planned
in-process ``chatterbox_v1.py`` backend — under the per-engine-process model there
is no in-process Chatterbox backend at all (§1.4).

The worker returns RAW float32 PCM + an ``X-Sample-Rate`` header; the caller
(``TTSService``) wraps that in a single ``AudioChunk`` and encodes via the shared
``StreamingAudioWriter``. Every call carries a bounded timeout so a wedged/hung
worker surfaces as an error, never a hang (§7 "two processes, one liveness anchor").
"""

import logging
from typing import Optional, Tuple

import httpx

from ..core.config import settings

log = logging.getLogger(__name__)

# Bounded timeouts. Connect is short so an unbound/dead worker fails fast (a first
# clone can race the worker's bind, §1.3). Synthesis can be slow for long text.
_CONNECT_TIMEOUT = 2.0
_SYNTH_READ_TIMEOUT = 300.0
_CONTROL_READ_TIMEOUT = 5.0


class ChatterboxUnavailable(RuntimeError):
    """The Chatterbox worker was unreachable, timed out, or returned an error.

    Raised on the synth/clone path so the router surfaces a clean failure rather
    than hanging. VRAM probe/release methods swallow errors and return ``None``
    instead (the aggregator/release side treats that as unreachable)."""


def _url(path: str) -> str:
    return settings.chatterbox_control_url.rstrip("/") + path


async def _generate(
    path: str,
    text: str,
    audio_prompt_path: Optional[str],
    exaggeration: Optional[float],
    cfg_weight: Optional[float],
) -> Tuple[bytes, int]:
    payload: dict = {"text": text}
    if audio_prompt_path is not None:
        payload["audio_prompt_path"] = audio_prompt_path
    if exaggeration is not None:
        payload["exaggeration"] = exaggeration
    if cfg_weight is not None:
        payload["cfg_weight"] = cfg_weight
    return await _post(path, payload)


async def _post(path: str, payload: dict) -> Tuple[bytes, int]:
    """POST ``payload`` to the worker ``path`` and return (raw PCM, sample_rate).
    Shared by the single-clip (/synth, /clone) and multi-clip (/blend) paths."""
    timeout = httpx.Timeout(_SYNTH_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(_url(path), json=payload)
    except httpx.HTTPError as e:
        raise ChatterboxUnavailable(f"chatterbox worker unreachable: {e}") from e

    if resp.status_code != 200:
        raise ChatterboxUnavailable(
            f"chatterbox worker returned {resp.status_code}: {resp.text[:200]}"
        )

    sr = resp.headers.get("X-Sample-Rate")
    if sr is None:
        raise ChatterboxUnavailable("chatterbox worker response missing X-Sample-Rate")
    try:
        sample_rate = int(sr)
    except (TypeError, ValueError) as e:
        raise ChatterboxUnavailable(f"invalid X-Sample-Rate {sr!r}") from e

    return resp.content, sample_rate


async def synth(
    text: str,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
) -> Tuple[bytes, int]:
    """Synthesise ``text`` in Chatterbox's default voice. Returns (raw float32
    PCM bytes, sample_rate). Raises ``ChatterboxUnavailable`` on any failure."""
    return await _generate("/synth", text, None, exaggeration, cfg_weight)


async def clone(
    text: str,
    audio_prompt_path: str,
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
) -> Tuple[bytes, int]:
    """Zero-shot clone: synthesise ``text`` in the voice of ``audio_prompt_path``
    (a reference clip). Wired to a ticket-gated route in Phase 3. Returns (raw
    float32 PCM bytes, sample_rate)."""
    return await _generate("/clone", text, audio_prompt_path, exaggeration, cfg_weight)


async def blend(
    text: str,
    references: "list[tuple[str, float]]",
    exaggeration: Optional[float] = None,
    cfg_weight: Optional[float] = None,
) -> Tuple[bytes, int]:
    """Weighted multi-clip blend: synthesise ``text`` in a NEW voice interpolated
    from ``references`` (a list of ``(audio_prompt_path, weight)``). The worker
    interpolates the clips' speaker embeddings. Returns (raw float32 PCM, sr)."""
    payload = {
        "text": text,
        "references": [
            {"audio_prompt_path": p, "weight": float(w)} for p, w in references
        ],
    }
    if exaggeration is not None:
        payload["exaggeration"] = exaggeration
    if cfg_weight is not None:
        payload["cfg_weight"] = cfg_weight
    return await _post("/blend", payload)


async def probe_state() -> Optional[dict]:
    """The worker's full ``/vram-state`` body — ``{"held_bytes": int|None,
    "resident": bool}`` — or ``None`` if the worker is unreachable/malformed.

    Phase 2's aggregator uses BOTH fields: ``held_bytes`` for the cross-process
    held sum (a ``None`` here, or a ``None`` return, collapses the whole answer to
    the tri-state ``unreachable`` — never a false 0, self.ai#74), and ``resident``
    to OR the worker's residency into the observability ``model_resident`` flag.
    Never raises. ``held_bytes`` is passed through verbatim (incl. its own
    ``None`` = worker-probe-raised), validated by the aggregator, not here."""
    timeout = httpx.Timeout(_CONTROL_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(_url("/vram-state"))
        if resp.status_code != 200:
            return None
        body = resp.json()
    except Exception as e:
        log.debug("chatterbox probe_state unreachable (%r)", e)
        return None
    if not isinstance(body, dict):
        return None
    held = body.get("held_bytes")
    # held may legitimately be None (worker-side probe raised); pass it through so
    # the aggregator applies the collapse. A non-None held must be a real int.
    if held is not None and (isinstance(held, bool) or not isinstance(held, int)):
        return None
    return {"held_bytes": held, "resident": bool(body.get("resident", False))}


async def probe_held() -> Optional[int]:
    """The worker's own reserved VRAM in bytes, or ``None`` if the worker is
    unreachable/malformed. Phase 2's aggregator collapses a ``None`` here to the
    tri-state ``unreachable`` (never a false 0). Never raises."""
    timeout = httpx.Timeout(_CONTROL_READ_TIMEOUT, connect=_CONNECT_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(_url("/vram-state"))
        if resp.status_code != 200:
            return None
        held = resp.json().get("held_bytes")
    except Exception as e:
        log.debug("chatterbox probe_held unreachable (%r)", e)
        return None
    if held is None or isinstance(held, bool) or not isinstance(held, int):
        return None
    return held


async def release(
    target_bytes: int, timeout_seconds: float, force: bool = False
) -> Optional[dict]:
    """Ask the worker to unload + ``empty_cache()`` toward ``target_bytes``.
    Returns the worker's ``{status, freed_bytes}`` body, or ``None`` if the worker
    was unreachable. Never raises (best-effort, mirrors the Kokoro unload leg).

    ``force`` = e-stop: the worker skips its own synth-lock drain-wait and unloads
    immediately (forwarded from the main handler)."""
    # Read budget = the worker's own drain-wait budget plus slack for the release.
    read_timeout = max(_CONTROL_READ_TIMEOUT, float(timeout_seconds) + 5.0)
    timeout = httpx.Timeout(read_timeout, connect=_CONNECT_TIMEOUT)
    body = {
        "target_bytes": int(target_bytes),
        "timeout_seconds": float(timeout_seconds),
        "force": bool(force),
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(_url("/vram-release"), json=body)
        if resp.status_code != 200:
            log.warning("chatterbox release returned HTTP %s", resp.status_code)
            return None
        return resp.json()
    except Exception as e:
        log.warning("chatterbox release unreachable (%r); continuing to live-verify", e)
        return None
