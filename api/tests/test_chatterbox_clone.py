"""Phase-3 Chatterbox clone/preview control routes + the service dispatch that
feeds them (INTEGRATION-PLAN-v2.md Phase 3).

Two layers:
  * route level — POST /api/voices/{clone,preview} multipart parse, ticket gate,
    the 200 happy path, the 503 engine-unavailable path, and reference validation.
    ``get_tts_service`` and ``AudioService.convert_audio`` are patched so no real
    model / encoder runs (mirrors test_openai_endpoints.py).
  * service level — generate_audio_stream(engine="chatterbox", audio_prompt_path=…)
    routes to chatterbox_client.clone; with no path it routes to .synth. This is
    the one genuinely new branch Phase 3 adds to the hot path.

TestClient(app) is never entered as a ``with`` block, so FastAPI's lifespan (real
Kokoro load) never runs — same convention as test_ticket_auth.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

import api.src.routers.control as control
from api.src.inference.base import AudioChunk
from api.src.inference.chatterbox_client import ChatterboxUnavailable
from api.src.main import app
from api.tests.conftest import mint_test_ticket


def _auth_headers(scope="audio:synthesize"):
    return {"X-Selfai-Ticket": mint_test_ticket(scope=scope)}


def _multipart(text="hello", ref=b"RIFFxxxxWAVE", fmt="wav"):
    files = {"reference": ("ref.wav", ref, "audio/wav")}
    data = {"text": text, "response_format": fmt}
    return files, data


def _chunk(output):
    # convert_audio returns an object exposing .output bytes.
    return SimpleNamespace(output=output)


class TestCloneRoute:
    def test_clone_happy_path_returns_encoded_audio(self):
        client = TestClient(app)
        svc = SimpleNamespace(
            generate_audio=AsyncMock(return_value=AudioChunk(np.zeros(4, dtype=np.int16)))
        )
        with patch.object(control, "get_tts_service", AsyncMock(return_value=svc)), \
             patch.object(control.AudioService, "convert_audio",
                          AsyncMock(side_effect=[_chunk(b"BODY"), _chunk(b"TAIL")])):
            files, data = _multipart()
            resp = client.post("/api/voices/clone", files=files, data=data,
                               headers=_auth_headers())
        assert resp.status_code == 200
        assert resp.content == b"BODYTAIL"
        assert resp.headers["content-type"].startswith("audio/wav")
        # The clone went through generate_audio with the staged reference path.
        _, kwargs = svc.generate_audio.call_args
        assert kwargs["engine"] == "chatterbox"
        assert kwargs["audio_prompt_path"]  # a real temp path was passed
        assert kwargs["text"] == "hello"

    def test_clone_worker_down_is_503(self):
        client = TestClient(app)
        svc = SimpleNamespace(
            generate_audio=AsyncMock(side_effect=ChatterboxUnavailable("no worker"))
        )
        with patch.object(control, "get_tts_service", AsyncMock(return_value=svc)):
            files, data = _multipart()
            resp = client.post("/api/voices/clone", files=files, data=data,
                               headers=_auth_headers())
        assert resp.status_code == 503
        assert resp.json()["detail"]["error"] == "engine_unavailable"

    def test_clone_empty_reference_is_400(self):
        client = TestClient(app)
        with patch.object(control, "get_tts_service", AsyncMock()):
            files, data = _multipart(ref=b"")
            resp = client.post("/api/voices/clone", files=files, data=data,
                               headers=_auth_headers())
        assert resp.status_code == 400

    def test_clone_requires_ticket(self):
        client = TestClient(app)
        files, data = _multipart()
        resp = client.post("/api/voices/clone", files=files, data=data)
        assert resp.status_code in (401, 403)

    def test_clone_wrong_scope_rejected(self):
        client = TestClient(app)
        files, data = _multipart()
        resp = client.post("/api/voices/clone", files=files, data=data,
                           headers=_auth_headers(scope="system:read"))
        assert resp.status_code == 403

    def test_clone_missing_text_is_422(self):
        client = TestClient(app)
        resp = client.post("/api/voices/clone",
                           files={"reference": ("r.wav", b"RIFF", "audio/wav")},
                           data={"response_format": "wav"}, headers=_auth_headers())
        assert resp.status_code == 422


class TestPreviewRoute:
    def test_preview_defaults_text_when_omitted(self):
        client = TestClient(app)
        svc = SimpleNamespace(
            generate_audio=AsyncMock(return_value=AudioChunk(np.zeros(4, dtype=np.int16)))
        )
        with patch.object(control, "get_tts_service", AsyncMock(return_value=svc)), \
             patch.object(control.AudioService, "convert_audio",
                          AsyncMock(side_effect=[_chunk(b"A"), _chunk(b"B")])):
            resp = client.post("/api/voices/preview",
                               files={"reference": ("r.wav", b"RIFF", "audio/wav")},
                               data={"response_format": "wav"}, headers=_auth_headers())
        assert resp.status_code == 200
        _, kwargs = svc.generate_audio.call_args
        assert kwargs["text"] == control._PREVIEW_DEFAULT_TEXT


class TestServiceDispatch:
    """generate_audio_stream routes chatterbox work to clone vs synth by whether a
    reference path is present — the new Phase-3 branch on the hot path."""

    async def _run(self, audio_prompt_path):
        from api.src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)  # skip __init__ (no model needed)
        writer = MagicMock(sample_rate=24000)
        pcm = (np.zeros(8, dtype=np.float32)).tobytes()
        with patch("api.src.inference.chatterbox_client.clone",
                   AsyncMock(return_value=(pcm, 24000))) as m_clone, \
             patch("api.src.inference.chatterbox_client.synth",
                   AsyncMock(return_value=(pcm, 24000))) as m_synth, \
             patch("api.src.services.tts_service.AudioNormalizer") as m_norm:
            m_norm.return_value.normalize.side_effect = lambda a: a
            chunks = [
                c async for c in svc.generate_audio_stream(
                    "hi", "", writer, engine="chatterbox",
                    output_format=None, audio_prompt_path=audio_prompt_path,
                )
            ]
        assert chunks  # produced at least one AudioChunk
        return m_clone, m_synth

    async def test_reference_path_routes_to_clone(self):
        m_clone, m_synth = await self._run("/tmp/ref.wav")
        m_clone.assert_awaited_once()
        m_synth.assert_not_awaited()

    async def test_no_reference_routes_to_synth(self):
        m_clone, m_synth = await self._run(None)
        m_synth.assert_awaited_once()

    async def _run_refs(self, references):
        from api.src.services.tts_service import TTSService

        svc = TTSService.__new__(TTSService)
        writer = MagicMock(sample_rate=24000)
        pcm = (np.zeros(8, dtype=np.float32)).tobytes()
        with patch("api.src.inference.chatterbox_client.blend",
                   AsyncMock(return_value=(pcm, 24000))) as m_blend, \
             patch("api.src.inference.chatterbox_client.clone",
                   AsyncMock(return_value=(pcm, 24000))) as m_clone, \
             patch("api.src.services.tts_service.AudioNormalizer") as m_norm:
            m_norm.return_value.normalize.side_effect = lambda a: a
            chunks = [
                c async for c in svc.generate_audio_stream(
                    "hi", "", writer, engine="chatterbox",
                    output_format=None, references=references,
                )
            ]
        assert chunks
        return m_blend, m_clone

    async def test_two_references_route_to_blend(self):
        m_blend, m_clone = await self._run_refs([("/tmp/a.wav", 0.3), ("/tmp/b.wav", 0.7)])
        m_blend.assert_awaited_once()
        m_clone.assert_not_awaited()

    async def test_single_reference_routes_to_clone(self):
        # One "blend" reference is just a clone of that clip — no /blend call.
        m_blend, m_clone = await self._run_refs([("/tmp/a.wav", 1.0)])
        m_clone.assert_awaited_once()
        m_blend.assert_not_awaited()


class TestBlendRoute:
    """POST /api/voices/blend — the multi-sample Voice Workshop path. The engine's
    embedding interpolation itself is GPU-only (validated live); here we assert the
    HTTP contract: multiple reference files + parallel weights reach generate_audio
    as a `references` list, the ticket gate holds, and empty clips are rejected."""

    def _blend_multipart(self):
        files = [
            ("reference", ("a.wav", b"RIFFaaaaWAVE", "audio/wav")),
            ("reference", ("b.wav", b"RIFFbbbbWAVE", "audio/wav")),
        ]
        data = {"text": "blend me", "response_format": "wav", "weights": ["0.3", "0.7"]}
        return files, data

    def test_blend_two_refs_returns_audio(self):
        client = TestClient(app)
        svc = SimpleNamespace(
            generate_audio=AsyncMock(return_value=AudioChunk(np.zeros(4, dtype=np.int16)))
        )
        with patch.object(control, "get_tts_service", AsyncMock(return_value=svc)), \
             patch.object(control.AudioService, "convert_audio",
                          AsyncMock(side_effect=[_chunk(b"BODY"), _chunk(b"TAIL")])):
            files, data = self._blend_multipart()
            resp = client.post("/api/voices/blend", files=files, data=data,
                               headers=_auth_headers())
        assert resp.status_code == 200
        assert resp.content == b"BODYTAIL"
        _, kwargs = svc.generate_audio.call_args
        assert kwargs["engine"] == "chatterbox"
        refs = kwargs["references"]
        assert len(refs) == 2
        assert [w for _, w in refs] == [0.3, 0.7]
        assert all(p for p, _ in refs)  # both clips staged to real temp paths

    def test_blend_requires_ticket(self):
        client = TestClient(app)
        files, data = self._blend_multipart()
        resp = client.post("/api/voices/blend", files=files, data=data)
        assert resp.status_code in (401, 403)

    def test_blend_empty_reference_is_400(self):
        client = TestClient(app)
        svc = SimpleNamespace(generate_audio=AsyncMock())
        with patch.object(control, "get_tts_service", AsyncMock(return_value=svc)):
            files = [("reference", ("a.wav", b"", "audio/wav"))]
            data = {"text": "x", "weights": ["1"]}
            resp = client.post("/api/voices/blend", files=files, data=data,
                               headers=_auth_headers())
        assert resp.status_code == 400
