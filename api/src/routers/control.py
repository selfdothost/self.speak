"""Control-port endpoints for the self.ai UI voice catalog.

self.ai's UI reaches self.speak on two logical surfaces that happen to share one
port: the OpenAI-compatible *serving* surface (``/v1/audio/speech``) and a
*control* surface the UI's typed-connection voice catalog reads to discover which
voices this backend actually offers. The self.ai transcribe/voice-catalog routers
address that control surface as ``{control_base_url}/api/voices`` (the STT/TTS
analog of self.llamolotl's ``/api/models`` control port), so this router exposes
exactly that path.

It is a thin alias over the same voice list the OpenAI-compatible
``GET /v1/audio/voices`` returns — one source of truth (``TTSService.list_voices``),
two addresses — so the control view can never drift from what the serving surface
can synthesize.
"""

import os
import tempfile

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from loguru import logger

from ..core.auth import SCOPE_AUDIO_SYNTHESIZE, require_scope
from ..core.config import settings
from ..inference.base import AudioChunk
from ..inference.chatterbox_client import ChatterboxUnavailable
from ..services.audio import AudioService
from ..services.streaming_audio_writer import StreamingAudioWriter
from ..structures.schemas import CloneSpeechRequest
from .openai_compatible import engine_sample_rate, get_tts_service

router = APIRouter(tags=["control"])

# The audition text the preview route falls back to when the caller sends no text
# of its own — a short pangram-ish line that exercises a range of phonemes.
_PREVIEW_DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."

# Reference-clip guardrails. self.speak stages the clip only to a per-request temp
# file it deletes in a finally; persistent storage is self.ai's job (Phase 4).
_MAX_REFERENCE_BYTES = 25 * 1024 * 1024  # 25 MiB — comfortably over any real clip
_CONTENT_TYPE = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


async def _clone_to_response(reference: UploadFile, req: CloneSpeechRequest) -> Response:
    """Shared clone/preview body: materialise the reference clip to a temp path,
    synthesise ``req.text`` in its voice via the Chatterbox worker (through
    ``TTSService.generate_audio`` → ``track_synthesis`` → the localhost proxy),
    encode with the shared ``StreamingAudioWriter``, and return the full file.

    Stateless by design: the temp clip is always deleted in the ``finally``, so no
    biometric-ish reference audio is retained on this backend.
    """
    data = await reference.read()
    if not data:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "Empty reference clip",
                    "type": "invalid_request_error"},
        )
    if len(data) > _MAX_REFERENCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"error": "payload_too_large",
                    "message": f"Reference clip exceeds {_MAX_REFERENCE_BYTES} bytes",
                    "type": "invalid_request_error"},
        )

    # Preserve the clip's extension so the worker's loader can sniff the container.
    _, ext = os.path.splitext(reference.filename or "")
    staging_dir = settings.chatterbox_voices_dir
    try:
        os.makedirs(staging_dir, exist_ok=True)
    except OSError:
        staging_dir = None  # fall back to the system temp dir
    fd, tmp_path = tempfile.mkstemp(suffix=ext or ".wav", dir=staging_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)

        content_type = _CONTENT_TYPE.get(
            req.response_format, f"audio/{req.response_format}"
        )
        writer = StreamingAudioWriter(
            req.response_format, sample_rate=engine_sample_rate("chatterbox")
        )
        try:
            tts_service = await get_tts_service()
            audio = await tts_service.generate_audio(
                text=req.text,
                voice="",  # unused for Chatterbox (reference clip is the voice)
                writer=writer,
                engine="chatterbox",
                audio_prompt_path=tmp_path,
                volume_multiplier=req.volume_multiplier,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
            )
            body = await AudioService.convert_audio(
                audio, req.response_format, writer,
                is_last_chunk=False, trim_audio=False,
            )
            final = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                req.response_format, writer, is_last_chunk=True,
            )
            output = body.output + final.output
        finally:
            writer.close()
    except ChatterboxUnavailable as e:
        # The worker is down/absent/mid-respawn — a clean 503, never a hang.
        logger.warning(f"Chatterbox worker unavailable for clone: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "engine_unavailable",
                    "message": "The Chatterbox voice engine is not available",
                    "type": "server_error"},
        ) from e
    finally:
        # Stateless: never retain the reference clip.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return Response(
        content=output,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename=voice.{req.response_format}",
                 "Cache-Control": "no-cache"},
    )


async def _blend_to_response(
    references: "list[UploadFile]", weights: "list[float]", req: CloneSpeechRequest
) -> Response:
    """Multi-clip blend body: materialise EACH reference clip to its own temp path,
    synthesise ``req.text`` in the voice interpolated across them (Chatterbox worker
    ``/blend`` — weighted speaker-embedding blend), encode, and return the file.

    Same stateless guarantee as ``_clone_to_response``: every temp clip is deleted
    in the ``finally``, so no reference audio is retained. ``weights`` is parallel to
    ``references``; a missing entry defaults to 1.0 and the worker renormalises."""
    if not references:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_request", "message": "No reference clips",
                    "type": "invalid_request_error"},
        )

    staging_dir = settings.chatterbox_voices_dir
    try:
        os.makedirs(staging_dir, exist_ok=True)
    except OSError:
        staging_dir = None  # fall back to the system temp dir

    tmp_paths: list[str] = []
    output = b""
    try:
        refs: list[tuple[str, float]] = []
        for i, ref in enumerate(references):
            data = await ref.read()
            if not data:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid_request", "message": "Empty reference clip",
                            "type": "invalid_request_error"},
                )
            if len(data) > _MAX_REFERENCE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={"error": "payload_too_large",
                            "message": f"Reference clip exceeds {_MAX_REFERENCE_BYTES} bytes",
                            "type": "invalid_request_error"},
                )
            _, ext = os.path.splitext(ref.filename or "")
            fd, tmp_path = tempfile.mkstemp(suffix=ext or ".wav", dir=staging_dir)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            tmp_paths.append(tmp_path)
            refs.append((tmp_path, float(weights[i]) if i < len(weights) else 1.0))

        content_type = _CONTENT_TYPE.get(
            req.response_format, f"audio/{req.response_format}"
        )
        writer = StreamingAudioWriter(
            req.response_format, sample_rate=engine_sample_rate("chatterbox")
        )
        try:
            tts_service = await get_tts_service()
            audio = await tts_service.generate_audio(
                text=req.text,
                voice="",
                writer=writer,
                engine="chatterbox",
                references=refs,
                volume_multiplier=req.volume_multiplier,
                exaggeration=req.exaggeration,
                cfg_weight=req.cfg_weight,
            )
            body = await AudioService.convert_audio(
                audio, req.response_format, writer,
                is_last_chunk=False, trim_audio=False,
            )
            final = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                req.response_format, writer, is_last_chunk=True,
            )
            output = body.output + final.output
        finally:
            writer.close()
    except ChatterboxUnavailable as e:
        logger.warning(f"Chatterbox worker unavailable for blend: {e}")
        raise HTTPException(
            status_code=503,
            detail={"error": "engine_unavailable",
                    "message": "The Chatterbox voice engine is not available",
                    "type": "server_error"},
        ) from e
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    return Response(
        content=output,
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename=voice.{req.response_format}",
                 "Cache-Control": "no-cache"},
    )


@router.get("/api/voices")
async def list_control_voices():
    """List this backend's real voices for the self.ai UI voice catalog.

    Returns ``{"voices": [<voice id>, ...]}`` — the same shape and the same
    underlying list as ``GET /v1/audio/voices``, at the ``/api/voices`` control
    path the UI's ``resolve_tts_control_url`` fetch targets.
    """
    try:
        tts_service = await get_tts_service()
        voices = await tts_service.list_voices()
        return {"voices": voices}
    except Exception as e:
        logger.error(f"Error listing control voices: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "server_error",
                "message": "Failed to retrieve voice list",
                "type": "server_error",
            },
        )


@router.post("/api/voices/clone")
async def clone_voice(
    reference: UploadFile = File(..., description="Reference voice clip (audio)"),
    text: str = Form(..., description="Text to synthesise in the reference voice"),
    response_format: str = Form("wav"),
    exaggeration: float | None = Form(None),
    cfg_weight: float | None = Form(None),
    volume_multiplier: float = Form(1.0),
    _auth: dict = Depends(require_scope(SCOPE_AUDIO_SYNTHESIZE)),
) -> Response:
    """Zero-shot clone: synthesise ``text`` in the voice of the uploaded reference
    clip (INTEGRATION-PLAN-v2.md Phase 3). Ticket-gated ``audio:synthesize`` — the
    v0 reuse; a dedicated ``voice:clone`` scope is a deferred maintainer decision.

    Multipart form: ``reference`` (file) + the ``CloneSpeechRequest`` fields.
    Returns the encoded audio file. Stateless — the reference clip is not stored.
    """
    req = CloneSpeechRequest(
        text=text, response_format=response_format, exaggeration=exaggeration,
        cfg_weight=cfg_weight, volume_multiplier=volume_multiplier,
    )
    return await _clone_to_response(reference, req)


@router.post("/api/voices/preview")
async def preview_voice(
    reference: UploadFile = File(..., description="Reference voice clip (audio)"),
    text: str | None = Form(None, description="Optional audition text"),
    response_format: str = Form("wav"),
    exaggeration: float | None = Form(None),
    cfg_weight: float | None = Form(None),
    volume_multiplier: float = Form(1.0),
    _auth: dict = Depends(require_scope(SCOPE_AUDIO_SYNTHESIZE)),
) -> Response:
    """Quick audition of a reference clip's voice — identical mechanics to
    ``/clone`` but ``text`` is optional and falls back to a standard audition line
    (INTEGRATION-PLAN-v2.md Phase 3, the Sound Studio Preview node). Stateless.
    """
    req = CloneSpeechRequest(
        text=text or _PREVIEW_DEFAULT_TEXT, response_format=response_format,
        exaggeration=exaggeration, cfg_weight=cfg_weight,
        volume_multiplier=volume_multiplier,
    )
    return await _clone_to_response(reference, req)


@router.post("/api/voices/blend")
async def blend_voice(
    reference: list[UploadFile] = File(..., description="Reference clips to blend (2 or more)"),
    weights: list[float] = Form([], description="Per-clip blend weights, parallel to reference (equal if omitted)"),
    text: str | None = Form(None, description="Optional audition text"),
    response_format: str = Form("wav"),
    exaggeration: float | None = Form(None),
    cfg_weight: float | None = Form(None),
    volume_multiplier: float = Form(1.0),
    _auth: dict = Depends(require_scope(SCOPE_AUDIO_SYNTHESIZE)),
) -> Response:
    """Blend several reference clips into a NEW voice and audition ``text`` in it —
    the Voice Workshop's multi-sample path (INTEGRATION-PLAN-v2.md Phase 3+). The
    worker interpolates the clips' speaker embeddings by ``weights``; delivery is
    taken from the highest-weight clip. Same ticket gate + statelessness as
    ``/clone``/``/preview``. A single reference degrades to a plain clone.

    Multipart form: repeated ``reference`` files + parallel ``weights`` + the
    ``CloneSpeechRequest`` fields. Returns the encoded audio file.
    """
    req = CloneSpeechRequest(
        text=text or _PREVIEW_DEFAULT_TEXT, response_format=response_format,
        exaggeration=exaggeration, cfg_weight=cfg_weight,
        volume_multiplier=volume_multiplier,
    )
    return await _blend_to_response(reference, weights, req)
