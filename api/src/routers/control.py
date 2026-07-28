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

from fastapi import APIRouter, HTTPException
from loguru import logger

from .openai_compatible import get_tts_service

router = APIRouter(tags=["control"])


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
