from email.policy import default
from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class VoiceCombineRequest(BaseModel):
    """Request schema for voice combination endpoint that accepts either a string with + or a list"""

    voices: Union[str, List[str]] = Field(
        ...,
        description="Either a string with voices separated by + (e.g. 'voice1+voice2') or a list of voice names to combine",
    )


class TTSStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"  # For files removed by cleanup


# OpenAI-compatible schemas
class WordTimestamp(BaseModel):
    """Word-level timestamp information"""

    word: str = Field(..., description="The word or token")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")


class CaptionedSpeechResponse(BaseModel):
    """Response schema for captioned speech endpoint"""

    audio: str = Field(..., description="The generated audio data encoded in base 64")
    audio_format: str = Field(..., description="The format of the output audio")
    timestamps: Optional[List[WordTimestamp]] = Field(
        ..., description="Word-level timestamps"
    )


class NormalizationOptions(BaseModel):
    """Options for the normalization system"""

    normalize: bool = Field(
        default=True,
        description="Normalizes input text to make it easier for the model to say",
    )
    unit_normalization: bool = Field(
        default=False, description="Transforms units like 10KB to 10 kilobytes"
    )
    url_normalization: bool = Field(
        default=True,
        description="Changes urls so they can be properly pronounced by kokoro",
    )
    email_normalization: bool = Field(
        default=True,
        description="Changes emails so they can be properly pronouced by kokoro",
    )
    optional_pluralization_normalization: bool = Field(
        default=True,
        description="Replaces (s) with s so some words get pronounced correctly",
    )
    phone_normalization: bool = Field(
        default=True,
        description="Changes phone numbers so they can be properly pronouced by kokoro",
    )
    replace_remaining_symbols: bool = Field(
        default=True,
        description="Replaces the remaining symbols after normalization with their words"
    )


class OpenAISpeechRequest(BaseModel):
    """Request schema for OpenAI-compatible speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    download_format: Optional[Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]] = (
        Field(
            default=None,
            description="Optional different format for the final download. If not provided, uses response_format.",
        )
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default = 1.0,
        description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )
    # ── Chatterbox-only controls (ignored by the Kokoro engine) ─────────────
    # Optional knobs forwarded verbatim to the Chatterbox worker when
    # model=chatterbox resolves to that engine (INTEGRATION-PLAN-v2.md §1.4 /
    # §5). Left None for Kokoro requests, which never see them.
    exaggeration: Optional[float] = Field(
        default=None,
        description="Chatterbox only: emotion/intensity exaggeration knob. Ignored by Kokoro.",
    )
    cfg_weight: Optional[float] = Field(
        default=None,
        description="Chatterbox only: classifier-free-guidance weight / pacing knob. Ignored by Kokoro.",
    )


class VramStateResponse(BaseModel):
    """R1 — the reply body of ``GET /api/system/vram-state``.

    Reports the VRAM this self.speak process currently holds and its total
    addressable GPU capacity so core's lease registry can track this instance.
    All figures are **bytes**. The null-vs-0 semantics are load-bearing and
    encode the R1-AC4 tri-state (a caller distinguishes "genuinely near-zero"
    from "unknown" via ``status``, never by guessing at a null):

    * (a) **GPU reachable & probed** → ``held_vram_bytes`` /
      ``total_capacity_bytes`` are real ``int`` (held near-zero, NOT null, when
      the model is not resident); ``status="ok"``, ``gpu_reachable=True``.
    * (b) **GPU present but unprobable/unreachable** (a ``torch.cuda`` call
      raised — driver shadowing / CUDA-init failure) → both fields serialise to
      JSON ``null`` (NEVER 0 — a false zero would let core over-grant VRAM that
      is not actually free, R1-AC5); ``status="unreachable"``,
      ``gpu_reachable=False``.
    * (c) **no CUDA / CPU deployment** → both fields are the integer ``0``
      (a known-zero, no-op consumer, distinct from case (b)'s null);
      ``status="no_gpu"``, ``gpu_reachable=False``.

    ``Optional[int]`` (not ``int``) is deliberate: a plain ``int`` field would
    forbid the case-(b) null and force the very false-zero R1-AC5 bans.

    ``held_vram_bytes`` means: **the VRAM this process would give back if asked
    to fully release** — ``torch.cuda.memory_reserved()``, which is precisely
    what our own release path's ``empty_cache()`` returns to the driver. It is
    NOT ``memory_allocated()`` (live-tensor bytes only, blind to the reserved
    pool and so an under-report core would over-grant against, self.ai#74), and
    it is NOT the whole card — that is ``device_used_bytes``, reported
    separately below because core SUMS held across consumers.
    """

    held_vram_bytes: Optional[int] = Field(
        ...,
        description=(
            "Currently-held VRAM in BYTES, computed live per request. int when "
            "the GPU is reachable (incl. near-zero when the model is not "
            "resident); null when GPU-present-but-unprobable (case b, unknown — "
            "never 0); 0 on a no-CUDA/CPU deployment (case c)."
        ),
    )
    total_capacity_bytes: Optional[int] = Field(
        ...,
        description=(
            "Total addressable GPU capacity in BYTES, computed live per "
            "request. Same tri-state as held_vram_bytes: int when reachable, "
            "null when unprobable (case b), 0 on CPU (case c)."
        ),
    )
    gpu_reachable: bool = Field(
        ...,
        description=(
            "True only when a live torch.cuda probe succeeded (case a). False "
            "for both the unreachable (b) and no-CUDA (c) states."
        ),
    )
    status: str = Field(
        ...,
        description=(
            "Tri-state discriminator: 'ok' (case a, real ints), 'unreachable' "
            "(case b, null figures), or 'no_gpu' (case c, integer-0 figures)."
        ),
    )
    model_resident: bool = Field(
        ...,
        description=(
            "Whether the Kokoro backend is currently loaded in this process "
            "(observability; not part of the byte accounting)."
        ),
    )
    device_used_bytes: Optional[int] = Field(
        default=None,
        description=(
            "WHOLE-CARD VRAM in use in BYTES, every process included — a "
            "DIFFERENT quantity from held_vram_bytes and never to be summed "
            "with it (self.ai#74). Core sums held_vram_bytes across consumers, "
            "so a whole-card figure in that field would double-count every "
            "sibling; this one lets core account for CUDA contexts and "
            "processes that are not lease consumers at all. null whenever the "
            "device could not be read (cases b AND c) — with no CUDA we cannot "
            "see the card, and 0 would falsely assert an empty one."
        ),
    )
    device_total_bytes: Optional[int] = Field(
        default=None,
        description=(
            "Total card size in BYTES accompanying device_used_bytes. Equal to "
            "total_capacity_bytes in case (a); null whenever the device could "
            "not be read."
        ),
    )


class VramReleaseRequest(BaseModel):
    """R2 — the request body of ``POST /api/system/vram-release``.

    Amount-based only, matching core's outbound body verbatim
    (``{"target_bytes": int(...), "timeout_seconds": float(...)}``). There is
    deliberately **NO mechanism field**: self.speak decides HOW to free, so a
    future multi-engine self.speak (Chatterbox / GPT-SoVITS resident alongside
    Kokoro) is a same-protocol upgrade, not a wire change.
    """

    target_bytes: int = Field(
        ...,
        description="Target amount of VRAM to free, in BYTES.",
    )
    timeout_seconds: float = Field(
        ...,
        description=(
            "Wall-clock budget: the handler drains in-flight synthesis bounded "
            "by this and NEVER blocks past it."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "E-STOP semantics. Default False = the cooperative lease release "
            "(drain in-flight synthesis first, bounded by timeout_seconds). True "
            "= unload NOW regardless of in-flight work: skip the drain-wait and "
            "force-unload EVERY engine (Kokoro + the Chatterbox worker) "
            "immediately. Set only by the admin 'Unload All Models' e-stop — "
            "'stop now, short of pulling the plug' — never by the routine "
            "priority-based broker path. Not a mechanism field: self.speak still "
            "decides HOW to free; force only decides whether to wait."
        ),
    )


class VramReleaseResponse(BaseModel):
    """R2 — the reply body of ``POST /api/system/vram-release``.

    Core's ``_map_response`` treats ``status`` ∈ {released, partial, confirmed}
    AND ``isinstance(freed_bytes, int) and not isinstance(freed_bytes, bool)``
    as a confirmed release; an HTTP 409 or in-band ``status == "busy"`` is read
    as "not now" (never a denial). self.speak emits only ``released`` (target
    met), ``partial`` (a real but smaller/zero freed), or ``busy`` (single-flight
    reject).

    ``freed_bytes`` is typed ``int`` (pydantic) so a Python ``bool`` cannot
    survive to the wire — core explicitly rejects a boolean ``freed_bytes`` as a
    mistyped confirmation (R2-AC4). It is always a live-measured figure, never
    inferred from the fact that an unload call returned.
    """

    status: Literal["released", "partial", "busy"] = Field(
        ...,
        description=(
            "'released' = target met; 'partial' = a real but smaller (often 0) "
            "amount freed, or drain-deadline hit with nothing safely freed; "
            "'busy' = a release is already in flight (single-flight reject)."
        ),
    )
    freed_bytes: int = Field(
        ...,
        description=(
            "VRAM freed, in BYTES, measured via a live torch.cuda before/after "
            "delta — a JSON integer, NEVER a boolean."
        ),
    )


class CaptionedSpeechRequest(BaseModel):
    """Request schema for captioned speech endpoint"""

    model: str = Field(
        default="kokoro",
        description="The model to use for generation. Supported models: tts-1, tts-1-hd, kokoro",
    )
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(
        default="af_heart",
        description="The voice to use for generation. Can be a base voice or a combined voice name.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="The format to return audio in. Supported formats: mp3, opus, flac, wav, pcm. PCM format returns raw 16-bit samples without headers. AAC is not currently supported.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="The speed of the generated audio. Select a value from 0.25 to 4.0.",
    )
    stream: bool = Field(
        default=True,  # Default to streaming for OpenAI compatibility
        description="If true (default), audio will be streamed as it's generated. Each chunk will be a complete sentence.",
    )
    return_timestamps: bool = Field(
        default=True,
        description="If true (default), returns word-level timestamps in the response",
    )
    return_download_link: bool = Field(
        default=False,
        description="If true, returns a download link in X-Download-Path header after streaming completes",
    )
    lang_code: Optional[str] = Field(
        default=None,
        description="Optional language code to use for text processing. If not provided, will use first letter of voice name.",
    )
    volume_multiplier: Optional[float] = Field(
        default = 1.0,
        description="A volume multiplier to multiply the output audio by."
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=NormalizationOptions(),
        description="Options for the normalization system",
    )


class CloneSpeechRequest(BaseModel):
    """Field contract for the Chatterbox clone/preview control routes
    (``POST /api/voices/clone`` and ``/api/voices/preview``, INTEGRATION-PLAN-v2.md
    Phase 3). The reference audio itself rides the request as a multipart file, so
    these routes bind the non-file fields via ``Form(...)`` and validate them
    through this model — one source of truth for the accepted params.

    self.speak's clone surface is deliberately STATELESS: it takes reference bytes
    + text and returns audio. Persistent reference-clip storage and the voice asset
    live in self.ai (the Phase-4 connector + Garage), never on this backend.
    """

    text: str = Field(
        ...,
        min_length=1,
        description="Text to synthesise in the reference clip's voice.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="wav",
        description="Audio output format (same set as /v1/audio/speech).",
    )
    exaggeration: Optional[float] = Field(
        default=None,
        description="Chatterbox emotion-exaggeration control (engine default when null).",
    )
    cfg_weight: Optional[float] = Field(
        default=None,
        description="Chatterbox classifier-free-guidance weight (engine default when null).",
    )
    volume_multiplier: Optional[float] = Field(
        default=1.0,
        description="A volume multiplier to multiply the output audio by.",
    )
