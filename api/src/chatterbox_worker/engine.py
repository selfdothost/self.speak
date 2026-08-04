"""Chatterbox TTS engine adapter (Resemble AI, MIT) — WORKER-venv module.

Ported from the standup scaffold's ``chatterbox_engine.py`` (the one genuinely
new, non-copied module). It owns the lazily-loaded model singleton and
load/unload/generate. Per INTEGRATION-PLAN-v2.md §1.4 / §5 this now lives in the
Chatterbox WORKER venv/process and returns the RAW waveform — all mp3/opus/wav
encoding stays in the MAIN process's ``StreamingAudioWriter`` (the scaffold's
``_encode`` is dropped; main encodes, single-sourced and torch-version-independent).

Gate corrections baked in (INTEGRATION-PLAN-v2.md §3.1):
  * NO ``CHATTERBOX_MODEL`` Turbo/Nano selector — there is no such API.
    ``ChatterboxTTS.from_pretrained(...)`` downloads the single published model.
  * Weights are pulled at runtime from HF Hub into ``HF_HOME`` (a self.speak
    model-cache PVC subPath at deploy), never baked.

Upstream API (resemble-ai/chatterbox), reconciled against chatterbox-tts==0.1.7:

    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device="cuda")
    wav = model.generate(text, audio_prompt_path="ref_clip.wav",
                         exaggeration=..., cfg_weight=...)

``audio_prompt_path`` is the reference clip → zero-shot voice clone (Phase 3);
without it the model synthesises its default voice (Phase 1). ``model.sr`` is the
output sample rate. Weights are MIT (HF ResembleAI/chatterbox).
"""

import logging
import os
import threading
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# The device the worker opens its CUDA context on. Inherited from the container's
# nvidia-runtime-injected GPU time-slice (§1.1). No model-variant selector.
DEVICE = os.environ.get("CHATTERBOX_DEVICE", "cuda")


class ChatterboxEngine:
    """Process-singleton owner of the Chatterbox model. One worker process → one
    instance; all state here is in-process (module singleton)."""

    _instance: "Optional[ChatterboxEngine]" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._sr: Optional[int] = None

    @classmethod
    def instance(cls) -> "ChatterboxEngine":
        # Double-checked lock so a burst of concurrent first-requests loads once.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def peek_loaded(cls) -> bool:
        """Best-effort residency read that NEVER constructs the singleton — for
        the /vram-state probe (an observability read must not have side effects)."""
        inst = cls._instance
        return bool(inst is not None and inst._model is not None)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def sr(self) -> Optional[int]:
        return self._sr

    def load(self) -> None:
        """Load the model into VRAM (idempotent). Lazy: called on first synthesis
        (and again after a release-driven unload). NOT called at worker startup —
        the ~2 GB HF pull must not block the worker's bind (§1.3)."""
        if self._model is not None:
            return
        # Import inside the method so /health and /vram-state don't drag torch +
        # chatterbox in at import time (matches the lazy-inference-import posture).
        from chatterbox.tts import ChatterboxTTS  # type: ignore

        log.info("chatterbox: loading model device=%s", DEVICE)
        self._model = ChatterboxTTS.from_pretrained(device=DEVICE)
        self._sr = int(getattr(self._model, "sr", 24000))
        log.info("chatterbox: model loaded sr=%s", self._sr)

    def unload(self) -> None:
        """Drop the model and free the CUDA cache — the release path's actuator.
        Never raises out; a failed unload is measured by the before/after delta."""
        try:
            self._model = None
            self._sr = None
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:  # pragma: no cover
            log.warning("chatterbox: unload raised (%r); continuing to live-verify", e)

    def generate(
        self, text: str, audio_prompt_path: Optional[str] = None, **controls
    ) -> Tuple[np.ndarray, int]:
        """Synthesise ``text``. With ``audio_prompt_path`` set this is a zero-shot
        clone of that reference clip; without it, the model's default voice.
        ``controls`` forwards optional Chatterbox knobs (``exaggeration``,
        ``cfg_weight``) verbatim. Loads lazily if needed.

        Returns ``(waveform, sample_rate)`` where waveform is a 1-D float32 numpy
        array — the RAW PCM the main process wraps in a single ``AudioChunk`` and
        runs through its encoder. No encoding happens here.
        """
        self.load()
        wav = self._model.generate(text, audio_prompt_path=audio_prompt_path, **controls)
        return self._to_float32_mono(wav), int(self._sr)

    def generate_blend(
        self,
        text: str,
        references: "list[tuple[str, float]]",
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Synthesise ``text`` in a NEW voice interpolated from several reference
        clips. ``references`` is a list of ``(audio_prompt_path, weight)``.

        The blend is done at the CONDITIONING level, not by mixing audio. For each
        clip we run Chatterbox's own ``prepare_conditionals`` to get its
        ``Conditionals(t3, gen)``, then:

          * the SPEAKER EMBEDDINGS — ``t3.speaker_emb`` (voice-encoder identity) and
            ``gen['embedding']`` (s3gen speaker vector) — are the voice's identity,
            fixed-shape per clip, so we weighted-average them across clips. This is
            the actual "blend" knob: 0 = all clip A, 1 = all clip B.
          * the PROSODY PROMPT — ``t3.cond_prompt_speech_tokens`` and the s3gen
            ``prompt_token``/``prompt_feat`` (+ lens) — is a variable-length sequence
            lifted from one clip's audio; different clips give different lengths, so
            these cannot be averaged. We take them wholesale from the highest-weight
            ("dominant") clip, which sets delivery/cadence while the identity is the
            interpolation.

        One reference degrades to a plain single-clip clone. Weights are clamped
        non-negative and renormalised to sum to 1 (all-zero → equal weights).
        Returns ``(waveform float32 mono, sample_rate)`` like ``generate``.
        """
        self.load()
        import torch

        refs = [(p, max(0.0, float(w))) for p, w in references if p]
        if not refs:
            raise ValueError("blend requires at least one reference clip")
        if len(refs) == 1:
            controls: dict = {}
            if exaggeration is not None:
                controls["exaggeration"] = exaggeration
            if cfg_weight is not None:
                controls["cfg_weight"] = cfg_weight
            return self.generate(text, audio_prompt_path=refs[0][0], **controls)

        total = sum(w for _, w in refs)
        weights = [w / total for _, w in refs] if total > 0 else [1.0 / len(refs)] * len(refs)
        exag = 0.5 if exaggeration is None else float(exaggeration)

        # Chatterbox internals — imported here so the module's lazy-import posture
        # (health/vram-state never drag torch+chatterbox in) is preserved. Matched
        # to chatterbox-tts==0.1.7: prepare_conditionals builds a fresh
        # Conditionals(t3_cond, s3gen_ref_dict) and stores it on self._model.conds.
        from chatterbox.models.t3.modules.cond_enc import T3Cond  # type: ignore
        from chatterbox.tts import Conditionals  # type: ignore

        per_clip = []
        for path, _ in refs:
            self._model.prepare_conditionals(path, exaggeration=exag)
            per_clip.append(self._model.conds)  # a distinct object each call

        dominant = max(range(len(weights)), key=lambda i: weights[i])
        base = per_clip[dominant]

        # Weighted sum of the identity embeddings (same shape across clips).
        spk = None
        gen_emb = None
        for w, conds in zip(weights, per_clip):
            s = conds.t3.speaker_emb * w
            g = conds.gen["embedding"] * w
            spk = s if spk is None else spk + s
            gen_emb = g if gen_emb is None else gen_emb + g

        device = self._model.device
        blended_t3 = T3Cond(
            speaker_emb=spk,
            cond_prompt_speech_tokens=base.t3.cond_prompt_speech_tokens,
            emotion_adv=exag * torch.ones(1, 1, 1),
        ).to(device=device)
        blended_gen = dict(base.gen)  # dominant clip's prosody prompt (token/feat/lens)
        blended_gen["embedding"] = gen_emb  # ...but the interpolated speaker vector
        self._model.conds = Conditionals(blended_t3, blended_gen).to(device)

        # audio_prompt_path=None → generate() reuses the blended self._model.conds
        # instead of recomputing from a single clip.
        controls = {"exaggeration": exag}
        if cfg_weight is not None:
            controls["cfg_weight"] = cfg_weight
        wav = self._model.generate(text, audio_prompt_path=None, **controls)
        return self._to_float32_mono(wav), int(self._sr)

    @staticmethod
    def _to_float32_mono(wav) -> np.ndarray:
        """Normalise the model output (torch.Tensor shape (1,N) or (N,), or a
        numpy array) to a contiguous 1-D float32 numpy array."""
        try:
            import torch

            if isinstance(wav, torch.Tensor):
                wav = wav.detach().to("cpu", dtype=torch.float32).numpy()
        except Exception:  # pragma: no cover - torch always present in worker venv
            pass
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(arr, dtype=np.float32)
