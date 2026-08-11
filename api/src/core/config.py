import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = True  # Whether to use GPU acceleration if available
    device_type: str | None = (
        None  # Will be auto-detected if None, can be "cuda", "mps", or "cpu"
    )
    allow_local_voice_saving: bool = (
        False  # Whether to allow saving combined voices locally
    )

    # Container absolute paths
    model_dir: str = "/app/api/src/models"  # Absolute path in container
    voices_dir: str = "/app/api/src/voices/v1_0"  # Absolute path in container

    # ── Chatterbox WORKER (second engine, sibling process) ──────────────────
    # The main process proxies Chatterbox synthesis to a sibling worker over
    # localhost (INTEGRATION-PLAN-v2.md §1.4). These are all INTERNAL to the
    # container — core never talks to the worker; it only ever talks to main.
    #
    # NOTE: there is deliberately NO CHATTERBOX_MODEL selector — there is no
    # Turbo/Nano API; ChatterboxTTS.from_pretrained() downloads the one model.
    #
    # Whether Chatterbox is part of THIS deployment's VRAM footprint (env:
    # CHATTERBOX_ENABLED). This is the gate for the Phase-2 cross-process VRAM
    # aggregation, NOT for whether the worker process launches (the entrypoint
    # starts the worker whenever its venv exists). It answers one question for the
    # VRAM lease: "should self.speak's held_vram_bytes include a Chatterbox
    # worker's slice?" Default False so a Kokoro-only deploy — and the whole
    # pre-Phase-1-deploy window — behaves EXACTLY as the single-engine lease did:
    # the aggregator never touches the worker, and an absent/dark worker can never
    # collapse the working Kokoro lease to "unreachable". Set True on the GPU
    # sibling (speak-gpu.yaml) once the worker is deployed and measured. When True,
    # an EXPECTED-but-unaccountable worker collapses held to null/unreachable
    # rather than under-reporting it (the self.ai#74 over-grant).
    chatterbox_enabled: bool = False
    chatterbox_control_url: str = (
        "http://127.0.0.1:8881"  # worker's localhost control API (env: CHATTERBOX_CONTROL_URL)
    )

    # Kokoro WORKER cutover (self.speak#5 P3). When True, main proxies every
    # generation to the sibling worker and never imports kokoro or allocates on
    # the GPU -- which is the entire point: main is the pod's sole liveness path,
    # so it can never exit, so its ~470 MiB CUDA primary context was permanently
    # unreclaimable. Moving generation out means main never creates one.
    #
    # MUST be set together with KOKORO_WORKER_ENABLED on the entrypoint. They are
    # ONE switch in two places: entrypoint-only gives you TWO contexts (worker +
    # main) and is strictly worse than before, while this-only gives you a main
    # proxying to a worker nobody started.
    kokoro_worker_enabled: bool = False
    kokoro_worker_url: str = (
        "http://127.0.0.1:8882"  # worker's localhost API (env: KOKORO_WORKER_URL)
    )
    chatterbox_voices_dir: str = (
        "/app/api/src/chatterbox_voices"  # reference-clip staging (clone route, Phase 3)
    )
    # Chatterbox's native output sample rate. The worker reports the real rate per
    # request via X-Sample-Rate; this is the rate the main process builds its
    # encoder at, and a mismatch is logged (chatterbox-tts==0.1.7 is 24 kHz).
    chatterbox_sample_rate: int = 24000
    # HF Hub cache home for the worker's runtime weight pull (~2 GB, NOT baked).
    # A self.speak model-cache PVC subPath at deploy (self.transcribe's pattern).
    # Exported into the container env so huggingface_hub in the worker honours it.
    hf_home: str = "/app/.cache/huggingface"  # env: HF_HOME

    # Audio Settings
    sample_rate: int = 24000
    default_volume_multiplier: float = 1.0
    # Text Processing Settings
    target_min_tokens: int = 175  # Target minimum tokens per chunk
    target_max_tokens: int = 250  # Target maximum tokens per chunk
    absolute_max_tokens: int = 450  # Absolute maximum tokens per chunk
    advanced_text_normalization: bool = True  # Preproesses the text before misiki
    voice_weight_normalization: bool = (
        True  # Normalize the voice weights so they add up to 1
    )

    gap_trim_ms: int = (
        1  # Base amount to trim from streaming chunk ends in milliseconds
    )
    dynamic_gap_trim_padding_ms: int = 410  # Padding to add to dynamic gap trim
    dynamic_gap_trim_padding_char_multiplier: dict[str, float] = {
        ".": 1,
        "!": 0.9,
        "?": 1,
        ",": 0.8,
    }

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files
    cors_origins: list[str] = ["*"]  # CORS origins for web player
    cors_enabled: bool = True  # Whether to enable CORS

    # Temp File Settings for WEB Ui
    temp_file_dir: str = "api/temp_files"  # Directory for temporary audio files (relative to project root)
    max_temp_dir_size_mb: int = 2048  # Maximum size of temp directory (2GB)
    max_temp_dir_age_hours: int = 1  # Remove temp files older than 1 hour
    max_temp_dir_count: int = 3  # Maximum number of temp files to keep

    class Config:
        env_file = ".env"

    def get_device(self) -> str:
        """Get the appropriate device based on settings and availability"""
        if not self.use_gpu:
            return "cpu"

        if self.device_type:
            return self.device_type

        # Auto-detect device
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


settings = Settings()
