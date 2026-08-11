"""Kokoro V1 model management."""

import asyncio
import sys
from typing import Optional

from loguru import logger

from ..core import paths
from ..core.config import settings
from ..core.model_config import ModelConfig, model_config
from .base import BaseModelBackend

# Set by the WORKER process at startup. Not read from the environment on
# purpose: main and the worker share KOKORO_WORKER_ENABLED (Settings has no
# env_prefix, so settings.kokoro_worker_enabled reads the same variable the
# entrypoint gates the worker launch on). That sharing is what made the first
# enablement attempt fail -- with the flag true the WORKER also selected
# KokoroWorkerBackend and would have proxied to ITSELF on 127.0.0.1:8882.
#
# The two processes need OPPOSITE answers to "should I proxy?", so the answer
# cannot come from a value they both see. A process knows what it is; it should
# say so explicitly rather than infer it.
_FORCE_IN_PROCESS = False


def force_in_process_backend() -> None:
    """Declare THIS process the one that owns the model. Called by the worker."""
    global _FORCE_IN_PROCESS
    _FORCE_IN_PROCESS = True


class ModelManager:
    """Manages Kokoro V1 model loading and inference."""

    # Singleton instance
    _instance = None

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize manager.

        Args:
            config: Optional model configuration override
        """
        self._config = config or model_config
        # Deliberately untyped as KokoroV1: under the P3 cutover this holds a
        # KokoroWorkerBackend instead, and naming the concrete class here was
        # what made the rest of the codebase assume it.
        self._backend: Optional[BaseModelBackend] = None
        self._device: Optional[str] = None
        # Serializes lazy (re)loads so concurrent requests arriving after an
        # unload trigger at most ONE reload. See ensure_loaded().
        self._reload_lock = asyncio.Lock()

    def _determine_device(self) -> str:
        """Determine device based on settings."""
        return "cuda" if settings.use_gpu else "cpu"

    async def initialize(self) -> None:
        """Initialize the Kokoro backend -- in-process, or the worker (P3).

        The kokoro import is INSIDE the branch on purpose. This process is the
        pod's sole liveness path, so it can never exit, so any CUDA primary
        context it creates (~470 MiB on the 4090) is unreclaimable forever. Under
        the cutover main must therefore never touch the GPU at all -- and the
        cheapest way to be sure of that is to never import the thing that would.
        """
        try:
            self._device = self._determine_device()

            if settings.kokoro_worker_enabled and not _FORCE_IN_PROCESS:
                # NOTE: no `from .kokoro_v1 import KokoroV1` on this path, and
                # that is the feature, not an optimisation.
                from .kokoro_client import KokoroWorkerBackend

                logger.info(
                    "Kokoro runs in the WORKER process (%s); this process will not "
                    "load a model or allocate on the GPU",
                    settings.kokoro_worker_url,
                )
                self._backend = KokoroWorkerBackend()
                return

            # Resolve through THIS module rather than `from .kokoro_v1 import
            # KokoroV1`. A function-local import binds the real class directly
            # and would silently bypass patch("...model_manager.KokoroV1") -- the
            # patch would resolve without error and simply never be used, which
            # is worse than an ImportError because the test still looks wired up.
            # Attribute access here hits the module __getattr__ below when
            # unpatched, so the import stays lazy either way.
            backend_cls = getattr(sys.modules[__name__], "KokoroV1")

            logger.info(f"Initializing Kokoro V1 on {self._device}")
            self._backend = backend_cls()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Kokoro V1: {e}")

    async def initialize_with_warmup(self, voice_manager) -> tuple[str, str, int]:
        """Initialize and warm up model.

        Args:
            voice_manager: Voice manager instance for warmup

        Returns:
            Tuple of (device, backend type, voice count)

        Raises:
            RuntimeError: If initialization fails
        """
        import time

        start = time.perf_counter()

        try:
            # Initialize backend
            await self.initialize()

            # Load model
            model_path = self._config.pytorch_kokoro_v1_file
            await self.load_model(model_path)

            # Use paths module to get voice path
            try:
                voices = await paths.list_voices()
                voice_path = await paths.get_voice_path(settings.default_voice)

                if settings.kokoro_worker_enabled and not _FORCE_IN_PROCESS:
                    # NO warmup generation when the model lives in the worker,
                    # for two independent reasons:
                    #
                    # 1. It would fail startup. The entrypoint launches the
                    #    worker in the background and then execs main, so main
                    #    can reach this line before the worker has bound its
                    #    socket -- turning a race into a CrashLoop.
                    # 2. It would be wrong even if it won the race. The worker
                    #    lazy-loads on first use precisely so it can step aside
                    #    on a forced VRAM release and come back cheaply;
                    #    reaching across to force a load at startup defeats that.
                    logger.info(
                        "Skipping warmup: Kokoro lives in the worker and loads lazily"
                    )
                else:
                    # Warm up with short text
                    warmup_text = "Warmup text for initialization."
                    # Use default voice name for warmup
                    voice_name = settings.default_voice
                    logger.debug(f"Using default voice '{voice_name}' for warmup")
                    async for _ in self.generate(warmup_text, (voice_name, voice_path)):
                        pass
            except Exception as e:
                raise RuntimeError(f"Failed to get default voice: {e}")

            ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Warmup completed in {ms}ms")

            return self._device, "kokoro_v1", len(voices)
        except FileNotFoundError as e:
            logger.error("""
Model files not found! You need to download the Kokoro V1 model:

1. Download model using the script:
   python docker/scripts/download_model.py --output api/src/models/v1_0

2. Or set environment variable in docker-compose:
   DOWNLOAD_MODEL=true
""")
            exit(0)
        except Exception as e:
            raise RuntimeError(f"Warmup failed: {e}")

    async def ensure_loaded(self) -> None:
        """Lazily (re)load the model if the backend is not resident.

        The backend is loaded once at startup by ``initialize_with_warmup``, but
        a VRAM-lease release (``unload_all()``, driven by self.ai's GPU broker
        when another service needs the shared card) drops it to free VRAM. Without
        this, the next synthesis after a release fails permanently with "Backend
        not initialized" until the pod restarts — so the serving path calls this
        first to transparently cold-reload (the accepted latency cost of having
        yielded the VRAM; ~a few seconds while the model reloads).

        No-op in the common case (backend already resident). The reload itself
        skips the startup warmup synthesis — the real request that triggered it
        IS the warmup. Serialized by ``_reload_lock`` with a double-check so a
        burst of requests after a release causes exactly one reload; the drain
        guard in the release path (it waits for in-flight synthesis to finish
        before unloading) plus this lock keep unload and reload from racing.
        """
        if self._backend is not None:
            return
        async with self._reload_lock:
            if self._backend is not None:  # another coroutine reloaded while we waited
                return
            logger.info(
                "Backend not resident (cold start or post-VRAM-release) — reloading Kokoro"
            )
            await self.initialize()
            await self.load_model(self._config.pytorch_kokoro_v1_file)
            logger.info("Kokoro backend reloaded and ready")

    def get_backend(self) -> BaseModelBackend:
        """Get initialized backend.

        Returns:
            Initialized backend instance

        Raises:
            RuntimeError: If backend not initialized
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")
        return self._backend

    async def load_model(self, path: str) -> None:
        """Load model using initialized backend.

        Args:
            path: Path to model file

        Raises:
            RuntimeError: If loading fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            await self._backend.load_model(path)
        except FileNotFoundError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    async def generate(self, *args, **kwargs):
        """Generate audio using initialized backend.

        Raises:
            RuntimeError: If generation fails
        """
        if not self._backend:
            raise RuntimeError("Backend not initialized")

        try:
            async for chunk in self._backend.generate(*args, **kwargs):
                if settings.default_volume_multiplier != 1.0:
                    chunk.audio *= settings.default_volume_multiplier
                yield chunk
        except Exception as e:
            # `from e` so the original type and traceback survive being flattened
            # into RuntimeError. Without it a CUDA OOM, a voice-load failure and a
            # tensor-shape bug are indistinguishable to everything downstream.
            raise RuntimeError(f"Generation failed: {e}") from e

    def unload_all(self) -> None:
        """Unload model and free resources."""
        if self._backend:
            self._backend.unload()
            self._backend = None

    @property
    def current_backend(self) -> str:
        """Get current backend type."""
        return "kokoro_v1"


async def get_manager(config: Optional[ModelConfig] = None) -> ModelManager:
    """Get model manager instance.

    Args:
        config: Optional configuration override

    Returns:
        ModelManager instance
    """
    if ModelManager._instance is None:
        ModelManager._instance = ModelManager(config)
    return ModelManager._instance


def __getattr__(name):
    """Expose ``KokoroV1`` lazily at module level (PEP 562).

    The top-level import was removed so a worker-mode main never drags in kokoro
    (self.speak#5 P3), but ``patch("...model_manager.KokoroV1")`` is an
    established test seam and there is no reason to break it. Attribute access
    imports on demand; mock's setattr/delattr cycle then works normally, since
    the name is absent from this module's __dict__ until something patches it.
    """
    if name == "KokoroV1":
        from .kokoro_v1 import KokoroV1

        return KokoroV1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
