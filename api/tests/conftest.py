import os

# ---------------------------------------------------------------------------
# Service-ticket auth (self.ai#25 / api/src/core/auth.py) -- SERVICE_AUTH_SECRET
# is read once at import time by api.src.core.auth, so it must be set before
# anything imports that module (transitively, via api.src.main ->
# api.src.routers.openai_compatible -> ..core.auth below). Any non-empty
# value works; test_ticket_auth.py exercises real validation against this
# exact value, and the rest of this suite just needs create_speech's
# require_scope("audio:synthesize") dependency to be satisfiable via a
# minted ticket rather than 401ing every existing TTS-behavior test.
# ---------------------------------------------------------------------------
os.environ.setdefault("SERVICE_AUTH_SECRET", "pytest-only-service-auth-secret")

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import numpy as np
import pytest
import pytest_asyncio
import torch

from api.src.inference.model_manager import ModelManager
from api.src.inference.voice_manager import VoiceManager
from api.src.services.tts_service import TTSService
from api.src.structures.model_schemas import VoiceConfig

# Test-only HMAC secret/audience for the service-ticket auth layer
# (self.ai#25). Must match the SERVICE_AUTH_SECRET set above. Never used
# outside pytest -- real deployments get SERVICE_AUTH_SECRET from the
# selfai-service-auth ExternalSecret.
TEST_SERVICE_AUTH_SECRET = os.environ["SERVICE_AUTH_SECRET"]
TEST_SERVICE_AUTH_AUDIENCE = "self.speak"


def mint_test_ticket(
    scope="audio:synthesize",
    audience=TEST_SERVICE_AUTH_AUDIENCE,
    secret=TEST_SERVICE_AUTH_SECRET,
    ttl_seconds=120,
    **extra_claims,
):
    """Mint a service ticket signed with the pytest test secret. Mirrors
    self.ai's minting side (routers/audio.py's mint_service_ticket calls)
    closely enough to exercise the same validation path as production."""
    now = int(time.time())
    payload = {
        "iss": "self.ai",
        "aud": audience,
        "scope": scope,
        "iat": now,
        "exp": now + ttl_seconds,
    }
    payload.update(extra_claims)
    return jwt.encode(payload, secret, algorithm="HS256")


@pytest.fixture
def mock_voice_tensor():
    """Load a real voice tensor for testing."""
    voice_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "src/voices/af_bella.pt"
    )
    return torch.load(voice_path, map_location="cpu", weights_only=False)


@pytest.fixture
def mock_audio_output():
    """Load pre-generated test audio for consistent testing."""
    test_audio_path = os.path.join(
        os.path.dirname(__file__), "test_data/test_audio.npy"
    )
    return np.load(test_audio_path)  # Return as numpy array instead of bytes


@pytest_asyncio.fixture
async def mock_model_manager(mock_audio_output):
    """Mock model manager for testing."""
    manager = AsyncMock(spec=ModelManager)
    manager.get_backend = MagicMock()

    async def mock_generate(*args, **kwargs):
        # Simulate successful audio generation
        return np.random.rand(24000).astype(np.float32)  # 1 second of random audio data

    manager.generate = AsyncMock(side_effect=mock_generate)
    return manager


@pytest_asyncio.fixture
async def mock_voice_manager(mock_voice_tensor):
    """Mock voice manager for testing."""
    manager = AsyncMock(spec=VoiceManager)
    manager.get_voice_path = MagicMock(return_value="/mock/path/voice.pt")
    manager.load_voice = AsyncMock(return_value=mock_voice_tensor)
    manager.list_voices = AsyncMock(return_value=["voice1", "voice2"])
    manager.combine_voices = AsyncMock(return_value="voice1_voice2")
    return manager


@pytest_asyncio.fixture
async def tts_service(mock_model_manager, mock_voice_manager):
    """Get mocked TTS service instance."""
    service = TTSService()
    service.model_manager = mock_model_manager
    service._voice_manager = mock_voice_manager
    return service


@pytest.fixture
def test_voice():
    """Return a test voice name."""
    return "voice1"


@pytest.fixture
def valid_ticket_header():
    """A ready-to-use X-Selfai-Ticket header dict, correctly scoped for
    POST /v1/audio/speech. Use for tests exercising TTS behavior (not auth
    itself) so create_speech's require_scope("audio:synthesize") dependency
    doesn't 401 them -- real scope/audience/expiry enforcement is covered
    separately in test_ticket_auth.py."""
    return {"X-Selfai-Ticket": mint_test_ticket()}
