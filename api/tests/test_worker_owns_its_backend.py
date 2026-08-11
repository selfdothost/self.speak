"""The worker must own an in-process model (self.speak#5).

**These tests exist because the first enablement attempt broke every synthesis
and no existing test could have caught it.** Every worker test to date injected
a FAKE manager — including the integration suite added specifically to cover the
client<->worker seam. It crossed the HTTP boundary faithfully and then stubbed
out the one component whose real behaviour was wrong.

So these use the REAL ModelManager. The backend construction is stubbed (that is
the part needing a GPU), but the selection logic and the initialisation call are
the genuine article, because those are what failed:

  1. the worker never called ensure_loaded(), so every request died on
     "Backend not initialized"
  2. the worker reads the SAME KOKORO_WORKER_ENABLED as main, so once (1) was
     fixed it would have selected KokoroWorkerBackend and proxied to ITSELF
"""

from unittest.mock import AsyncMock, patch

import pytest

from api.src.core.config import settings
from api.src.inference import model_manager


@pytest.fixture
def worker_flag_on():
    """Exactly the deployed situation: the flag is true in BOTH processes."""
    before = settings.kokoro_worker_enabled
    settings.kokoro_worker_enabled = True
    yield
    settings.kokoro_worker_enabled = before


@pytest.fixture(autouse=True)
def _reset_role():
    """The role flag is process-global; never let it leak between tests."""
    before = model_manager._FORCE_IN_PROCESS
    yield
    model_manager._FORCE_IN_PROCESS = before


class _FakeKokoro:
    """Stands in for KokoroV1 so no GPU is needed."""

    streams_text = True

    def __init__(self):
        self.loaded = False

    async def load_model(self, path):
        self.loaded = True

    @property
    def is_loaded(self):
        return self.loaded


@pytest.mark.asyncio
async def test_the_worker_does_not_proxy_to_itself(worker_flag_on):
    """The bug that would have survived fixing the other one.

    With KOKORO_WORKER_ENABLED true in both processes, a worker that did not
    declare its role selected KokoroWorkerBackend and pointed at 127.0.0.1:8882
    — itself.
    """
    model_manager.force_in_process_backend()
    mgr = model_manager.ModelManager()

    with patch.object(model_manager, "KokoroV1", _FakeKokoro, create=True):
        await mgr.initialize()

    backend = mgr.get_backend()
    assert isinstance(backend, _FakeKokoro), (
        f"worker selected {type(backend).__name__} — it is proxying to itself"
    )


@pytest.mark.asyncio
async def test_main_still_proxies_when_the_role_is_not_declared(worker_flag_on):
    """The role flag must not break main, which SHOULD proxy."""
    from api.src.inference.kokoro_client import KokoroWorkerBackend

    mgr = model_manager.ModelManager()
    await mgr.initialize()

    assert isinstance(mgr.get_backend(), KokoroWorkerBackend)


@pytest.mark.asyncio
async def test_the_worker_loads_its_model_before_serving(worker_flag_on):
    """The bug that took production down.

    Nothing else initialises the backend in the worker process, so without an
    ensure_loaded() every request died on "Backend not initialized".
    """
    from api.src.kokoro_worker import main as worker

    fake = _FakeKokoro()
    mgr = model_manager.ModelManager()

    with (
        patch.object(model_manager, "get_manager", AsyncMock(return_value=mgr)),
        patch.object(model_manager, "KokoroV1", lambda: fake, create=True),
        patch.object(mgr, "load_model", AsyncMock(side_effect=fake.load_model)),
    ):
        got = await worker._engine()

    assert got is mgr
    assert got.get_backend() is fake, "worker did not initialise a backend"
    assert fake.loaded, "worker did not load the model"


@pytest.mark.asyncio
async def test_engine_is_awaitable(worker_flag_on):
    """_engine() used to be sync and is now async; the call sites must agree.

    A mismatch here is silent: an un-awaited coroutine is truthy, so the caller
    would carry on and fail somewhere less obvious.
    """
    import inspect

    from api.src.kokoro_worker import main as worker

    assert inspect.iscoroutinefunction(worker._engine)


def test_the_role_is_declared_not_inferred_from_the_environment():
    """The role must NOT come from a value both processes can see.

    main and the worker share KOKORO_WORKER_ENABLED, and they need opposite
    answers to "should I proxy?". Deriving the role from any environment
    variable reintroduces exactly the bug that took TTS down.
    """
    import inspect

    src = inspect.getsource(model_manager.force_in_process_backend)
    assert "environ" not in src and "getenv" not in src
    assert "settings" not in src
