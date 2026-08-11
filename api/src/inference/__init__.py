"""Model inference package.

``KokoroV1`` is exported LAZILY (PEP 562). Importing this package used to pull in
kokoro -- and therefore the whole model stack -- as a side effect of importing
anything else in it. Under the P3 cutover (self.speak#5) main must never touch
the GPU, because it is the pod's sole liveness path and so can never exit to
return a CUDA primary context. Keeping the import behind attribute access means
a main that proxies to the worker genuinely never loads kokoro, rather than
loading it and merely declining to use it.

``from api.src.inference import KokoroV1`` still works and still imports it.
"""

from .base import BaseModelBackend
from .model_manager import ModelManager, get_manager

__all__ = [
    "BaseModelBackend",
    "ModelManager",
    "get_manager",
    "KokoroV1",
]


def __getattr__(name):
    if name == "KokoroV1":
        from .kokoro_v1 import KokoroV1

        return KokoroV1
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
