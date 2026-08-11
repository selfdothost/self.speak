"""Run a BLOCKING async generator without blocking the serving event loop.

This is the piece the P1 review flagged as real design work rather than a
detail. ``KokoroV1.generate()`` is an ``async`` generator, but the work inside
it is a synchronous ``for result in pipeline(text, voice=..., model=...)``. An
``async def`` wrapper around blocking code is still blocking: between yields the
event loop cannot run anything, so ``/health`` and ``/vram-state`` go dark for
the length of an utterance.

Chatterbox dodged this with ``await run_in_threadpool(_run_generate, req)`` —
one call, one return value. **That does not work for a generator.** There is no
``run_in_threadpool`` for something that yields N times; awaiting it would just
move the first ``__anext__`` off the loop and leave the rest on it.

So the generator runs to completion on its own thread, in its own event loop,
pushing each chunk through a ``queue.Queue``. The serving coroutine takes items
via ``run_in_executor``, which parks an executor thread rather than the loop.

Why this shape:

* **``queue.Queue`` with a maxsize, not an unbounded ``asyncio.Queue``.** A
  blocking ``put`` gives back-pressure for free: if a client reads slower than
  Kokoro generates, the producer thread stalls instead of buffering an entire
  utterance of float32 PCM in RAM.
* **Every ``get`` has a timeout.** A producer that wedges — a degenerate
  pipeline loop, a thrashing GPU — must not park the consumer forever. This is
  what makes the whole-stream deadline in ``main`` enforceable at all; without a
  bounded get there is nothing to enforce it against.
* **The terminal item is explicit** (``_DONE`` / ``_ERROR``), never "the queue
  went quiet". Quiet is indistinguishable from slow, and inferring the end of a
  stream from silence is precisely the class of bug that made a failed
  generation look like a complete one (self.speak#6).
* **``stop()`` is cooperative.** A cancelled client sets the flag; the producer
  notices at its next yield and returns. The blocking torch call in flight
  cannot be interrupted, so the thread lives until that call returns — which is
  why it is a daemon thread and why the deadline exists.
"""

import asyncio
import logging
import queue
import threading
from typing import Any, AsyncIterator, Callable

log = logging.getLogger("kokoro_worker.bridge")

_DONE = object()
_ERROR = object()


class BridgeTimeout(Exception):
    """No item arrived within the allowed window — the producer is wedged."""


class GeneratorBridge:
    """Bridge one async generator from a private thread to the serving loop.

    Single-use: construct, iterate, discard.
    """

    def __init__(
        self,
        factory: Callable[[], AsyncIterator[Any]],
        *,
        maxsize: int = 4,
        item_timeout: float = 120.0,
    ):
        self._factory = factory
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._item_timeout = item_timeout
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- producer side (its own thread, its own loop) ------------------------

    def _run(self) -> None:
        async def _pump():
            agen = self._factory()
            try:
                async for item in agen:
                    if self._stop.is_set():
                        break
                    # Blocking put: back-pressure. The timeout keeps a producer
                    # from hanging forever against a consumer that has gone away
                    # without setting the stop flag (a hard client disconnect).
                    while not self._stop.is_set():
                        try:
                            self._q.put(item, timeout=1.0)
                            break
                        except queue.Full:
                            continue
            finally:
                aclose = getattr(agen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001 - cleanup must not mask
                        log.debug("kokoro-bridge: aclose raised during teardown")

        try:
            asyncio.run(_pump())
        except BaseException as e:  # noqa: BLE001 - must reach the consumer
            # Deliberately BaseException: the consumer has to learn that the
            # stream died, whatever killed it. Swallowing here would leave the
            # consumer waiting on a queue nothing will ever fill.
            self._put_terminal(_ERROR, e)
            return
        self._put_terminal(_DONE, None)

    def _put_terminal(self, kind: object, payload: Any) -> None:
        """Deliver the terminator even if the queue is full.

        A full queue must never lose the end-of-stream marker; the consumer is
        draining and will make room. Bounded so a vanished consumer cannot pin
        this thread forever.
        """
        for _ in range(60):
            try:
                self._q.put((kind, payload), timeout=1.0)
                return
            except queue.Full:
                if self._stop.is_set():
                    return
        log.warning("kokoro-bridge: gave up delivering the terminal marker")

    def start(self) -> "GeneratorBridge":
        self._thread = threading.Thread(
            target=self._run, name="kokoro-generate", daemon=True
        )
        self._thread.start()
        return self

    # -- consumer side (the serving loop) -----------------------------------

    async def __aiter__(self):
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(
                    None, self._q.get, True, self._item_timeout
                )
            except queue.Empty as e:
                raise BridgeTimeout(
                    f"no audio from the generator within {self._item_timeout}s"
                ) from e

            if isinstance(item, tuple) and len(item) == 2 and item[0] is _ERROR:
                raise item[1]
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _DONE:
                return
            yield item

    def stop(self) -> None:
        """Ask the producer to stop at its next yield. Idempotent, non-blocking.

        Cannot interrupt a torch call already in flight — the thread is a daemon
        precisely so that cannot hold up process exit.
        """
        self._stop.set()
