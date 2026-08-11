"""Return this worker's CUDA primary context — the only way that exists.

``torch.cuda.empty_cache()`` returns the caching allocator's blocks. It does not
touch the CUDA PRIMARY CONTEXT, which is ~470 MiB on the deployed 4090, created
on the process's first allocation, and released only when the process exits.
Nothing in torch frees it: not empty_cache(), not dropping every tensor.

So a release request that unloading cannot satisfy has exactly one honest
answer left — stop being a process. The entrypoint respawn loop brings the
worker back, and it lazy-loads on the next /synth or /clone.

REACTIVE ONLY, on purpose. self.sketch got an idle timer as well; self.speak
deliberately does not. sketch is a bursty ride-along where a cold start is a
shrug, while self.speak fronts assistant devices where CUDA-init plus a model
load on the next request is a real cost. Keeping the worker warm is worth
~470 MiB right up until something else actually needs the card — and a FORCED
release from the broker is precisely the signal that something does.

The main uvicorn process is never exited this way. It is the sole liveness
path, so killing it drops the whole TTS API rather than one optional engine.
"""

import asyncio
import logging
import os

log = logging.getLogger("chatterbox_worker.reclaim")

# Kill-switch. Set to 0 to make a forced release report its shortfall and keep
# the process alive.
RELEASE_MAY_EXIT = os.environ.get("CHATTERBOX_RELEASE_MAY_EXIT", "1") not in (
    "0",
    "false",
    "False",
)

# Grace before exiting, so the HTTP response reaches the caller. Without it the
# broker sees a dropped connection instead of our answer and cannot tell
# "stepped aside" from "crashed" — which matters, because one is cooperation
# and the other is a fault worth alerting on.
_EXIT_GRACE_SECONDS = float(os.environ.get("CHATTERBOX_EXIT_GRACE_SECONDS", "1.5"))

_exit_scheduled = False


def context_held() -> bool:
    """True when this process holds GPU memory worth reclaiming.

    ``torch.cuda.memory_reserved() > 0`` — NOT ``torch.cuda.is_initialized()``.
    is_initialized() returns True after a mere device-properties query, with no
    context created and nothing to free. Using it as the test in self.sketch's
    watchdog cost hours of a service restarting every 15 minutes to reclaim
    memory it did not hold; the same wrong assumption is not repeated here.

    memory_reserved() is itself context-free and >0 only after a real
    allocation. A worker that has merely started can never authorise an exit.
    """
    try:
        import torch

        return bool(torch.cuda.memory_reserved() > 0)
    except Exception:
        return False


def schedule_exit(reason: str) -> bool:
    """Exit after a short grace period. Idempotent.

    ``os._exit`` rather than SIGTERM: this process is a child of the entrypoint
    respawn loop and installs no signal handlers, so a SIGTERM would depend on
    default disposition rather than anything we control. ``os._exit`` cannot be
    ignored, and the model is already unloaded by the time we get here — what
    remains is the context, and only death returns it.
    """
    global _exit_scheduled
    if _exit_scheduled:
        return False
    _exit_scheduled = True

    log.warning(
        "chatterbox-reclaim: stepping aside — %s. Exiting in %.1fs to return the "
        "CUDA primary context; the entrypoint respawn loop will restart this "
        "worker and it will lazy-load on the next request.",
        reason,
        _EXIT_GRACE_SECONDS,
    )

    async def _die():
        await asyncio.sleep(_EXIT_GRACE_SECONDS)
        log.warning("chatterbox-reclaim: exiting now")
        os._exit(0)

    try:
        asyncio.get_event_loop().create_task(_die())
    except Exception:
        # No loop to schedule on. A dropped response beats a context held
        # forever against a consumer that asked for it.
        os._exit(0)
    return True
