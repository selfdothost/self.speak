"""P2 worker tests (self.speak#5).

Covers the two things P2 actually decides: getting a blocking generator off the
event loop, and bounding a stream so a wedged worker cannot make self.speak
permanently unreclaimable to the VRAM broker.

Torch-free by construction: the bridge is pure asyncio/threading, and the frame
paths are exercised through the P1 codec, so this runs on a laptop.
"""

import asyncio
import time

import numpy as np
import pytest

from api.src.kokoro_worker import codec, stream
from api.src.kokoro_worker.bridge import BridgeTimeout, GeneratorBridge


def _agen(items, *, delay=0.0, raise_at=None):
    """Build a factory yielding `items`, optionally slowly or fatally."""

    async def _factory():
        for i, item in enumerate(items):
            if raise_at is not None and i == raise_at:
                raise RuntimeError("engine exploded")
            if delay:
                await asyncio.sleep(delay)
            yield item

    return _factory


async def _drain(bridge):
    return [x async for x in bridge.__aiter__()]


# --- the bridge -------------------------------------------------------------


@pytest.mark.asyncio
async def test_bridge_passes_every_item_in_order():
    bridge = GeneratorBridge(_agen(list(range(50)))).start()
    assert await _drain(bridge) == list(range(50))


@pytest.mark.asyncio
async def test_bridge_propagates_producer_exceptions():
    """A dead generator must reach the consumer.

    Swallowing it would leave the consumer waiting on a queue nothing will
    ever fill -- a hang, which is worse than the error.
    """
    bridge = GeneratorBridge(_agen([1, 2, 3], raise_at=1)).start()
    with pytest.raises(RuntimeError, match="engine exploded"):
        await _drain(bridge)


@pytest.mark.asyncio
async def test_bridge_reports_a_wedged_producer_instead_of_hanging():
    """The whole point of a per-item timeout."""

    async def _stalls():
        yield 1
        await asyncio.sleep(30)
        yield 2

    bridge = GeneratorBridge(lambda: _stalls(), item_timeout=0.5).start()
    with pytest.raises(BridgeTimeout):
        await _drain(bridge)


@pytest.mark.asyncio
async def test_bridge_keeps_the_event_loop_responsive():
    """The regression that motivates the bridge at all.

    A synchronous `for result in pipeline(...)` inside an async generator blocks
    the loop between yields, so /health and /vram-state go dark for the length
    of an utterance and the broker reads self.speak as unreachable.
    """

    async def _blocking():
        for i in range(5):
            time.sleep(0.1)  # blocking on purpose -- this is the real shape
            yield i

    ticks = 0

    async def _heartbeat():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    hb = asyncio.create_task(_heartbeat())
    bridge = GeneratorBridge(lambda: _blocking()).start()
    got = await _drain(bridge)
    hb.cancel()

    assert got == list(range(5))
    # ~500ms of blocking work; if it ran on this loop the heartbeat would be
    # starved. Threshold is deliberately loose -- the assertion is "the loop
    # kept running", not a precise tick count.
    assert ticks > 5, f"event loop was starved during generation (ticks={ticks})"


@pytest.mark.asyncio
async def test_bridge_applies_backpressure_rather_than_buffering_everything():
    """A slow consumer must stall the producer, not accumulate PCM in RAM."""
    produced = 0

    async def _counting():
        nonlocal produced
        for i in range(100):
            produced += 1
            yield i

    bridge = GeneratorBridge(lambda: _counting(), maxsize=4).start()
    agen = bridge.__aiter__()
    await agen.__anext__()
    await asyncio.sleep(0.3)  # let the producer run ahead as far as it can

    assert produced < 100, "producer ran to completion despite a stalled consumer"
    bridge.stop()


@pytest.mark.asyncio
async def test_the_default_queue_is_bounded():
    """Backpressure must be the DEFAULT, not something callers opt into.

    The test above passes maxsize explicitly, so it cannot notice the default
    going unbounded -- which is what production actually uses. A queue.Queue
    with maxsize=0 is infinite, so a fast producer would buffer an entire
    utterance of float32 PCM in RAM with every existing test still green.
    """
    bridge = GeneratorBridge(_agen([1]))
    assert bridge._q.maxsize > 0, "the default bridge queue is unbounded"


@pytest.mark.asyncio
async def test_stop_is_idempotent_and_non_blocking():
    bridge = GeneratorBridge(_agen([1, 2, 3])).start()
    bridge.stop()
    bridge.stop()


# --- stream framing + bounds ------------------------------------------------


class _Chunk:
    def __init__(self, audio, word_timestamps=None):
        self.audio = audio
        self.word_timestamps = word_timestamps


def _audio(n=8):
    return np.full(n, 0.1, dtype=np.float32)


class _FakeManager:
    def __init__(self, chunks, delay=0.0):
        self._chunks = chunks
        self._delay = delay

    def generate(self, text, voice, **kwargs):
        async def _gen():
            for c in self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                yield c

        return _gen()


async def _collect_frames(chunks, delay=0.0, **bounds):
    """Run the framing against a fake engine. No FastAPI, no torch, no GPU."""
    return [
        f
        async for f in stream.encode_stream(
            _FakeManager(chunks, delay), text="hello there", voice="af_heart", **bounds
        )
    ]


def _decode(frames):
    dec = codec.FrameDecoder()
    got = []
    for f in frames:
        got.extend(dec.feed(f))
    return dec, got


@pytest.mark.asyncio
async def test_stream_round_trips_through_the_codec():
    frames = await _collect_frames([_Chunk(_audio(16)), _Chunk(_audio(16))])
    dec, got = _decode(frames)
    dec.finish()

    assert isinstance(got[-1], codec.EndFrame)
    assert got[-1].frame_count == 2


@pytest.mark.asyncio
async def test_empty_chunks_are_skipped_without_breaking_seq():
    """The engine yields empty chunks; they must not create gaps in seq."""
    frames = await _collect_frames([_Chunk(_audio(8)), _Chunk(_audio(0)), _Chunk(_audio(8))])
    dec, got = _decode(frames)
    dec.finish()

    audio_frames = [g for g in got if isinstance(g, codec.AudioFrame)]
    assert [a.seq for a in audio_frames] == [0, 1]


@pytest.mark.asyncio
async def test_a_failed_generation_ends_with_an_ERROR_frame():
    """Never a body that merely stops -- that is indistinguishable from a kill."""

    class _BadManager:
        def generate(self, text, voice, **kwargs):
            async def _gen():
                yield _Chunk(_audio(8))
                raise RuntimeError("CUDA OOM")

            return _gen()

    frames = [
        f async for f in stream.encode_stream(_BadManager(), text="hi", voice="af")
    ]
    dec, got = _decode(frames)

    assert isinstance(got[-1], codec.ErrorFrame)
    assert "CUDA OOM" in got[-1].message
    dec.finish()  # an ERROR is a legitimate terminator


@pytest.mark.asyncio
async def test_whole_stream_deadline_terminates_a_wedged_stream():
    """The starvation bug this bound exists to prevent.

    A worker emitting one small frame every ~60s never trips a per-read timeout,
    so the stream never ends, so main's track_synthesis() pins _synthesis_count
    above zero forever -- and every later cooperative /vram-release returns
    partial/freed_bytes=0 with no unload. self.speak then becomes permanently
    unreclaimable to the broker until a force e-stop or a pod restart.
    """

    frames = await _collect_frames(
        [_Chunk(_audio(8)) for _ in range(100)], delay=0.05, deadline_seconds=0.2
    )
    dec, got = _decode(frames)

    assert isinstance(got[-1], codec.ErrorFrame)
    assert got[-1].error_class == "TimeoutError"
    assert len(got) < 100, "deadline did not actually cut the stream short"


@pytest.mark.asyncio
async def test_frame_count_bound_terminates_the_stream():
    frames = await _collect_frames([_Chunk(_audio(8)) for _ in range(20)], max_frames=5)
    dec, got = _decode(frames)
    assert isinstance(got[-1], codec.ErrorFrame)
    assert "frames" in got[-1].message


@pytest.mark.asyncio
async def test_bounds_are_generous_enough_for_a_real_synthesis():
    """A guard that fires on ordinary traffic is worse than no guard."""
    frames = await _collect_frames([_Chunk(_audio(24000)) for _ in range(20)])
    dec, got = _decode(frames)
    dec.finish()
    assert isinstance(got[-1], codec.EndFrame)
    assert got[-1].frame_count == 20
