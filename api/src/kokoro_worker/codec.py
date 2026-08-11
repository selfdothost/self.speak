"""Wire format for streaming Kokoro audio out of a worker process (self.speak#5, P1).

Kokoro cannot use the chatterbox worker's shape. Chatterbox returns ONE complete
waveform over a plain POST, so raw PCM plus a sample-rate header is a sufficient
wire format. ``KokoroV1.generate()`` is an ``AsyncGenerator[AudioChunk, None]``
that yields incrementally, and each chunk carries word timestamps alongside the
audio. So the boundary needs framing, and framing is the risky part — hence P1
being its own phase with its own review.

Frame layout, all integers big-endian::

    SYNC     4 bytes   b"KWF1"
    TYPE     1 byte    0x01 AUDIO | 0x02 END | 0x03 ERROR
    HLEN     4 bytes   uint32, header length
    PLEN     4 bytes   uint32, payload length
    HEADER   HLEN      UTF-8 JSON
    PAYLOAD  PLEN      little-endian float32 PCM (AUDIO only; 0 for END/ERROR)

Design rules, each one a defect found in review rather than a preference. They
are stated here because the reasons are not recoverable from the code:

1.  **float32 only.** An earlier draft allowed ``{"<f4","<f8","<i2","<i4"}`` on
    the theory that a wider allowlist fails more loudly. It is the reverse: the
    parent does ``chunk.audio *= volume_multiplier`` in place
    (``model_manager.py``, ``tts_service.py``), and numpy raises UFuncTypeError
    on an in-place float multiply into an int array. That lands in a handler
    that logs and returns, so an int dtype produced a 200 with no audio. One
    dtype, rejected at the boundary, is the loud version.

2.  **1-D, and ``prod(shape) * 4`` must equal PLEN exactly.** ``reshape``
    accepts negative dimensions, so ``shape: [-1]`` silently means "whatever the
    payload implies" — the opposite of the self-describing guarantee that
    carrying shape was supposed to buy. A 2-D shape is worse than an error:
    ``len(audio)`` then counts ROWS, so the caption offset advances by a
    thousandth of the real duration and every subsequent timestamp is wrong,
    with no exception anywhere.

3.  **A non-finite timestamp must never cost audio.** ``kokoro_v1`` computes
    ``float(token.start_ts) + current_offset`` with no finiteness guard, and
    ``WordTimestamp`` is a plain pydantic float that accepts NaN. An earlier
    draft specified ``json.dumps(allow_nan=False)`` and let the encoder raise,
    which converts a cosmetic caption defect into a dropped chunk — up to ~450
    tokens of speech, silently. Here a non-finite entry is DROPPED FROM THE
    CAPTIONS and counted in ``dropped_timestamps``; the audio always ships.
    Degrade the annotation, never the thing the user actually hears.

4.  **Timestamps decode to ``WordTimestamp``, not dicts.** The parent mutates
    them in place (``timestamp.start_time += current_offset``), which is an
    AttributeError on a dict — caught upstream by a handler that drops the whole
    text chunk. Pydantic would coerce, but only downstream of both mutation
    sites, so coercion arrives too late to help.

5.  **``sample_rate`` is validated, not merely carried.** It was previously
    justified as closing a drift hole while having no consumer at all — the
    decoder built an ``AudioChunk``, which has nowhere to put it, and the parent
    divided by a hard-coded 24000 regardless. A field nobody reads closes
    nothing. Here a mismatch is a protocol error at the boundary.

6.  **END distinguishes cooperation from a crash.** ``reason="stepping_aside"``
    exists because ``chatterbox_worker/reclaim.py`` grew a whole grace period to
    preserve exactly that distinction: a dropped connection cannot tell the
    broker whether the worker yielded the card on request or died. Both
    collapsing to "stream truncated" would throw that away and page an operator
    for an expected e-stop.

7.  **One exception type out.** Every malformed-input path — bad sync, bad JSON,
    bad shape, short payload, oversize length — raises
    ``KokoroWorkerProtocolError``. Bare ``ValueError`` from numpy or
    ``JSONDecodeError`` from the header would slip past the caller's
    ``except KokoroWorkerProtocolError`` and be swallowed further up.

8.  **The peer is unauthenticated.** It is a localhost socket with no auth in
    the threat model, so HLEN and PLEN are bounded BEFORE anything is allocated.
    A length field is an allocation request from an untrusted party.
"""

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Union

import numpy as np

from ..structures.schemas import WordTimestamp

SYNC = b"KWF1"

TYPE_AUDIO = 0x01
TYPE_END = 0x02
TYPE_ERROR = 0x03

_PREFIX = struct.Struct(">4sBII")
_PREFIX_LEN = _PREFIX.size  # 13

# Bounds applied before allocation. Generous enough that no legitimate frame
# comes close: a 450-token chunk is roughly 1.4 MB of float32 PCM.
MAX_HEADER_BYTES = 1 << 20  # 1 MiB
MAX_PAYLOAD_BYTES = 64 << 20  # 64 MiB

# The only dtype on the wire. See rule 1.
_WIRE_DTYPE = np.dtype("<f4")
_ITEMSIZE = _WIRE_DTYPE.itemsize

DEFAULT_SAMPLE_RATE = 24000

# END reasons. See rule 6.
END_COMPLETE = "complete"
END_STEPPING_ASIDE = "stepping_aside"
_END_REASONS = frozenset({END_COMPLETE, END_STEPPING_ASIDE})


class KokoroWorkerProtocolError(Exception):
    """The bytes on the wire were not a valid frame stream.

    Distinct from a worker being unreachable: that is a transport fault and is
    retryable, while this means the peer is speaking something other than this
    protocol and retrying will not help.
    """


@dataclass
class AudioFrame:
    seq: int
    audio: np.ndarray
    sample_rate: int
    word_timestamps: Optional[List[WordTimestamp]] = None
    dropped_timestamps: int = 0


@dataclass
class EndFrame:
    reason: str
    frame_count: int

    @property
    def cooperative(self) -> bool:
        """True when the worker chose to stop — not a fault worth paging on."""
        return self.reason == END_STEPPING_ASIDE


@dataclass
class ErrorFrame:
    error_class: str
    message: str
    detail: dict = field(default_factory=dict)


Frame = Union[AudioFrame, EndFrame, ErrorFrame]


def _sanitize_timestamps(word_timestamps):
    """Drop caption entries that cannot be represented, keeping a count.

    Rule 3. A NaN or Inf start/end time is a defect in the caption, not in the
    audio, and it must not be allowed to become a reason to discard the audio.
    Returns ``(clean_list_or_None, dropped_count)``.
    """
    if word_timestamps is None:
        return None, 0

    clean = []
    dropped = 0
    for ts in word_timestamps:
        if isinstance(ts, WordTimestamp):
            word, start, end = ts.word, ts.start_time, ts.end_time
        elif isinstance(ts, dict):
            word = ts.get("word")
            start = ts.get("start_time")
            end = ts.get("end_time")
        else:
            word = getattr(ts, "word", None)
            start = getattr(ts, "start_time", None)
            end = getattr(ts, "end_time", None)

        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            dropped += 1
            continue

        if not (math.isfinite(start) and math.isfinite(end)) or word is None:
            dropped += 1
            continue

        clean.append({"word": str(word), "start_time": start, "end_time": end})

    return clean, dropped


def _frame(frame_type: int, header: dict, payload: bytes = b"") -> bytes:
    # allow_nan=False is safe here only because timestamps are sanitised first;
    # it exists to catch a non-finite value that slipped in some other way, at
    # the producer, where it is a bug rather than a mystery.
    header_bytes = json.dumps(header, allow_nan=False).encode("utf-8")
    if len(header_bytes) > MAX_HEADER_BYTES:
        raise KokoroWorkerProtocolError(
            f"header of {len(header_bytes)} bytes exceeds the {MAX_HEADER_BYTES} limit"
        )
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise KokoroWorkerProtocolError(
            f"payload of {len(payload)} bytes exceeds the {MAX_PAYLOAD_BYTES} limit"
        )
    return _PREFIX.pack(SYNC, frame_type, len(header_bytes), len(payload)) + header_bytes + payload


def encode_audio_frame(
    audio: np.ndarray,
    seq: int,
    word_timestamps=None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Encode one AudioChunk's worth of audio.

    Rejects anything but 1-D float32 AT THE PRODUCER, where a wrong dtype is a
    local bug with a stack trace, rather than shipping it to a consumer that can
    only fail obscurely (rule 1).
    """
    if not isinstance(audio, np.ndarray):
        raise KokoroWorkerProtocolError(f"audio must be an ndarray, got {type(audio).__name__}")
    if audio.dtype != _WIRE_DTYPE:
        raise KokoroWorkerProtocolError(
            f"audio must be float32 (got {audio.dtype}); the parent multiplies it "
            "in place by a float volume multiplier and any int dtype raises there"
        )
    if audio.ndim != 1:
        raise KokoroWorkerProtocolError(f"audio must be 1-D, got shape {audio.shape}")

    clean_ts, dropped = _sanitize_timestamps(word_timestamps)

    header = {
        "seq": int(seq),
        "shape": [int(audio.shape[0])],
        "sample_rate": int(sample_rate),
        "word_timestamps": clean_ts,
        "dropped_timestamps": dropped,
    }
    return _frame(TYPE_AUDIO, header, np.ascontiguousarray(audio).tobytes())


def encode_end_frame(frame_count: int, reason: str = END_COMPLETE) -> bytes:
    """Encode the terminator. Its absence is what makes truncation detectable."""
    if reason not in _END_REASONS:
        raise KokoroWorkerProtocolError(f"unknown end reason {reason!r}")
    return _frame(TYPE_END, {"reason": reason, "frame_count": int(frame_count)})


def encode_error_frame(error_class: str, message: str, detail: Optional[dict] = None) -> bytes:
    """Encode a failure the worker wants the parent to know about.

    ``error_class`` is a free string, deliberately not a closed allowlist of
    builtins. The earlier allowlist was justified by needing FileNotFoundError
    to survive to a specific handler — but that handler sits on the load path,
    which P1 does not carry, and the serving path flattens every exception to
    RuntimeError two frames later regardless. A closed list bought nothing and
    would silently mistranslate anything unforeseen.
    """
    return _frame(
        TYPE_ERROR,
        {"class": str(error_class), "message": str(message), "detail": detail or {}},
    )


class FrameDecoder:
    """Incremental decoder. Feed it arbitrary byte runs; it yields whole frames.

    Chunk boundaries from the transport have nothing to do with frame
    boundaries, so every read is appended to a buffer and frames are emitted
    only once complete.

    On ``finish()`` and early termination — the ambiguity that made an earlier
    draft unimplementable. The caller stopping early is a NORMAL, frequent path
    here: the speech endpoint checks ``client_request.is_disconnected`` on every
    chunk and breaks when an assistant device hangs up. So:

      * ``finish()`` means "the transport reported EOF; was the stream whole?"
        Call it only on that path. It raises when no END/ERROR arrived.
      * A caller that stops early must NOT call it. Check ``.ended`` instead —
        putting ``finish()`` in a bare ``finally`` turns every user hangup into
        a logged protocol fault during generator unwinding.
    """

    def __init__(self, expected_sample_rate: int = DEFAULT_SAMPLE_RATE):
        self._buf = bytearray()
        self._expected_sample_rate = expected_sample_rate
        self._next_seq = 0
        self._audio_frames = 0
        self.ended = False
        self.end_frame: Optional[EndFrame] = None

    def feed(self, data: bytes) -> Iterator[Frame]:
        """Append bytes and yield every frame that is now complete."""
        if self.ended:
            raise KokoroWorkerProtocolError("data arrived after the stream ended")
        self._buf.extend(data)

        while True:
            if len(self._buf) < _PREFIX_LEN:
                return
            sync, frame_type, hlen, plen = _PREFIX.unpack_from(self._buf, 0)
            if sync != SYNC:
                raise KokoroWorkerProtocolError(f"bad frame sync {bytes(sync)!r}")
            # Bound BEFORE waiting on (and therefore buffering) the declared
            # length — the peer is unauthenticated (rule 8).
            if hlen > MAX_HEADER_BYTES:
                raise KokoroWorkerProtocolError(f"header length {hlen} exceeds limit")
            if plen > MAX_PAYLOAD_BYTES:
                raise KokoroWorkerProtocolError(f"payload length {plen} exceeds limit")

            total = _PREFIX_LEN + hlen + plen
            if len(self._buf) < total:
                return

            header_bytes = bytes(self._buf[_PREFIX_LEN : _PREFIX_LEN + hlen])
            payload = bytes(self._buf[_PREFIX_LEN + hlen : total])
            del self._buf[:total]

            yield self._decode(frame_type, header_bytes, payload)

    def _decode(self, frame_type: int, header_bytes: bytes, payload: bytes) -> Frame:
        try:
            header = json.loads(header_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            # Rule 7: never let a raw JSONDecodeError out.
            raise KokoroWorkerProtocolError(f"undecodable frame header: {e}") from e
        if not isinstance(header, dict):
            raise KokoroWorkerProtocolError("frame header must be a JSON object")

        if frame_type == TYPE_AUDIO:
            return self._decode_audio(header, payload)
        if frame_type == TYPE_END:
            reason = header.get("reason")
            if reason not in _END_REASONS:
                raise KokoroWorkerProtocolError(f"unknown end reason {reason!r}")
            declared = header.get("frame_count")
            if declared != self._audio_frames:
                raise KokoroWorkerProtocolError(
                    f"end frame declares {declared} audio frames, {self._audio_frames} arrived"
                )
            self.ended = True
            self.end_frame = EndFrame(reason=reason, frame_count=self._audio_frames)
            return self.end_frame
        if frame_type == TYPE_ERROR:
            self.ended = True
            return ErrorFrame(
                error_class=str(header.get("class", "Exception")),
                message=str(header.get("message", "")),
                detail=header.get("detail") or {},
            )
        raise KokoroWorkerProtocolError(f"unknown frame type 0x{frame_type:02x}")

    def _decode_audio(self, header: dict, payload: bytes) -> AudioFrame:
        shape = header.get("shape")
        # Rule 2. Checked explicitly rather than left to reshape, which accepts
        # negative dimensions and would turn a malformed header into a silently
        # mis-shaped array.
        if (
            not isinstance(shape, list)
            or len(shape) != 1
            or not isinstance(shape[0], int)
            or isinstance(shape[0], bool)
            or shape[0] < 0
        ):
            raise KokoroWorkerProtocolError(f"shape must be [n>=0], got {shape!r}")
        expected = shape[0] * _ITEMSIZE
        if expected != len(payload):
            raise KokoroWorkerProtocolError(
                f"shape {shape} implies {expected} payload bytes, got {len(payload)}"
            )

        sample_rate = header.get("sample_rate")
        # Rule 5: the field is only worth carrying if a mismatch is an error.
        if sample_rate != self._expected_sample_rate:
            raise KokoroWorkerProtocolError(
                f"worker sent sample_rate {sample_rate!r}, expected "
                f"{self._expected_sample_rate}; the parent's caption maths assumes the latter"
            )

        seq = header.get("seq")
        if seq != self._next_seq:
            raise KokoroWorkerProtocolError(f"expected seq {self._next_seq}, got {seq!r}")
        self._next_seq += 1
        self._audio_frames += 1

        # Copy: np.frombuffer over an immutable bytes gives a read-only array,
        # and the parent multiplies it in place.
        audio = np.frombuffer(bytearray(payload), dtype=_WIRE_DTYPE)

        raw_ts = header.get("word_timestamps")
        word_timestamps = None
        if raw_ts is not None:
            if not isinstance(raw_ts, list):
                raise KokoroWorkerProtocolError("word_timestamps must be a list or null")
            try:
                # Rule 4: real WordTimestamp objects, because the parent mutates
                # their attributes in place.
                word_timestamps = [WordTimestamp(**ts) for ts in raw_ts]
            except Exception as e:
                raise KokoroWorkerProtocolError(f"malformed word_timestamps: {e}") from e

        return AudioFrame(
            seq=seq,
            audio=audio,
            sample_rate=sample_rate,
            word_timestamps=word_timestamps,
            dropped_timestamps=int(header.get("dropped_timestamps") or 0),
        )

    def finish(self) -> None:
        """Assert the stream ended properly. See the class docstring on when.

        Trailing bytes are a violation too: they mean a frame was cut mid-way,
        which is what a hard-killed producer looks like when the transport did
        not also fail.
        """
        if not self.ended:
            raise KokoroWorkerProtocolError(
                f"stream truncated after {self._audio_frames} audio frame(s): "
                "no END or ERROR frame arrived"
            )
        if self._buf:
            raise KokoroWorkerProtocolError(
                f"{len(self._buf)} trailing byte(s) after the terminating frame"
            )
