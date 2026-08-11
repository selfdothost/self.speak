"""P1 wire-format tests (self.speak#5).

Structured so each defect found in review has a test named after it. The point
of P1 is that the format is reviewed before a worker is built on it, so a
regression here should say which rule it broke, not just "codec broken".

Deliberately importable without torch: the codec depends on numpy and pydantic
only, so the format is verifiable on a laptop rather than only in CI.
"""

import json
import struct

import numpy as np
import pytest

from api.src.kokoro_worker.codec import (
    DEFAULT_SAMPLE_RATE,
    END_COMPLETE,
    END_STEPPING_ASIDE,
    MAX_PAYLOAD_BYTES,
    SYNC,
    TYPE_AUDIO,
    AudioFrame,
    EndFrame,
    ErrorFrame,
    FrameDecoder,
    KokoroWorkerProtocolError,
    encode_audio_frame,
    encode_end_frame,
    encode_error_frame,
)
from api.src.structures.schemas import WordTimestamp


def _audio(n=8, value=0.25):
    return np.full(n, value, dtype=np.float32)


def _decode_all(*blobs, expected_sample_rate=DEFAULT_SAMPLE_RATE):
    """Feed blobs as one stream and collect every frame."""
    dec = FrameDecoder(expected_sample_rate=expected_sample_rate)
    frames = []
    for blob in blobs:
        frames.extend(dec.feed(blob))
    return dec, frames


def _raw_frame(header: dict, payload: bytes = b"", frame_type: int = TYPE_AUDIO, sync=SYNC):
    """Hand-build a frame so malformed inputs can be tested."""
    hb = json.dumps(header).encode("utf-8")
    return struct.pack(">4sBII", sync, frame_type, len(hb), len(payload)) + hb + payload


# --- the happy path ---------------------------------------------------------


def test_round_trip_preserves_audio_exactly():
    audio = np.linspace(-1.0, 1.0, 512, dtype=np.float32)
    dec, frames = _decode_all(encode_audio_frame(audio, seq=0), encode_end_frame(1))
    dec.finish()

    assert isinstance(frames[0], AudioFrame)
    np.testing.assert_array_equal(frames[0].audio, audio)
    assert isinstance(frames[1], EndFrame)


def test_round_trip_preserves_word_timestamps():
    """The acceptance criterion: timestamps identical across the boundary."""
    ts = [
        WordTimestamp(word="hello", start_time=0.0, end_time=0.4),
        WordTimestamp(word="world", start_time=0.4, end_time=0.9),
    ]
    _, frames = _decode_all(encode_audio_frame(_audio(), seq=0, word_timestamps=ts))

    got = frames[0].word_timestamps
    assert [(t.word, t.start_time, t.end_time) for t in got] == [
        (t.word, t.start_time, t.end_time) for t in ts
    ]


def test_frames_survive_arbitrary_transport_chunking():
    """Transport reads have nothing to do with frame boundaries."""
    stream = encode_audio_frame(_audio(64), seq=0) + encode_audio_frame(
        _audio(64), seq=1
    ) + encode_end_frame(2)

    for size in (1, 3, 7, 13, 64, 4096):
        dec = FrameDecoder()
        frames = []
        for i in range(0, len(stream), size):
            frames.extend(dec.feed(stream[i : i + size]))
        dec.finish()
        assert len(frames) == 3, f"chunk size {size} lost frames"


def test_empty_audio_frame_is_legal():
    """A zero-length chunk is not malformed; kokoro can yield one."""
    _, frames = _decode_all(encode_audio_frame(_audio(0), seq=0))
    assert len(frames[0].audio) == 0


# --- rule 4: timestamps decode as objects, not dicts ------------------------


def test_decoded_timestamps_support_in_place_mutation():
    """The parent does `timestamp.start_time += offset`.

    A dict here is an AttributeError upstream, caught by a handler that drops
    the whole text chunk — 200 with no audio for every captioned request.
    """
    ts = [WordTimestamp(word="hi", start_time=0.1, end_time=0.2)]
    _, frames = _decode_all(encode_audio_frame(_audio(), seq=0, word_timestamps=ts))

    decoded = frames[0].word_timestamps[0]
    decoded.start_time += 1.0  # must not raise
    assert decoded.start_time == pytest.approx(1.1)


# --- rule 3: a bad caption must never cost audio ----------------------------


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_timestamp_drops_the_caption_not_the_audio(bad):
    """The inversion that mattered most.

    Encoding used to raise on a non-finite timestamp, which upstream became a
    dropped chunk: ~450 tokens of speech gone, HTTP 200, no client-visible
    error. A NaN start_ts is a caption defect; the audio is fine and must ship.
    """
    audio = _audio(32)
    ts = [
        WordTimestamp(word="good", start_time=0.0, end_time=0.1),
        WordTimestamp(word="bad", start_time=bad, end_time=0.3),
    ]
    _, frames = _decode_all(encode_audio_frame(audio, seq=0, word_timestamps=ts))

    np.testing.assert_array_equal(frames[0].audio, audio)
    assert [t.word for t in frames[0].word_timestamps] == ["good"]
    assert frames[0].dropped_timestamps == 1


def test_all_timestamps_non_finite_still_ships_audio():
    audio = _audio(16)
    ts = [WordTimestamp(word="x", start_time=float("nan"), end_time=float("nan"))]
    _, frames = _decode_all(encode_audio_frame(audio, seq=0, word_timestamps=ts))

    np.testing.assert_array_equal(frames[0].audio, audio)
    assert frames[0].word_timestamps == []
    assert frames[0].dropped_timestamps == 1


def test_no_timestamps_stays_none():
    """None and [] are different: null means "not requested"."""
    _, frames = _decode_all(encode_audio_frame(_audio(), seq=0, word_timestamps=None))
    assert frames[0].word_timestamps is None


# --- rule 1: float32 only ---------------------------------------------------


@pytest.mark.parametrize("dtype", ["int16", "int32", "float64", "uint8"])
def test_encoder_refuses_non_float32(dtype):
    """An int dtype reaching the parent raises UFuncTypeError on an in-place
    float multiply, which is swallowed upstream — 200 with silence."""
    with pytest.raises(KokoroWorkerProtocolError, match="float32"):
        encode_audio_frame(np.ones(4, dtype=dtype), seq=0)


def test_encoder_refuses_multidimensional_audio():
    with pytest.raises(KokoroWorkerProtocolError, match="1-D"):
        encode_audio_frame(np.ones((2, 4), dtype=np.float32), seq=0)


def test_decoded_audio_is_writable():
    """The parent multiplies in place; frombuffer over bytes is read-only."""
    _, frames = _decode_all(encode_audio_frame(_audio(), seq=0))
    frames[0].audio *= 2.0  # must not raise


# --- rule 2: shape is validated, not handed to reshape ----------------------


def test_negative_shape_is_rejected():
    """`reshape([-1])` silently means "whatever the payload implies", which is
    the opposite of the self-describing guarantee shape exists to provide."""
    blob = _raw_frame(
        {"seq": 0, "shape": [-1], "sample_rate": DEFAULT_SAMPLE_RATE, "word_timestamps": None},
        payload=b"\x00" * 8,
    )
    with pytest.raises(KokoroWorkerProtocolError, match="shape"):
        _decode_all(blob)


def test_multidimensional_shape_is_rejected():
    """A 2-D array makes len(audio) count ROWS, so the caption offset advances
    by a thousandth of the real duration — desync with no exception at all."""
    blob = _raw_frame(
        {"seq": 0, "shape": [2, 3], "sample_rate": DEFAULT_SAMPLE_RATE, "word_timestamps": None},
        payload=b"\x00" * 24,
    )
    with pytest.raises(KokoroWorkerProtocolError, match="shape"):
        _decode_all(blob)


@pytest.mark.parametrize("declared,payload_len", [(12, 44), (11, 48), (0, 4)])
def test_shape_must_match_payload_length_exactly(declared, payload_len):
    blob = _raw_frame(
        {
            "seq": 0,
            "shape": [declared],
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "word_timestamps": None,
        },
        payload=b"\x00" * payload_len,
    )
    with pytest.raises(KokoroWorkerProtocolError, match="payload bytes"):
        _decode_all(blob)


# --- rule 5: sample_rate is consumed --------------------------------------


def test_sample_rate_mismatch_is_a_protocol_error():
    """Previously carried and discarded, so a mismatched worker drifted
    captions ~9% with no assertion possible anywhere."""
    blob = encode_audio_frame(_audio(), seq=0, sample_rate=22050)
    with pytest.raises(KokoroWorkerProtocolError, match="sample_rate"):
        _decode_all(blob)


def test_matching_sample_rate_passes_through():
    _, frames = _decode_all(
        encode_audio_frame(_audio(), seq=0, sample_rate=16000), expected_sample_rate=16000
    )
    assert frames[0].sample_rate == 16000


# --- rule 6: cooperation is distinguishable from a crash -------------------


def test_stepping_aside_is_distinguishable_from_completion():
    """A dropped connection cannot tell the broker whether the worker yielded
    the card on request or died. Both reducing to "truncated" would page an
    operator for an expected e-stop."""
    _, frames = _decode_all(encode_audio_frame(_audio(), seq=0), encode_end_frame(1, END_STEPPING_ASIDE))
    end = frames[-1]
    assert end.cooperative is True
    assert end.reason == END_STEPPING_ASIDE


def test_normal_completion_is_not_flagged_cooperative():
    _, frames = _decode_all(encode_end_frame(0, END_COMPLETE))
    assert frames[0].cooperative is False


def test_unknown_end_reason_is_rejected():
    with pytest.raises(KokoroWorkerProtocolError, match="end reason"):
        encode_end_frame(0, "whatever")


# --- rule 7: one exception type out ----------------------------------------


def test_bad_sync_raises_protocol_error():
    with pytest.raises(KokoroWorkerProtocolError, match="sync"):
        _decode_all(_raw_frame({"seq": 0}, sync=b"XXXX"))


def test_unparseable_header_raises_protocol_error():
    """Bare JSONDecodeError would slip past the caller's except clause."""
    hb = b"{not json"
    blob = struct.pack(">4sBII", SYNC, TYPE_AUDIO, len(hb), 0) + hb
    with pytest.raises(KokoroWorkerProtocolError, match="header"):
        _decode_all(blob)


def test_non_object_header_raises_protocol_error():
    hb = b"[1,2,3]"
    blob = struct.pack(">4sBII", SYNC, TYPE_AUDIO, len(hb), 0) + hb
    with pytest.raises(KokoroWorkerProtocolError, match="JSON object"):
        _decode_all(blob)


def test_unknown_frame_type_raises_protocol_error():
    with pytest.raises(KokoroWorkerProtocolError, match="frame type"):
        _decode_all(_raw_frame({"seq": 0}, frame_type=0x7F))


def test_malformed_word_timestamps_raise_protocol_error():
    blob = _raw_frame(
        {
            "seq": 0,
            "shape": [0],
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "word_timestamps": [{"word": "x"}],  # missing times
        }
    )
    with pytest.raises(KokoroWorkerProtocolError, match="word_timestamps"):
        _decode_all(blob)


# --- rule 8: the peer is unauthenticated -----------------------------------


def test_oversize_payload_length_is_rejected_before_allocating():
    """A length field is an allocation request from an untrusted party."""
    hb = b"{}"
    blob = struct.pack(">4sBII", SYNC, TYPE_AUDIO, len(hb), MAX_PAYLOAD_BYTES + 1) + hb
    with pytest.raises(KokoroWorkerProtocolError, match="payload length"):
        _decode_all(blob)


def test_oversize_header_length_is_rejected_before_allocating():
    blob = struct.pack(">4sBII", SYNC, TYPE_AUDIO, (1 << 20) + 1, 0)
    with pytest.raises(KokoroWorkerProtocolError, match="header length"):
        _decode_all(blob)


def test_incomplete_frame_yields_nothing_rather_than_guessing():
    dec = FrameDecoder()
    blob = encode_audio_frame(_audio(64), seq=0)
    assert list(dec.feed(blob[:-4])) == []


# --- truncation and ordering -----------------------------------------------


def test_finish_raises_when_no_terminator_arrived():
    """The hard-kill case: bytes stop with no END."""
    dec, frames = _decode_all(encode_audio_frame(_audio(), seq=0))
    with pytest.raises(KokoroWorkerProtocolError, match="truncated"):
        dec.finish()


def test_finish_reports_how_much_arrived():
    dec, _ = _decode_all(
        encode_audio_frame(_audio(), seq=0), encode_audio_frame(_audio(), seq=1)
    )
    with pytest.raises(KokoroWorkerProtocolError, match="2 audio frame"):
        dec.finish()


def test_early_termination_is_not_a_protocol_error():
    """An assistant device hanging up mid-utterance is normal and frequent.

    `.ended` is the check for a caller that stopped reading on purpose;
    `finish()` is only for the transport-EOF path. Conflating them turned every
    user hangup into a logged fault.
    """
    dec, frames = _decode_all(encode_audio_frame(_audio(), seq=0))
    assert dec.ended is False  # caller inspects this and simply stops


def test_frame_count_mismatch_is_detected():
    with pytest.raises(KokoroWorkerProtocolError, match="declares"):
        _decode_all(encode_audio_frame(_audio(), seq=0), encode_end_frame(5))


def test_out_of_order_seq_is_detected():
    with pytest.raises(KokoroWorkerProtocolError, match="seq"):
        _decode_all(encode_audio_frame(_audio(), seq=0), encode_audio_frame(_audio(), seq=7))


def test_data_after_end_is_rejected():
    dec, _ = _decode_all(encode_end_frame(0))
    with pytest.raises(KokoroWorkerProtocolError, match="after the stream ended"):
        list(dec.feed(encode_audio_frame(_audio(), seq=0)))


def test_trailing_bytes_after_terminator_are_rejected():
    dec = FrameDecoder()
    list(dec.feed(encode_end_frame(0) + b"\x00\x01"))
    with pytest.raises(KokoroWorkerProtocolError, match="trailing"):
        dec.finish()


# --- error frames -----------------------------------------------------------


def test_error_frame_round_trips_and_terminates_the_stream():
    dec, frames = _decode_all(encode_error_frame("RuntimeError", "CUDA OOM", {"retryable": True}))
    err = frames[0]
    assert isinstance(err, ErrorFrame)
    assert err.error_class == "RuntimeError"
    assert err.message == "CUDA OOM"
    assert err.detail == {"retryable": True}
    dec.finish()  # an ERROR is a legitimate terminator


def test_error_frame_class_is_not_a_closed_allowlist():
    """A fixed list of four builtins would mistranslate anything unforeseen."""
    _, frames = _decode_all(encode_error_frame("KokoroModelUnavailable", "nope"))
    assert frames[0].error_class == "KokoroModelUnavailable"
