"""TTS service using model and voice managers."""

import asyncio
import os
import re
import tempfile
import time
from typing import AsyncGenerator, List, Optional, Tuple, Union

import numpy as np
import torch
from kokoro import KPipeline
from loguru import logger

from ..core.config import settings
from ..inference.base import AudioChunk
from ..inference.model_manager import get_manager as get_model_manager
from ..inference.voice_manager import get_manager as get_voice_manager
from ..inference.vram_lease import track_synthesis
from ..structures.schemas import NormalizationOptions
from .audio import AudioNormalizer, AudioService
from .streaming_audio_writer import StreamingAudioWriter
from .text_processing import tokenize
from .text_processing.text_processor import process_text_chunk, smart_split


def _note_dropped_audio(failures: Optional[List[str]], message: str) -> None:
    """Record audio we failed to produce, in addition to logging it.

    Every drop site in the streaming path used to call ``logger.error`` and
    carry on. That is why self.speak#6 happened: a generation failure produced
    a 200 with a valid, playable, silently SHORTER file, and no consumer —
    assistant device, self.chat, a caller's retry logic — had any way to tell
    a complete synthesis from a truncated one.

    Logging is for the operator. This list is for the protocol: the stream
    consults it before writing the container trailer, and refuses to finish
    cleanly when anything was lost. Callers that only want the old
    log-and-continue behaviour pass ``None``.
    """
    logger.error(message)
    if failures is not None:
        failures.append(message)


class TTSService:
    """Text-to-speech service."""

    # Limit concurrent chunk processing
    _chunk_semaphore = asyncio.Semaphore(4)

    def __init__(self, output_dir: str = None):
        """Initialize service."""
        self.output_dir = output_dir
        self.model_manager = None
        self._voice_manager = None

    @classmethod
    async def create(cls, output_dir: str = None) -> "TTSService":
        """Create and initialize TTSService instance."""
        service = cls(output_dir)
        service.model_manager = await get_model_manager()
        service._voice_manager = await get_voice_manager()
        return service

    async def _process_chunk(
        self,
        chunk_text: str,
        tokens: List[int],
        voice_name: str,
        voice_path: str,
        speed: float,
        writer: StreamingAudioWriter,
        output_format: Optional[str] = None,
        is_first: bool = False,
        is_last: bool = False,
        volume_multiplier: Optional[float] = 1.0,
        normalizer: Optional[AudioNormalizer] = None,
        lang_code: Optional[str] = None,
        return_timestamps: Optional[bool] = False,
        failures: Optional[List[str]] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Process tokens into audio.

        ``failures`` collects any audio this chunk was asked for but could not
        produce. It is append-only and owned by the caller; see
        ``_note_dropped_audio``.
        """
        async with self._chunk_semaphore:
            try:
                # Handle stream finalization
                if is_last:
                    # Skip format conversion for raw audio mode
                    if not output_format:
                        yield AudioChunk(np.array([], dtype=np.int16), output=b"")
                        return
                    chunk_data = await AudioService.convert_audio(
                        AudioChunk(
                            np.array([], dtype=np.float32)
                        ),  # Dummy data for type checking
                        output_format,
                        writer,
                        speed,
                        "",
                        normalizer=normalizer,
                        is_last_chunk=True,
                    )
                    yield chunk_data
                    return

                # Skip empty chunks
                if not tokens and not chunk_text:
                    return

                # Get backend
                backend = self.model_manager.get_backend()

                # Generate audio using pre-warmed model
                # Capability, not concrete class. This used to be
                # `isinstance(backend, KokoroV1)`, which silently means "is the
                # IN-PROCESS class" -- so a worker-backed backend would have
                # fallen through to the legacy tokens-in/one-blob-out branch and
                # produced wrong audio instead of an error. The question is about
                # the contract (takes text, yields chunks), so ask it directly.
                if getattr(backend, "streams_text", False):
                    chunk_index = 0
                    # For Kokoro V1, pass text and voice info with lang_code
                    async for chunk_data in self.model_manager.generate(
                        chunk_text,
                        (voice_name, voice_path),
                        speed=speed,
                        lang_code=lang_code,
                        return_timestamps=return_timestamps,
                    ):
                        chunk_data.audio*=volume_multiplier
                        # For streaming, convert to bytes
                        if output_format:
                            try:
                                chunk_data = await AudioService.convert_audio(
                                    chunk_data,
                                    output_format,
                                    writer,
                                    speed,
                                    chunk_text,
                                    is_last_chunk=is_last,
                                    normalizer=normalizer,
                                )
                                yield chunk_data
                            except Exception as e:
                                _note_dropped_audio(
                                    failures, f"Failed to convert audio: {str(e)}"
                                )
                        else:
                            chunk_data = AudioService.trim_audio(
                                chunk_data, chunk_text, speed, is_last, normalizer
                            )
                            yield chunk_data
                        chunk_index += 1
                else:
                    # For legacy backends, load voice tensor
                    voice_tensor = await self._voice_manager.load_voice(
                        voice_name, device=backend.device
                    )
                    chunk_data = await self.model_manager.generate(
                        tokens,
                        voice_tensor,
                        speed=speed,
                        return_timestamps=return_timestamps,
                    )
                    
                    if chunk_data.audio is None:
                        _note_dropped_audio(
                            failures, "Model generated None for audio chunk"
                        )
                        return

                    if len(chunk_data.audio) == 0:
                        _note_dropped_audio(
                            failures, "Model generated empty audio chunk"
                        )
                        return

                    chunk_data.audio*=volume_multiplier

                    # For streaming, convert to bytes
                    if output_format:
                        try:
                            chunk_data = await AudioService.convert_audio(
                                chunk_data,
                                output_format,
                                writer,
                                speed,
                                chunk_text,
                                normalizer=normalizer,
                                is_last_chunk=is_last,
                            )
                            yield chunk_data
                        except Exception as e:
                            _note_dropped_audio(
                                failures, f"Failed to convert audio: {str(e)}"
                            )
                    else:
                        trimmed = AudioService.trim_audio(
                            chunk_data, chunk_text, speed, is_last, normalizer
                        )
                        yield trimmed
            except Exception as e:
                # Still swallowed on purpose — one bad chunk must not abort a
                # long synthesis. What changed is that it is no longer SILENT:
                # the caller now knows audio is missing and will refuse to
                # finalize the stream as if it were complete.
                _note_dropped_audio(failures, f"Failed to process tokens: {str(e)}")

    async def _load_voice_from_path(self, path: str, weight: float):
        # Check if the path is None and raise a ValueError if it is not
        if not path:
            raise ValueError(f"Voice not found at path: {path}")

        logger.debug(f"Loading voice tensor from path: {path}")
        return torch.load(path, map_location="cpu") * weight

    async def _get_voices_path(self, voice: str) -> Tuple[str, str]:
        """Get voice path, handling combined voices.

        Args:
            voice: Voice name or combined voice names (e.g., 'af_jadzia+af_jessica')

        Returns:
            Tuple of (voice name to use, voice path to use)

        Raises:
            RuntimeError: If voice not found
        """
        try:
            # Split the voice on + and - and ensure that they get added to the list eg: hi+bob = ["hi","+","bob"]
            split_voice = re.split(r"([-+])", voice)

            # If it is only once voice there is no point in loading it up, doing nothing with it, then saving it
            if len(split_voice) == 1:
                # Since its a single voice the only time that the weight would matter is if voice_weight_normalization is off
                if (
                    "(" not in voice and ")" not in voice
                ) or settings.voice_weight_normalization == True:
                    path = await self._voice_manager.get_voice_path(voice)
                    if not path:
                        raise RuntimeError(f"Voice not found: {voice}")
                    logger.debug(f"Using single voice path: {path}")
                    return voice, path

            total_weight = 0

            for voice_index in range(0, len(split_voice), 2):
                voice_object = split_voice[voice_index]

                if "(" in voice_object and ")" in voice_object:
                    voice_name = voice_object.split("(")[0].strip()
                    voice_weight = float(voice_object.split("(")[1].split(")")[0])
                else:
                    voice_name = voice_object
                    voice_weight = 1

                total_weight += voice_weight
                split_voice[voice_index] = (voice_name, voice_weight)

            # If voice_weight_normalization is false prevent normalizing the weights by setting the total_weight to 1 so it divides each weight by 1
            if settings.voice_weight_normalization == False:
                total_weight = 1

            # Load the first voice as the starting point for voices to be combined onto
            path = await self._voice_manager.get_voice_path(split_voice[0][0])
            combined_tensor = await self._load_voice_from_path(
                path, split_voice[0][1] / total_weight
            )

            # Loop through each + or - in split_voice so they can be applied to combined voice
            for operation_index in range(1, len(split_voice) - 1, 2):
                # Get the voice path of the voice 1 index ahead of the operator
                path = await self._voice_manager.get_voice_path(
                    split_voice[operation_index + 1][0]
                )
                voice_tensor = await self._load_voice_from_path(
                    path, split_voice[operation_index + 1][1] / total_weight
                )

                # Either add or subtract the voice from the current combined voice
                if split_voice[operation_index] == "+":
                    combined_tensor += voice_tensor
                else:
                    combined_tensor -= voice_tensor

            # Save the new combined voice so it can be loaded latter
            temp_dir = tempfile.gettempdir()
            combined_path = os.path.join(temp_dir, f"{voice}.pt")
            logger.debug(f"Saving combined voice to: {combined_path}")
            torch.save(combined_tensor, combined_path)
            return voice, combined_path
        except Exception as e:
            logger.error(f"Failed to get voice path: {e}")
            raise

    async def generate_audio_stream(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        volume_multiplier: Optional[float] = 1.0,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
        engine: str = "kokoro",
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        audio_prompt_path: Optional[str] = None,
        references: Optional["list[tuple[str, float]]"] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate and stream audio chunks.

        Wraps the whole synthesis lifetime in ``track_synthesis()`` (cavekit
        vram-lease-client R2-AC2) so a VRAM release-request can detect in-flight
        synthesis and wait for it to drain before unloading the model. This is
        the single choke-point through which the full-audio ``generate_audio``
        path also flows, so every ``POST /v1/audio/speech`` — streaming or full
        — is counted for its whole duration. The context manager's ``finally``
        runs when this generator is exhausted OR closed (GeneratorExit), so the
        counter is always decremented, even if the client disconnects mid-stream.

        ``engine`` selects the backend. ``"kokoro"`` (default) is the unchanged
        in-process Kokoro path. ``"chatterbox"`` proxies to the sibling worker
        process — and CRITICALLY does so from INSIDE this same ``track_synthesis()``
        wrapper (INTEGRATION-PLAN-v2.md §1.4 hard dependency): the in-flight
        counter must reflect Chatterbox work too, or a VRAM release could unload
        the worker mid-synthesis with no drain protection.
        """
        async with track_synthesis():
            if engine == "chatterbox":
                async for chunk in self._chatterbox_audio_stream_impl(
                    text,
                    writer,
                    output_format=output_format,
                    volume_multiplier=volume_multiplier,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    audio_prompt_path=audio_prompt_path,
                    references=references,
                ):
                    yield chunk
            elif engine == "kokoro":
                async for chunk in self._generate_audio_stream_impl(
                    text,
                    voice,
                    writer,
                    speed=speed,
                    output_format=output_format,
                    lang_code=lang_code,
                    volume_multiplier=volume_multiplier,
                    normalization_options=normalization_options,
                    return_timestamps=return_timestamps,
                ):
                    yield chunk
            else:
                # Engine-mismatch guard: never silently fall through to Kokoro for
                # an unknown engine (that would mask a routing/config bug).
                raise ValueError(f"Unsupported TTS engine: {engine!r}")

    async def _chatterbox_audio_stream_impl(
        self,
        text: str,
        writer: StreamingAudioWriter,
        output_format: Optional[str] = None,
        volume_multiplier: Optional[float] = 1.0,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        audio_prompt_path: Optional[str] = None,
        references: Optional["list[tuple[str, float]]"] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Chatterbox path: one localhost proxy call → one raw waveform → one
        AudioChunk through the SAME ``StreamingAudioWriter`` the Kokoro path uses.

        The worker returns bare float32 PCM + an ``X-Sample-Rate`` header; all
        mp3/opus/wav encoding stays here (single-sourced, torch-version-
        independent — §1.4). With no ``audio_prompt_path`` this is the default
        voice (``/synth``); with one it is a zero-shot clone in the reference
        clip's voice (``/clone``) — the Phase-3 clone/preview path. Either way the
        call is made from inside ``track_synthesis()`` by the caller.
        """
        # Imported lazily so the Kokoro-only import graph / VRAM probe path never
        # drags httpx in at module load.
        from ..inference import chatterbox_client

        if references and len(references) >= 2:
            # Multi-clip blend → a NEW interpolated voice (worker /blend).
            pcm_bytes, sr = await chatterbox_client.blend(
                text,
                references,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        elif references and len(references) == 1:
            # A single "blend" reference is just a clone of that clip.
            pcm_bytes, sr = await chatterbox_client.clone(
                text,
                references[0][0],
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        elif audio_prompt_path is not None:
            pcm_bytes, sr = await chatterbox_client.clone(
                text,
                audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
        else:
            pcm_bytes, sr = await chatterbox_client.synth(
                text, exaggeration=exaggeration, cfg_weight=cfg_weight
            )

        if sr != writer.sample_rate:
            # The encoder was built at settings.chatterbox_sample_rate; a mismatch
            # would pitch-shift the output. Log loudly (Phase 1 does not resample).
            logger.warning(
                f"Chatterbox worker sample rate {sr} != writer rate "
                f"{writer.sample_rate}; audio may be mis-pitched"
            )

        audio = np.frombuffer(pcm_bytes, dtype=np.float32)
        if volume_multiplier is not None and volume_multiplier != 1.0:
            audio = audio * np.float32(volume_multiplier)
        # np.frombuffer yields a read-only view; downstream normalize/encode want a
        # writable, owned array.
        audio = np.array(audio, dtype=np.float32)

        chunk = AudioChunk(audio=audio, word_timestamps=[])

        if output_format:
            # Encode in TWO calls, exactly like create_speech's non-stream path:
            # the FIRST (is_last_chunk=False) yields the encoded audio BODY as
            # .output; the SECOND (empty audio, is_last_chunk=True) yields the
            # container TRAILER. A single is_last_chunk=True call would return ONLY
            # the trailer — StreamingAudioWriter truncates its buffer between
            # writes, so the body would be lost. Yielding both chunks lets the
            # router's stream concatenator (single_output / dual_output) emit a
            # complete, valid file for every format (mp3/opus/flac/aac/wav/pcm).
            body = await AudioService.convert_audio(
                chunk,
                output_format,
                writer,
                speed=1.0,
                chunk_text="",
                is_last_chunk=False,
                trim_audio=False,
                normalizer=AudioNormalizer(),
            )
            yield body
            final = await AudioService.convert_audio(
                AudioChunk(np.array([], dtype=np.int16)),
                output_format,
                writer,
                is_last_chunk=True,
            )
            yield final
        else:
            # Raw mode (the generate_audio collector path): yield int16 samples so
            # AudioChunk.combine (which concatenates as int16) accepts them.
            chunk.audio = AudioNormalizer().normalize(chunk.audio)
            yield chunk

    async def _generate_audio_stream_impl(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        output_format: str = "wav",
        lang_code: Optional[str] = None,
        volume_multiplier: Optional[float] = 1.0,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        return_timestamps: Optional[bool] = False,
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate and stream audio chunks (implementation)."""
        stream_normalizer = AudioNormalizer()
        chunk_index = 0
        current_offset = 0.0
        # Audio this request asked for and did not get. Non-empty means the
        # stream must NOT be finalized as a well-formed file (self.speak#6).
        failures: List[str] = []
        try:
            # Lazily (re)load the model if a VRAM-lease release unloaded it —
            # otherwise every synthesis after a release 500s "Backend not
            # initialized" until pod restart. Within the track_synthesis() scope
            # (the caller), so a concurrent release drain-waits for this reload.
            await self.model_manager.ensure_loaded()

            # Get backend
            backend = self.model_manager.get_backend()

            # Get voice path, handling combined voices
            voice_name, voice_path = await self._get_voices_path(voice)
            logger.debug(f"Using voice path: {voice_path}")

            # Use provided lang_code or determine from voice name
            pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
            logger.info(
                f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in audio stream"
            )

            # Process text in chunks with smart splitting, handling pause tags
            async for chunk_text, tokens, pause_duration_s in smart_split(
                text,
                lang_code=pipeline_lang_code,
                normalization_options=normalization_options,
            ):
                if pause_duration_s is not None and pause_duration_s > 0:
                    # --- Handle Pause Chunk ---
                    try:
                        logger.debug(f"Generating {pause_duration_s}s silence chunk")
                        silence_samples = int(pause_duration_s * 24000)  # 24kHz sample rate
                        # Create proper silence as int16 zeros to avoid normalization artifacts
                        silence_audio = np.zeros(silence_samples, dtype=np.int16)
                        pause_chunk = AudioChunk(audio=silence_audio, word_timestamps=[])  # Empty timestamps for silence

                        # Format and yield the silence chunk
                        if output_format:
                            formatted_pause_chunk = await AudioService.convert_audio(
                                pause_chunk, output_format, writer, speed=speed, chunk_text="",
                                is_last_chunk=False, trim_audio=False, normalizer=stream_normalizer,

                            )
                            if formatted_pause_chunk.output:
                                yield formatted_pause_chunk
                        else:  # Raw audio mode
                            # For raw audio mode, silence is already in the correct format (int16)
                            # Skip normalization to avoid any potential artifacts
                            if len(pause_chunk.audio) > 0:
                                yield pause_chunk

                        # Update offset based on silence duration
                        current_offset += pause_duration_s
                        chunk_index += 1  # Count pause as a yielded chunk

                    except Exception as e:
                        _note_dropped_audio(
                            failures, f"Failed to process pause chunk: {str(e)}"
                        )
                        continue

                elif tokens or chunk_text.strip():  # Process if there are tokens OR non-whitespace text
                    # --- Handle Text Chunk ---
                    try:
                        # Process audio for chunk
                        async for chunk_data in self._process_chunk(
                            chunk_text,  # Pass text for Kokoro V1
                            tokens,  # Pass tokens for legacy backends
                            voice_name,  # Pass voice name
                            voice_path,  # Pass voice path
                            speed,
                            writer,
                            output_format,
                            is_first=(chunk_index == 0),
                            volume_multiplier=volume_multiplier,
                            is_last=False,  # We'll update the last chunk later
                            normalizer=stream_normalizer,
                            lang_code=pipeline_lang_code,  # Pass lang_code
                            return_timestamps=return_timestamps,
                            failures=failures,
                        ):
                            if chunk_data.word_timestamps is not None:
                                for timestamp in chunk_data.word_timestamps:
                                    timestamp.start_time += current_offset
                                    timestamp.end_time += current_offset

                            # Update offset based on the actual duration of the generated audio chunk
                            chunk_duration = 0
                            if chunk_data.audio is not None and len(chunk_data.audio) > 0:
                                chunk_duration = len(chunk_data.audio) / 24000
                                current_offset += chunk_duration

                            # Yield the processed chunk (either formatted or raw)
                            if chunk_data.output is not None:
                                yield chunk_data
                            elif chunk_data.audio is not None and len(chunk_data.audio) > 0:
                                yield chunk_data
                            else:
                                logger.warning(
                                    f"No audio generated for chunk: '{chunk_text[:100]}...'"
                                )

                        chunk_index += 1  # Increment chunk index after processing text
                    except Exception as e:
                        _note_dropped_audio(
                            failures,
                            f"Failed to process audio for chunk: '{chunk_text[:100]}...'. Error: {str(e)}",
                        )
                        continue

            # Refuse to finalize a synthesis that lost audio.
            #
            # The HTTP status is committed at the first byte, so a truncated
            # result cannot be reported as a 4xx/5xx once streaming has begun.
            # The only in-band signal left is an ABNORMAL END OF BODY: raising
            # here means no container trailer is written and the chunked
            # transfer terminates without its final chunk, so the client sees a
            # broken transfer instead of a well-formed short file.
            #
            # Deliberately AFTER the chunk loop, not inside it: a single bad
            # chunk still does not stop the remaining ones from being produced
            # and sent. What the caller loses is only the false claim that what
            # it received was everything it asked for.
            #
            # The non-streaming collector path (generate_audio) consumes this
            # generator before any response is committed, so there the same
            # raise surfaces as an ordinary 500.
            if failures:
                raise RuntimeError(
                    f"Synthesis incomplete: {len(failures)} chunk(s) produced no audio "
                    f"({chunk_index} chunk(s) delivered). First failure: {failures[0]}"
                )

            # Only finalize if we successfully processed at least one chunk
            if chunk_index > 0:
                try:
                    # Empty tokens list to finalize audio
                    async for chunk_data in self._process_chunk(
                        "",  # Empty text
                        [],  # Empty tokens
                        voice_name,
                        voice_path,
                        speed,
                        writer,
                        output_format,
                        is_first=False,
                        is_last=True,  # Signal this is the last chunk
                        volume_multiplier=volume_multiplier,
                        normalizer=stream_normalizer,
                        lang_code=pipeline_lang_code,  # Pass lang_code
                    ):
                        if chunk_data.output is not None:
                            yield chunk_data
                except Exception as e:
                    # A failed finalization is the same failure mode as above:
                    # the bytes already sent have no valid trailer, so swallowing
                    # this would hand the client a corrupt file under a 200.
                    logger.error(f"Failed to finalize audio stream: {str(e)}")
                    raise

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise e

    async def generate_audio(
        self,
        text: str,
        voice: str,
        writer: StreamingAudioWriter,
        speed: float = 1.0,
        return_timestamps: bool = False,
        volume_multiplier: Optional[float] = 1.0,
        normalization_options: Optional[NormalizationOptions] = NormalizationOptions(),
        lang_code: Optional[str] = None,
        engine: str = "kokoro",
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        audio_prompt_path: Optional[str] = None,
        references: Optional["list[tuple[str, float]]"] = None,
    ) -> AudioChunk:
        """Generate complete audio for text using streaming internally."""
        audio_data_chunks = []

        try:
            async for audio_stream_data in self.generate_audio_stream(
                text,
                voice,
                writer,
                speed=speed,
                volume_multiplier=volume_multiplier,
                normalization_options=normalization_options,
                return_timestamps=return_timestamps,
                lang_code=lang_code,
                output_format=None,
                engine=engine,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                audio_prompt_path=audio_prompt_path,
                references=references,
            ):
                if len(audio_stream_data.audio) > 0:
                    audio_data_chunks.append(audio_stream_data)

            combined_audio_data = AudioChunk.combine(audio_data_chunks)
            return combined_audio_data
        except Exception as e:
            logger.error(f"Error in audio generation: {str(e)}")
            raise

    async def combine_voices(self, voices: List[str]) -> torch.Tensor:
        """Combine multiple voices.

        Returns:
            Combined voice tensor
        """

        return await self._voice_manager.combine_voices(voices)

    async def list_voices(self) -> List[str]:
        """List available voices."""
        return await self._voice_manager.list_voices()

    async def generate_from_phonemes(
        self,
        phonemes: str,
        voice: str,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
    ) -> Tuple[np.ndarray, float]:
        """Generate audio directly from phonemes.

        Args:
            phonemes: Phonemes in Kokoro format
            voice: Voice name
            speed: Speed multiplier
            lang_code: Optional language code override

        Returns:
            Tuple of (audio array, processing time)
        """
        start_time = time.time()
        try:
            # Get backend and voice path
            backend = self.model_manager.get_backend()
            voice_name, voice_path = await self._get_voices_path(voice)

            # Ask the BACKEND, not its internals. This used to reach into
            # backend._get_pipeline(...).generate_from_tokens(...), which was
            # fine while there was only ever one in-process backend and became a
            # dead route the moment the model moved to a worker (self.speak#7):
            # an internal cannot cross a process boundary. Both backends now
            # implement generate_from_phonemes, so this path stops caring where
            # the model lives.
            if hasattr(backend, "generate_from_phonemes"):
                pipeline_lang_code = lang_code if lang_code else voice[:1].lower()
                logger.info(
                    f"Using lang_code '{pipeline_lang_code}' for voice '{voice_name}' in phoneme pipeline"
                )
                try:
                    audio = await backend.generate_from_phonemes(
                        phonemes, voice_path, speed, pipeline_lang_code
                    )
                except Exception as e:
                    logger.error(f"Failed to generate from phonemes: {e}")
                    raise RuntimeError(f"Phoneme generation failed: {e}") from e

                if audio is None or len(audio) == 0:
                    raise ValueError("No audio generated")

                return audio, time.time() - start_time
            else:
                raise ValueError(
                    "Phoneme generation only supported with Kokoro V1 backend"
                )

        except Exception as e:
            logger.error(f"Error in phoneme audio generation: {str(e)}")
            raise
