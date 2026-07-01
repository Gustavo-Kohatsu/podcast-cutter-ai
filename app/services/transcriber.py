"""Audio transcription service using faster-whisper.

Responsible for transcribing an audio file locally using the faster-whisper
library, which runs OpenAI's Whisper model with CTranslate2 optimizations.

Key characteristics:
- Runs 100% locally with no API calls or costs.
- Supports quantization (INT8) for fast CPU inference.
- Lazy-loads the model on first use to avoid memory allocation on import.
- Returns a fully typed Transcript object, hiding faster-whisper internals.
"""

from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import TranscriptionInfo

from app.config.settings import Settings
from app.schemas.transcript import Transcript, TranscriptSegment
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Transcriber:
    """Transcribes audio files to text with timestamps using faster-whisper.

    The Whisper model is loaded lazily on the first call to ``transcribe()``.
    This avoids consuming significant memory/time if an earlier pipeline step
    (like download) fails before transcription is reached.

    Args:
        settings: Application settings containing Whisper model configuration.

    Example:
        >>> transcriber = Transcriber(settings)
        >>> transcript = transcriber.transcribe(audio_path, "dQw4w9WgXcQ", url)
        >>> print(f"Language: {transcript.language}, Segments: {transcript.segment_count}")
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None

    def transcribe(self, audio_path: Path, video_id: str, url: str) -> Transcript:
        """Transcribe an audio file, save the result to disk and return it.

        Loads the Whisper model on first call (lazy initialization).
        Saves the transcript as JSON to ``settings.transcripts_dir/{video_id}.json``
        so it can be inspected or reused without re-running Whisper.

        Args:
            audio_path: Path to the audio file to transcribe.
            video_id: YouTube video ID used as filename and identifier.
            url: Original YouTube URL stored as metadata in the transcript.

        Returns:
            A fully populated Transcript object with timed segments.

        Raises:
            FileNotFoundError: If the audio file does not exist at ``audio_path``.
            RuntimeError: If faster-whisper fails to load or process the audio.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Return cached transcript if it already exists — avoids re-running Whisper
        cached = self._load_cached(video_id)
        if cached is not None:
            return cached

        model = self._get_model()

        logger.info(
            "Transcribing '%s' with model '%s' on %s...",
            audio_path.name,
            self._settings.whisper_model,
            self._settings.whisper_device,
        )

        segments_generator, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=False,
            language=self._settings.whisper_language or None,
            # VAD filter removes silence, reducing hallucinations on quiet passages
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        logger.info(
            "Detected language: '%s' (confidence: %.1f%%)",
            info.language,
            info.language_probability * 100,
        )

        segments = self._collect_segments(segments_generator, info)

        transcript = Transcript(
            video_id=video_id,
            url=url,
            language=info.language,
            duration=info.duration,
            segments=segments,
        )

        logger.info(
            "Transcription complete: %d segments, %.1fs duration",
            transcript.segment_count,
            transcript.duration,
        )

        self._save(transcript)

        return transcript

    def _load_cached(self, video_id: str) -> Transcript | None:
        """Load a previously saved transcript from disk if it exists.

        Args:
            video_id: YouTube video ID used as the filename key.

        Returns:
            A Transcript instance loaded from cache, or None if not found.
        """
        cached_path = self._settings.transcripts_dir / f"{video_id}.json"
        if not cached_path.exists():
            return None

        logger.info(
            "Transcript cache found, skipping Whisper — loading from: %s",
            cached_path,
        )
        return Transcript.model_validate_json(cached_path.read_text(encoding="utf-8"))

    def _save(self, transcript: Transcript) -> Path:
        """Persist the transcript as a JSON file in the transcripts directory.

        The file is saved to ``storage/transcripts/{video_id}.json``.
        The directory is created automatically if it does not exist.

        Args:
            transcript: The transcript to persist.

        Returns:
            Path to the saved JSON file.
        """
        output_dir = self._settings.transcripts_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{transcript.video_id}.json"
        output_path.write_text(
            transcript.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info("Transcript saved: %s", output_path)
        return output_path

    def _get_model(self) -> WhisperModel:
        """Return the loaded WhisperModel, initializing it on first access.

        Returns:
            A ready-to-use WhisperModel instance.
        """
        if self._model is None:
            logger.info(
                "Loading Whisper model '%s' (%s / %s)...",
                self._settings.whisper_model,
                self._settings.whisper_device,
                self._settings.whisper_compute_type,
            )
            self._model = WhisperModel(
                self._settings.whisper_model,
                device=self._settings.whisper_device,
                compute_type=self._settings.whisper_compute_type,
            )
            logger.info("Whisper model loaded successfully.")
        return self._model

    def _collect_segments(
        self,
        segments_generator,
        info: TranscriptionInfo,
    ) -> list[TranscriptSegment]:
        """Consume the faster-whisper segments generator into a typed list.

        faster-whisper returns a lazy generator. We consume it fully here,
        logging progress so the user knows transcription is running.

        Args:
            segments_generator: Lazy generator of faster-whisper Segment objects.
            info: Transcription metadata (duration used for progress display).

        Returns:
            List of TranscriptSegment instances.
        """
        segments: list[TranscriptSegment] = []
        total_duration = info.duration or 1.0  # avoid division by zero

        for raw_segment in segments_generator:
            text = raw_segment.text.strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start=round(raw_segment.start, 2),
                    end=round(raw_segment.end, 2),
                    text=text,
                )
            )

            # Log progress every ~5 minutes of transcribed audio
            if len(segments) % 50 == 0:
                progress = (raw_segment.end / total_duration) * 100
                logger.debug(
                    "Transcription progress: %.1f%% (segment %d)",
                    progress,
                    len(segments),
                )

        return segments
