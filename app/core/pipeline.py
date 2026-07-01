"""Main pipeline orchestrator.

This module is the single point of coordination for the entire processing pipeline.
It owns the execution order and data flow between services — no service knows about
or calls any other service.

Responsibilities:
- Define and enforce the step-by-step execution order.
- Pass the output of each step as input to the next.
- Persist intermediate and final artifacts to disk.
- Emit structured log messages for each stage.
- Provide a clear entry point for future async/worker migration.

Future evolution path:
    When migrating to async workers (Celery, RQ, etc.), this class can be
    split into individual task functions, one per step, with each step
    publishing a message to the next queue on completion.
"""

import time
from pathlib import Path

from app.config.settings import Settings
from app.schemas.clip import ClipDetectionResult
from app.schemas.transcript import Transcript
from app.services.clip_detector import ClipDetector
from app.services.downloader import AudioDownloader
from app.services.transcriber import Transcriber
from app.services.video_cutter import VideoCutter
from app.services.video_downloader import VideoDownloader
from app.utils.logger import get_logger
from app.utils.validators import extract_video_id

logger = get_logger(__name__)


class Pipeline:
    """Orchestrates the full viral clip detection pipeline.

    Coordinates five services in sequence:
    1. AudioDownloader  — downloads audio from YouTube
    2. Transcriber      — transcribes audio to text with timestamps
    3. ClipDetector     — identifies viral clip candidates via LLM
    4. VideoDownloader  — downloads the full video
    5. VideoCutter      — extracts each clip as a standalone MP4

    Args:
        settings: Application settings, shared across all services.

    Example:
        >>> pipeline = Pipeline(settings)
        >>> result_path = pipeline.run("https://youtu.be/...")
        >>> print(f"Results saved to: {result_path}")
    """

    _TOTAL_STEPS = 5

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._downloader = AudioDownloader(settings)
        self._transcriber = Transcriber(settings)
        self._clip_detector = ClipDetector(settings)
        self._video_downloader = VideoDownloader(settings)
        self._cutter = VideoCutter(settings)

    def run(self, url: str) -> Path:
        """Execute the full pipeline for a given YouTube URL.

        Steps:
        1. Download audio       → storage/audio/{video_id}/
        2. Transcribe audio     → storage/transcripts/{video_id}.json
        3. Detect viral clips   → storage/jobs/{video_id}.json
        4. Download full video  → storage/videos/{video_id}/
        5. Cut clips            → storage/cuts/{video_id}/cut_001.mp4 …

        The video download (step 4) is intentionally placed after clip
        detection so the bandwidth cost is only incurred when clips are
        confirmed. Step 5 then cuts each confirmed clip from that video.

        Args:
            url: A validated YouTube URL.

        Returns:
            Path to the saved ClipDetectionResult JSON file.

        Raises:
            ValueError: If the URL is invalid (should be caught by main.py).
            FileNotFoundError: If the audio or video file cannot be found
                after download.
            ConnectionError: If Ollama is not running.
            RuntimeError: If Ollama model is not available.
        """
        pipeline_start = time.perf_counter()
        video_id = extract_video_id(url)

        logger.info("=" * 60)
        logger.info("PIPELINE START — video_id: %s", video_id)
        logger.info("=" * 60)

        # ── Step 1: Download Audio ────────────────────────────────────────────
        audio_path = self._run_step(
            step_number=1,
            step_name="Audio Download",
            fn=lambda: self._downloader.download(url, video_id),
        )

        # ── Step 2: Transcribe ────────────────────────────────────────────────
        transcript: Transcript = self._run_step(
            step_number=2,
            step_name="Transcription",
            fn=lambda: self._transcriber.transcribe(audio_path, video_id, url),
        )

        # ── Step 3: Detect Viral Clips ────────────────────────────────────────
        result: ClipDetectionResult = self._run_step(
            step_number=3,
            step_name="Clip Detection",
            fn=lambda: self._clip_detector.detect(transcript),
        )

        # ── Step 4: Download Full Video ───────────────────────────────────────
        video_path: Path = self._run_step(
            step_number=4,
            step_name="Video Download",
            fn=lambda: self._video_downloader.download(url, video_id),
        )

        # ── Step 5: Cut Clips ─────────────────────────────────────────────────
        cut_paths: list[Path] = self._run_step(
            step_number=5,
            step_name="Clip Cutting",
            fn=lambda: self._cutter.cut(video_path, result, video_id),
        )

        result_path = self._settings.jobs_dir / f"{video_id}.json"

        elapsed = time.perf_counter() - pipeline_start
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
        logger.info("%s", result.summary())
        logger.info(
            "%d cut(s) saved to: %s",
            len(cut_paths),
            self._settings.cuts_dir / video_id,
        )
        logger.info("Results saved to: %s", result_path)
        logger.info("=" * 60)

        return result_path

    def _run_step(self, step_number: int, step_name: str, fn) -> object:
        """Execute a pipeline step with timing and structured logging.

        Args:
            step_number: The step's position in the pipeline (for display).
            step_name: Human-readable name of the step.
            fn: Callable that executes the step and returns its result.

        Returns:
            The return value of ``fn()``.

        Raises:
            Exception: Re-raises any exception from ``fn()`` after logging it.
        """
        logger.info("── Step %d/%d: %s", step_number, self._TOTAL_STEPS, step_name)
        step_start = time.perf_counter()

        try:
            result = fn()
            elapsed = time.perf_counter() - step_start
            logger.info("   ✓ %s completed in %.1fs", step_name, elapsed)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - step_start
            logger.error(
                "   ✗ %s FAILED after %.1fs: %s",
                step_name,
                elapsed,
                exc,
                exc_info=True,
            )
            raise
