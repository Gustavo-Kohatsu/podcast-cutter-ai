"""Pipeline orchestrator — the only component that knows stage order.

The orchestrator's sole responsibility is to:
1. Create an initial PipelineContext from the user-supplied URL.
2. Execute each Stage in sequence, passing the context through.
3. Log timing and progress for every stage.
4. Propagate and log failures so the user gets actionable error messages.

It knows NOTHING about how individual stages work internally.  Swapping,
adding, or removing a stage only requires changing ``create_default_pipeline``
— the orchestrator class itself never changes.

Design notes
------------
- ``PipelineOrchestrator`` receives a list of ``Stage`` instances; it does not
  create them.  This is Dependency Injection: the orchestrator is completely
  decoupled from concrete implementations.
- ``create_default_pipeline`` is the *composition root* — the one place where
  concrete service and stage classes are imported and wired together.
- ``Pipeline`` (at the bottom) is a backward-compatibility shim so that
  existing code importing ``from app.core.pipeline import Pipeline`` still works
  without modification.
"""

from __future__ import annotations

import time
from pathlib import Path

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.utils.logger import get_logger
from app.utils.validators import extract_video_id

logger = get_logger(__name__)


class PipelineOrchestrator:
    """Executes an ordered list of stages, threading a shared context.

    Args:
        stages: Ordered list of Stage instances to execute.
        settings: Application settings (used for final log messages and the
            result-file path returned to the caller).

    Example:
        >>> pipeline = create_default_pipeline(settings)
        >>> result_path = pipeline.run("https://youtu.be/...")
    """

    def __init__(self, stages: list[Stage], settings: Settings) -> None:
        self._stages = stages
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, url: str) -> Path:
        """Execute the full pipeline for the given YouTube URL.

        Creates a fresh PipelineContext, runs every stage in sequence, logs
        a final summary, and returns the path to the clip-detection JSON file.

        Args:
            url: A validated YouTube URL.

        Returns:
            Path to ``storage/jobs/{video_id}.json``.

        Raises:
            ValueError: If the URL is invalid or a required context field is
                missing when a stage runs.
            Exception: Propagates any unhandled exception from any stage after
                logging it.
        """
        video_id = extract_video_id(url)
        ctx = PipelineContext(url=url, video_id=video_id)
        total = len(self._stages)
        pipeline_start = time.perf_counter()

        sep = "=" * 60
        logger.info(sep)
        logger.info("PIPELINE START — video_id: %s  |  %d stage(s)", video_id, total)
        logger.info(sep)

        for step, stage in enumerate(self._stages, start=1):
            self._execute_stage(stage, ctx, step, total)

        elapsed = time.perf_counter() - pipeline_start
        result_path = self._settings.jobs_dir / f"{video_id}.json"
        self._log_summary(ctx, elapsed, result_path)

        return result_path

    # ── Private ───────────────────────────────────────────────────────────────

    def _execute_stage(
        self,
        stage: Stage,
        ctx: PipelineContext,
        step: int,
        total: int,
    ) -> None:
        """Run one stage, logging its progress and timing.

        Args:
            stage: The stage to execute.
            ctx: Shared pipeline context (mutated in place).
            step: 1-based step index for the progress display.
            total: Total number of stages (for the ``N/M`` counter).

        Raises:
            Exception: Re-raises any exception from ``stage.run()`` after
                logging the failure details.
        """
        logger.info("── Stage %d/%d: %s", step, total, stage.name)
        start = time.perf_counter()

        try:
            stage.run(ctx)
            elapsed = time.perf_counter() - start
            logger.info("   ✓ %s completed in %.1fs", stage.name, elapsed)

        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "   ✗ %s FAILED after %.1fs: %s",
                stage.name,
                elapsed,
                exc,
                exc_info=True,
            )
            raise

    def _log_summary(self, ctx: PipelineContext, elapsed: float, result_path: Path) -> None:
        """Log the final pipeline summary after all stages complete."""
        sep = "=" * 60
        logger.info(sep)
        logger.info("PIPELINE COMPLETE in %.1fs", elapsed)

        if ctx.clip_result:
            logger.info("%s", ctx.clip_result.summary())

        if ctx.cut_paths:
            logger.info(
                "%d cut(s) saved to: %s",
                len(ctx.cut_paths),
                self._settings.cuts_dir / ctx.video_id,
            )

        if ctx.rendered_paths:
            logger.info(
                "%d rendered file(s) saved to: %s",
                len(ctx.rendered_paths),
                self._settings.rendered_dir / ctx.video_id,
            )

        logger.info("Results saved to: %s", result_path)
        logger.info(sep)


# ── Composition root ──────────────────────────────────────────────────────────


def create_default_pipeline(settings: Settings) -> PipelineOrchestrator:
    """Build the default 8-stage podcast pipeline.

    This is the **only** place in the codebase that imports concrete Stage
    and Service implementations.  Changing the pipeline structure means
    changing only this function.

    Stage order:
        1. AudioDownloadStage     — download audio via yt-dlp
        2. TranscriptionStage     — transcribe with faster-whisper
        3. ContentAnalysisStage   — detect viral clips via Ollama LLM
        4. VideoDownloadStage     — download full video via yt-dlp
        5. VideoCuttingStage      — cut each clip as standalone MP4
        6. SubtitleGenerationStage — generate SRT + ASS subtitle files
        7. SmartCropStage         — analyse face positions for 9:16 framing
        8. RenderingStage         — render final vertical 9:16 MP4s

    Args:
        settings: Application-wide configuration.

    Returns:
        A fully wired PipelineOrchestrator ready to call ``.run(url)``.
    """
    # Import concrete stages here (and ONLY here) to enforce the rule that
    # no other module in app/ needs to know about all 8 stages at once.
    from app.stages.audio_download import AudioDownloadStage
    from app.stages.content_analysis import ContentAnalysisStage
    from app.stages.rendering import RenderingStage
    from app.stages.smart_crop import SmartCropStage
    from app.stages.subtitle_generation import SubtitleGenerationStage
    from app.stages.transcription import TranscriptionStage
    from app.stages.video_cutting import VideoCuttingStage
    from app.stages.video_download import VideoDownloadStage

    stages: list[Stage] = [
        AudioDownloadStage(settings),
        TranscriptionStage(settings),
        ContentAnalysisStage(settings),
        VideoDownloadStage(settings),
        VideoCuttingStage(settings),
        SubtitleGenerationStage(settings),
        SmartCropStage(settings),
        RenderingStage(settings),
    ]

    return PipelineOrchestrator(stages=stages, settings=settings)
