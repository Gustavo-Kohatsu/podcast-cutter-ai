"""Shared pipeline context — the contract between stages.

PipelineContext is the single data object that flows through the entire
pipeline.  Every stage reads its required inputs from the context and writes
its outputs back into it.

Design rationale
----------------
Mutable dataclass (vs. immutable / return-new pattern)
    A sequential pipeline in Python benefits from in-place mutation: it
    removes boilerplate (no unpacking returned values), keeps each stage
    method signature identical (``run(ctx) -> None``), and is idiomatic
    (similar to Flask ``g``, Django request objects, etc.).

    The trade-off — a stage could accidentally read a field set by a later
    stage — is mitigated by having each stage document which fields it reads
    and which it writes, and by the orchestrator's strict sequential order.

Optional vs. required fields
    Fields that are only populated during execution are typed as
    ``T | None`` with a default of ``None``.  Collection fields default to
    empty lists.  This makes it safe to create a context from just a URL
    and let the stages fill in the rest.

Adding new stages
    Simply add the new output field here and have the new stage set it.
    No changes to the orchestrator or to any existing stage are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.schemas.clip import ClipDetectionResult
from app.schemas.crop import CropTimeline
from app.schemas.transcript import Transcript


@dataclass
class PipelineContext:
    """Shared mutable state passed between pipeline stages.

    Fields are populated sequentially as stages complete.  Stages that depend
    on an earlier stage's output should raise ``ValueError`` if their required
    field is ``None``, providing a clear error instead of an ``AttributeError``
    deep in business logic.

    Attributes:
        url: The original YouTube URL provided by the user.
        video_id: YouTube video ID extracted from the URL.
        audio_path: Path to the downloaded audio file (set by AudioDownloadStage).
        transcript: Full transcription with timed segments (set by TranscriptionStage).
        clip_result: LLM-detected viral clip candidates (set by ContentAnalysisStage).
        video_path: Path to the downloaded full video (set by VideoDownloadStage).
        cut_paths: Paths to the raw cut MP4 files, one per clip (VideoCuttingStage).
        subtitle_pairs: ``(srt_path, ass_path)`` per clip (SubtitleGenerationStage).
        crop_timelines: Face-tracking crop positions per clip (SmartCropStage).
        rendered_paths: Paths to the final 9:16 MP4 files (RenderingStage).
    """

    # ── Seed fields (set at context creation time) ─────────────────────────────
    url: str
    video_id: str

    # ── Stage outputs (set progressively during execution) ────────────────────
    audio_path: Path | None = None
    transcript: Transcript | None = None
    clip_result: ClipDetectionResult | None = None
    video_path: Path | None = None
    cut_paths: list[Path] = field(default_factory=list)
    subtitle_pairs: list[tuple[Path, Path]] = field(default_factory=list)
    crop_timelines: list[CropTimeline] = field(default_factory=list)
    rendered_paths: list[Path] = field(default_factory=list)

    # ── Guard helpers ──────────────────────────────────────────────────────────

    def require_audio_path(self) -> Path:
        """Return ``audio_path`` or raise if not set."""
        if self.audio_path is None:
            raise ValueError("audio_path is not set. Did AudioDownloadStage run?")
        return self.audio_path

    def require_transcript(self) -> Transcript:
        """Return ``transcript`` or raise if not set."""
        if self.transcript is None:
            raise ValueError("transcript is not set. Did TranscriptionStage run?")
        return self.transcript

    def require_clip_result(self) -> ClipDetectionResult:
        """Return ``clip_result`` or raise if not set."""
        if self.clip_result is None:
            raise ValueError("clip_result is not set. Did ContentAnalysisStage run?")
        return self.clip_result

    def require_video_path(self) -> Path:
        """Return ``video_path`` or raise if not set."""
        if self.video_path is None:
            raise ValueError("video_path is not set. Did VideoDownloadStage run?")
        return self.video_path
