"""Stage 3 — Content Analysis (Viral Clip Detection).

Reads:   ctx.transcript (via require_transcript)
Writes:  ctx.clip_result
Service: ClipDetector
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.clip_detector import ClipDetector


class ContentAnalysisStage(Stage):
    """Analyses the transcript with a local LLM to detect viral clip candidates."""

    def __init__(self, settings: Settings) -> None:
        self._service = ClipDetector(settings)

    @property
    def name(self) -> str:
        return "Content Analysis"

    def run(self, ctx: PipelineContext) -> None:
        transcript = ctx.require_transcript()
        ctx.clip_result = self._service.detect(transcript)
