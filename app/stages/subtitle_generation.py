"""Stage 6 — Subtitle Generation.

Reads:   ctx.clip_result (via require_clip_result), ctx.transcript (via require_transcript)
Writes:  ctx.subtitle_pairs   [(srt_path, ass_path), ...]
Service: SubtitleService
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.subtitle_service import SubtitleService


class SubtitleGenerationStage(Stage):
    """Generates SRT and ASS subtitle files for each detected clip."""

    def __init__(self, settings: Settings) -> None:
        self._service = SubtitleService(settings)

    @property
    def name(self) -> str:
        return "Subtitle Generation"

    def run(self, ctx: PipelineContext) -> None:
        clip_result = ctx.require_clip_result()
        transcript = ctx.require_transcript()
        ctx.subtitle_pairs = self._service.generate_all(clip_result, transcript)
