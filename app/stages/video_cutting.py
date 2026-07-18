"""Stage 5 — Video Cutting.

Reads:   ctx.video_path (via require_video_path), ctx.clip_result (via require_clip_result), ctx.video_id
Writes:  ctx.cut_paths
Service: VideoCutter
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.video_cutter import VideoCutter


class VideoCuttingStage(Stage):
    """Extracts each viral clip as a standalone MP4 via stream-copy (no re-encode)."""

    def __init__(self, settings: Settings) -> None:
        self._service = VideoCutter(settings)

    @property
    def name(self) -> str:
        return "Video Cutting"

    def run(self, ctx: PipelineContext) -> None:
        video_path = ctx.require_video_path()
        clip_result = ctx.require_clip_result()
        ctx.cut_paths = self._service.cut(video_path, clip_result, ctx.video_id)
