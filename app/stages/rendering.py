"""Stage 8 — Video Rendering.

Reads:   ctx.clip_result (via require_clip_result),
         ctx.cut_paths,
         ctx.subtitle_pairs,
         ctx.crop_timelines  (pre-computed by SmartCropStage)
Writes:  ctx.rendered_paths
Service: VideoRendererService
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.video_renderer import VideoRendererService


class RenderingStage(Stage):
    """Renders the final 9:16 vertical MP4 files with burnt-in subtitles."""

    def __init__(self, settings: Settings) -> None:
        self._service = VideoRendererService(settings)

    @property
    def name(self) -> str:
        return "Video Rendering"

    def run(self, ctx: PipelineContext) -> None:
        clip_result = ctx.require_clip_result()
        ctx.rendered_paths = self._service.render_all(
            result=clip_result,
            cut_paths=ctx.cut_paths,
            subtitle_pairs=ctx.subtitle_pairs,
            crop_timelines=ctx.crop_timelines if ctx.crop_timelines else None,
        )
