"""Stage 4 — Video Download.

Reads:   ctx.url, ctx.video_id
Writes:  ctx.video_path
Service: VideoDownloader

The video is downloaded only after clip detection (Stage 3) confirms there
are viral moments worth rendering — avoiding bandwidth waste when no clips
are found.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.video_downloader import VideoDownloader


class VideoDownloadStage(Stage):
    """Downloads the full video at the configured quality (default 720p)."""

    def __init__(self, settings: Settings) -> None:
        self._service = VideoDownloader(settings)

    @property
    def name(self) -> str:
        return "Video Download"

    def run(self, ctx: PipelineContext) -> None:
        ctx.video_path = self._service.download(ctx.url, ctx.video_id)
