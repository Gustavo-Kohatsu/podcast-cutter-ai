"""Stage 1 — Audio Download.

Reads:   ctx.url, ctx.video_id
Writes:  ctx.audio_path
Service: AudioDownloader
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.downloader import AudioDownloader


class AudioDownloadStage(Stage):
    """Downloads the audio stream from the YouTube URL."""

    def __init__(self, settings: Settings) -> None:
        self._service = AudioDownloader(settings)

    @property
    def name(self) -> str:
        return "Audio Download"

    def run(self, ctx: PipelineContext) -> None:
        ctx.audio_path = self._service.download(ctx.url, ctx.video_id)
