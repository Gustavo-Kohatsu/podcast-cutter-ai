"""Stage 2 — Audio Transcription.

Reads:   ctx.audio_path (via require_audio_path), ctx.video_id, ctx.url
Writes:  ctx.transcript
Service: Transcriber
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.transcriber import Transcriber


class TranscriptionStage(Stage):
    """Transcribes the audio file to timed text segments using faster-whisper."""

    def __init__(self, settings: Settings) -> None:
        self._service = Transcriber(settings)

    @property
    def name(self) -> str:
        return "Transcription"

    def run(self, ctx: PipelineContext) -> None:
        audio_path = ctx.require_audio_path()
        ctx.transcript = self._service.transcribe(audio_path, ctx.video_id, ctx.url)
