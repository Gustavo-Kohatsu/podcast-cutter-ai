"""Stage 7 — Smart Crop Analysis.

Reads:   ctx.cut_paths
Writes:  ctx.crop_timelines   [CropTimeline, ...]
Service: SmartCropService

Why is this a separate stage?
------------------------------
In the previous architecture SmartCropService was embedded inside
VideoRendererService, which violated the Single Responsibility Principle: the
renderer both analysed faces AND encoded video.  Extracting analysis as its
own stage means:

- SmartCropService can be tested independently without triggering FFmpeg.
- The CropTimeline is visible in the context, making it inspectable and
  serialisable for debugging or future use (e.g., passing to an analytics step).
- The renderer becomes a pure encoding stage — it only encodes, never analyses.
- Swapping the cropping algorithm (e.g., replacing MediaPipe with a YOLO-based
  approach) requires touching only this stage and its service.

Fallback behaviour
------------------
SmartCropService already handles the case where MediaPipe is unavailable or no
face is detected: it returns a single centred keyframe (transparent fallback).
This stage simply propagates that behaviour into the context, so RenderingStage
always receives a valid list of CropTimeline objects — one per cut clip.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.pipeline.context import PipelineContext
from app.pipeline.stage import Stage
from app.services.smart_crop import SmartCropService


class SmartCropStage(Stage):
    """Analyses each cut clip for face positions and builds a crop timeline."""

    def __init__(self, settings: Settings) -> None:
        self._service = SmartCropService(settings)

    @property
    def name(self) -> str:
        return "Smart Crop Analysis"

    def run(self, ctx: PipelineContext) -> None:
        timelines = []
        for cut_path in ctx.cut_paths:
            timeline = self._service.analyze(cut_path)
            timelines.append(timeline)
        ctx.crop_timelines = timelines
