"""Intermediate data model for the smart crop timeline.

CropTimeline is the contract between SmartCropService (analysis) and
VideoRendererService (rendering).  It holds a compact sequence of
(time, center_x) keyframes produced by face detection and temporal
smoothing, and can materialise them as an FFmpeg filter expression for
smooth, lossless piecewise-linear panning during video encoding.

Design notes
------------
- Uses plain dataclasses (not Pydantic) because this structure is
  never persisted to disk or sent over the network in the current design.
- center_x is expressed in *source video pixel coordinates* so it is
  independent of the target output resolution.
- The FFmpeg crop filter x parameter is the *left edge* of the crop
  window, not the centre.  All conversion happens inside this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CropKeyframe:
    """A single anchor point in the crop timeline.

    Attributes:
        time: Seconds from the first frame of the clip.
        center_x: Desired horizontal centre of the crop window expressed
            in source-video pixel coordinates.
    """

    time: float
    center_x: int


@dataclass
class CropTimeline:
    """Temporal sequence of face-tracking crop positions for one clip.

    Produced by SmartCropService and consumed by VideoRendererService.

    Attributes:
        keyframes: Ordered list of (time, center_x) anchors.  Must contain
            at least one entry.  Times are strictly non-decreasing.
        source_width: Width of the source video in pixels.
        source_height: Height of the source video in pixels.
        crop_width: Width of the crop window in source pixels.
            Equals ``source_height * 9 // 16`` rounded down to an even
            number so FFmpeg's yuv420p encoder accepts it without complaint.
        face_detected: True when at least one face was found during
            analysis.  False signals a centre-crop fallback is active.
    """

    keyframes: list[CropKeyframe]
    source_width: int
    source_height: int
    crop_width: int
    face_detected: bool = False

    # ── Interpolation helpers ─────────────────────────────────────────────────

    def center_x_at(self, t: float) -> int:
        """Return the linearly interpolated crop centre-x at time *t*.

        Clamps to the first or last keyframe value outside the defined range.

        Args:
            t: Timestamp in seconds.

        Returns:
            Pixel x coordinate in source-video space.
        """
        if not self.keyframes:
            return self.source_width // 2

        if t <= self.keyframes[0].time:
            return self.keyframes[0].center_x

        if t >= self.keyframes[-1].time:
            return self.keyframes[-1].center_x

        for i in range(len(self.keyframes) - 1):
            k0, k1 = self.keyframes[i], self.keyframes[i + 1]
            if k0.time <= t <= k1.time:
                dt = k1.time - k0.time
                if dt < 1e-6:
                    return k0.center_x
                alpha = (t - k0.time) / dt
                return round(k0.center_x + alpha * (k1.center_x - k0.center_x))

        return self.keyframes[-1].center_x

    def left_x_at(self, t: float) -> int:
        """Return the clamped left-edge x of the crop window at time *t*.

        The value is clamped to ``[0, source_width - crop_width]`` so the
        window never goes outside the source frame.

        Args:
            t: Timestamp in seconds.

        Returns:
            Left-edge pixel x coordinate, ready for FFmpeg's crop filter.
        """
        half = self.crop_width // 2
        cx = self.center_x_at(t)
        return max(0, min(self.source_width - self.crop_width, cx - half))

    # ── FFmpeg expression builder ─────────────────────────────────────────────

    def to_ffmpeg_crop_x_expr(self) -> str:
        """Build an FFmpeg filter expression for the time-varying crop x.

        Generates a piecewise-linear expression using nested ``if(lt(t,T),…)``
        calls.  FFmpeg evaluates this at every frame using the built-in ``t``
        variable (current timestamp in seconds), producing smooth sub-pixel
        interpolation between keyframe positions.

        For a typical podcast clip with 8 keyframes the expression is
        ~300 characters — well within FFmpeg's command-line limits.

        Returns:
            String ready to use as the ``x`` parameter of FFmpeg's ``crop``
            filter, e.g.::

                "if(lt(t,3.500),438.0+(160.0)*(t-0.000)/3.500,598.0)"

        Example filter usage::

            crop=404:720:if(lt(t,3.500),438.0+(160.0)*(t-0.000)/3.500,598.0):0
        """
        half = self.crop_width // 2

        def clamped_left(cx: int) -> float:
            clamped_cx = max(half, min(self.source_width - half, cx))
            return float(clamped_cx - half)

        if not self.keyframes:
            return f"{clamped_left(self.source_width // 2):.1f}"

        left_values = [clamped_left(kf.center_x) for kf in self.keyframes]

        if len(left_values) == 1:
            return f"{left_values[0]:.1f}"

        # Build right-to-left: start with the terminal value, wrap each
        # preceding segment in an if(lt(t, boundary), linear_interp, rest)
        expr = f"{left_values[-1]:.1f}"

        for i in range(len(self.keyframes) - 2, -1, -1):
            k0, k1 = self.keyframes[i], self.keyframes[i + 1]
            x0, x1 = left_values[i], left_values[i + 1]
            dt = k1.time - k0.time

            if dt < 1e-3 or x0 == x1:
                segment = f"{x0:.1f}"
            else:
                dx = x1 - x0
                segment = f"{x0:.1f}+({dx:.1f})*(t-{k0.time:.3f})/{dt:.3f}"

            expr = f"if(lt(t,{k1.time:.3f}),{segment},{expr})"

        return expr
