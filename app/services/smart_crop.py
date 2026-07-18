"""Smart crop analysis service — face-tracking pre-processing for Step 7.

Analyses a video clip with MediaPipe BlazeFace to find the optimal
horizontal crop position at each moment in time.  The output is a
CropTimeline that VideoRendererService consumes to build an FFmpeg
piecewise-linear crop expression, producing smooth, face-centred
vertical video without any jumpy cuts.

Analysis pipeline (per clip)
-----------------------------
1. Open the clip with OpenCV and sample frames at ANALYSIS_FPS (2 fps).
2. Detect faces in each sampled frame with MediaPipe BlazeFace.
3. Compute the ideal crop centre_x from the detected face(s).
4. Apply an Exponential Weighted Moving Average (EWMA) to suppress jitter.
5. Detect scene cuts via mean-absolute-difference between consecutive
   sampled frames and reset the EWMA position immediately on a cut.
6. Decimate the smoothed positions to a compact keyframe list
   (one keyframe per significant position change).
7. Return a CropTimeline with face_detected=True, or a single-centred
   fallback if no faces were found or the libraries are unavailable.

Technology choice — MediaPipe BlazeFace
----------------------------------------
MediaPipe's BlazeFace model is bundled inside the ``mediapipe`` package
(no separate download), runs on CPU in <15 ms per frame at 720p, and is
specifically trained on frontal to near-frontal faces — exactly what
podcasts and interviews contain.  The model handles two-person frames
well and returns confidence scores for filtering noisy detections.

YOLOv8n was considered but rejected: its general-object model is 6 MB
download + heavier CPU cost for a task where BlazeFace is more accurate
for the specific face-only use case.  OpenCV YuNet requires a manual
model download and has lower recall on side-lit faces typical in studios.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from app.config.settings import Settings
from app.schemas.crop import CropKeyframe, CropTimeline
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ── Optional heavy imports — no hard failure if missing ──────────────────────

try:
    import cv2  # comes with mediapipe as a transitive dependency

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress legacy-API deprecation
        import mediapipe as mp  # type: ignore[import-untyped]

    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    mp = None  # type: ignore[assignment]

# ── Tuning constants ──────────────────────────────────────────────────────────

# Frames per second to decode and analyse.  2 fps gives one sample every
# 500 ms — more than enough to track the slow camera-work of podcasts.
_ANALYSIS_FPS: float = 2.0

# EWMA decay factor.  alpha=0.20 → half-life ≈ 3 analysed frames (1.5 s).
# Smaller values → smoother, slower tracking.
# Larger values → faster response, more jitter.
_EWMA_ALPHA: float = 0.20

# Fraction of the frame that must differ (greyscale MAD / 255) between two
# consecutive sampled frames to be classified as a scene cut.  On a cut the
# EWMA is hard-reset to the new face position rather than blending slowly.
_SCENE_CUT_RATIO: float = 0.18

# MediaPipe confidence threshold.  0.65 reduces false positives on hands,
# logos and other face-like regions common in studio setups.
_MIN_CONFIDENCE: float = 0.65

# Minimum pixel change between two consecutive keyframes.  Below this
# threshold the position is considered stable and no keyframe is emitted.
_MIN_KEYFRAME_DELTA: int = 6

# Maximum seconds allowed between keyframes even when the position is stable.
# Ensures the FFmpeg expression always has a recent anchor at the right edge
# of each segment.
_MAX_KEYFRAME_INTERVAL: float = 2.0


class SmartCropService:
    """Analyses a video clip and returns a face-tracking CropTimeline.

    This service is intentionally read-only: it never writes to disk.  Its
    sole responsibility is producing a CropTimeline that other services can
    consume.  Keeping analysis and rendering separated means the timeline
    can be inspected, serialised, or replayed without re-running detection.

    Args:
        settings: Application settings (reserved for future configuration
            of analysis parameters via environment variables).

    Example:
        >>> service = SmartCropService(settings)
        >>> timeline = service.analyze(cut_path, source_width=1280, source_height=720)
        >>> print(f"face_detected={timeline.face_detected}, keyframes={len(timeline.keyframes)}")
        face_detected=True, keyframes=12
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, video_path: Path) -> CropTimeline:
        """Analyse a clip and return a smooth face-tracking crop timeline.

        Probes the source video dimensions automatically using OpenCV (when
        available) or ffprobe as a fallback, so callers never need to supply
        width/height.

        Falls back to a single centred keyframe when MediaPipe/OpenCV is
        unavailable or no face is detected in the clip.

        Args:
            video_path: Path to the cut MP4 from Step 5.

        Returns:
            CropTimeline with ``face_detected=True`` on success, or a
            centred fallback with ``face_detected=False``.
        """
        source_width, source_height = self._probe_dimensions(video_path)
        crop_w = _crop_width_for(source_height)

        if not _CV2_AVAILABLE or not _MP_AVAILABLE:
            missing = "OpenCV" if not _CV2_AVAILABLE else "MediaPipe"
            logger.warning(
                "%s not available — using centre-crop fallback for '%s'. "
                "Install mediapipe to enable smart crop: uv add mediapipe",
                missing,
                video_path.name,
            )
            return _centred_timeline(source_width, source_height, crop_w)

        try:
            return self._run_analysis(video_path, source_width, source_height, crop_w)
        except Exception as exc:
            logger.warning(
                "Smart crop analysis failed for '%s' (%s) — falling back to centre crop.",
                video_path.name,
                exc,
            )
            return _centred_timeline(source_width, source_height, crop_w)

    # ── Private ───────────────────────────────────────────────────────────────

    def _probe_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Return ``(width, height)`` of the video, using the best available method.

        Primary path: OpenCV ``VideoCapture`` (zero extra cost since we already
        open the file for analysis).  Fallback: subprocess ffprobe for when
        OpenCV is not installed.  Last resort: a hard-coded 1280x720 safe
        default that at least produces a valid (if approximate) timeline.

        Args:
            video_path: Path to any video file.

        Returns:
            Integer ``(width, height)`` tuple.
        """
        if _CV2_AVAILABLE:
            cap = cv2.VideoCapture(str(video_path))
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w > 0 and h > 0:
                    return w, h
            finally:
                cap.release()

        # OpenCV unavailable or returned zeros — try ffprobe
        return _ffprobe_dimensions(video_path)

    def _run_analysis(
        self,
        video_path: Path,
        source_width: int,
        source_height: int,
        crop_w: int,
    ) -> CropTimeline:
        """Frame-sampling loop: decode → detect → record raw positions."""
        cap: Any = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video: {video_path}")

        try:
            native_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
            # Decode every Nth frame to achieve the target analysis frame rate
            frame_interval = max(1, round(native_fps / _ANALYSIS_FPS))

            # (timestamp_seconds, center_x_or_None, is_scene_cut)
            raw: list[tuple[float, int | None, bool]] = []
            prev_gray: Any = None
            frame_idx = 0

            with mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # full-range model: better for 2 people / wider shots
                min_detection_confidence=_MIN_CONFIDENCE,
            ) as detector:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_interval != 0:
                        frame_idx += 1
                        continue

                    timestamp = frame_idx / native_fps
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Scene-cut detection: mean absolute pixel difference
                    is_cut = False
                    if prev_gray is not None:
                        diff = cv2.absdiff(prev_gray, gray)
                        if diff.mean() / 255.0 > _SCENE_CUT_RATIO:
                            is_cut = True
                    prev_gray = gray

                    cx = self._detect_center_x(detector, frame, source_width, crop_w)
                    raw.append((timestamp, cx, is_cut))
                    frame_idx += 1

        finally:
            cap.release()

        if not raw:
            return _centred_timeline(source_width, source_height, crop_w)

        return self._build_timeline(raw, source_width, source_height, crop_w)

    def _detect_center_x(
        self,
        detector: Any,
        frame: Any,
        source_width: int,
        crop_w: int,
    ) -> int | None:
        """Return the ideal crop centre_x for one frame, or None if no face.

        Face-count strategies:

        * 0 faces — return ``None`` (caller will hold last known position).
        * 1 face — centre on the face.
        * 2 faces — if they both fit within the crop window, split the
          difference; otherwise follow the face with the larger bounding box
          (typically the one closer to camera / currently speaking).
        * 3+ faces — area-weighted centroid of all detections.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if not results.detections:
            return None

        detections = results.detections

        if len(detections) == 1:
            bbox = detections[0].location_data.relative_bounding_box
            return int((bbox.xmin + bbox.width / 2.0) * source_width)

        # Gather (relative_cx, relative_area) for each detection
        faces: list[tuple[float, float]] = []
        for det in detections:
            b = det.location_data.relative_bounding_box
            faces.append((b.xmin + b.width / 2.0, b.width * b.height))

        if len(faces) == 2:
            cx0, cx1 = faces[0][0], faces[1][0]
            spread_px = abs(cx1 - cx0) * source_width
            # Both faces fit inside the crop window — centre between them
            if spread_px < crop_w * 0.90:
                return int(((cx0 + cx1) / 2.0) * source_width)
            # Faces too far apart — follow the more prominent (larger bbox) one
            if faces[0][1] >= faces[1][1]:
                return int(faces[0][0] * source_width)
            return int(faces[1][0] * source_width)

        # 3+ faces: area-weighted centroid
        total_area = sum(a for _, a in faces)
        if total_area < 1e-9:
            return int(sum(cx for cx, _ in faces) / len(faces) * source_width)
        return int(sum(cx * a for cx, a in faces) / total_area * source_width)

    def _build_timeline(
        self,
        raw: list[tuple[float, int | None, bool]],
        source_width: int,
        source_height: int,
        crop_w: int,
    ) -> CropTimeline:
        """Apply EWMA smoothing and decimate to a compact keyframe list.

        Pass 1 — Smooth:
            Fill None gaps by holding the current EWMA value.
            Reset EWMA immediately on detected scene cuts.

        Pass 2 — Decimate:
            Emit a keyframe only when the smoothed position changes by at
            least MIN_KEYFRAME_DELTA pixels, or when more than
            MAX_KEYFRAME_INTERVAL seconds have elapsed since the last keyframe.
        """
        default_cx = source_width // 2
        ewma_cx = float(default_cx)
        face_ever_detected = False
        smoothed: list[tuple[float, int]] = []

        # ── Pass 1: EWMA smoothing ────────────────────────────────────────────
        for t, cx, is_cut in raw:
            if cx is not None:
                face_ever_detected = True
                target = float(cx)
            else:
                target = ewma_cx  # hold position when face temporarily lost

            if is_cut:
                # Hard jump on scene cut — no blending
                ewma_cx = target
            else:
                ewma_cx = _EWMA_ALPHA * target + (1.0 - _EWMA_ALPHA) * ewma_cx

            smoothed.append((t, round(ewma_cx)))

        if not face_ever_detected:
            logger.info("No faces detected in entire clip — using centre-crop fallback.")
            return _centred_timeline(source_width, source_height, crop_w)

        # ── Pass 2: Keyframe decimation ───────────────────────────────────────
        keyframes: list[CropKeyframe] = [CropKeyframe(time=smoothed[0][0], center_x=smoothed[0][1])]
        last_kf_cx = smoothed[0][1]
        last_kf_time = smoothed[0][0]

        for t, cx in smoothed[1:]:
            delta = abs(cx - last_kf_cx)
            gap = t - last_kf_time

            if delta >= _MIN_KEYFRAME_DELTA or gap >= _MAX_KEYFRAME_INTERVAL:
                keyframes.append(CropKeyframe(time=t, center_x=cx))
                last_kf_cx = cx
                last_kf_time = t

        # Always include the last position to anchor the expression's right end
        last_t, last_cx = smoothed[-1]
        if keyframes[-1].time < last_t:
            keyframes.append(CropKeyframe(time=last_t, center_x=last_cx))

        logger.info(
            "Smart crop analysis complete — %d keyframe(s) from %d sampled frame(s) "
            "(face_detected=True, alpha=%.2f).",
            len(keyframes),
            len(smoothed),
            _EWMA_ALPHA,
        )

        return CropTimeline(
            keyframes=keyframes,
            source_width=source_width,
            source_height=source_height,
            crop_width=crop_w,
            face_detected=True,
        )


# ── Module-level helpers ──────────────────────────────────────────────────────


def _ffprobe_dimensions(video_path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` via ffprobe subprocess.

    Used when OpenCV is not available.  Falls back to 1280x720 if ffprobe
    is also unavailable, which still produces a valid (approximate) timeline.

    Args:
        video_path: Path to any video file.

    Returns:
        Integer ``(width, height)`` tuple.
    """
    import json
    import shutil
    import subprocess

    import static_ffmpeg

    static_ffmpeg.add_paths()
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        logger.debug("ffprobe not found — using 1280x720 dimension fallback.")
        return 1280, 720

    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-select_streams",
        "v:0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        return int(stream.get("width", 1280)), int(stream.get("height", 720))
    except Exception as exc:
        logger.debug("ffprobe dimension probe failed (%s) — using 1280x720 fallback.", exc)
        return 1280, 720


def _crop_width_for(source_height: int) -> int:
    """Return the crop width for a 9:16 vertical slice of *source_height* pixels.

    Result is floored to the nearest even number so FFmpeg's yuv420p
    encoder accepts it without raising ``height/width not divisible by 2``.

    Args:
        source_height: Source video height in pixels.

    Returns:
        Even integer crop width.

    Example:
        >>> _crop_width_for(720)
        404   # 720 * 9/16 = 405 → floored to 404
        >>> _crop_width_for(1080)
        606   # 1080 * 9/16 = 607.5 → floored to 606
    """
    w = int(source_height * 9 / 16)
    return w - (w % 2)


def _centred_timeline(
    source_width: int,
    source_height: int,
    crop_w: int,
) -> CropTimeline:
    """Return a single-keyframe timeline centred on the frame.

    This is the transparent fallback used when MediaPipe/OpenCV is
    unavailable or when no face is detected — it reproduces exactly the
    same behaviour as the previous fixed centre-crop implementation.

    Args:
        source_width: Source video width in pixels.
        source_height: Source video height in pixels.
        crop_w: Pre-computed crop width from :func:`_crop_width_for`.

    Returns:
        CropTimeline with one keyframe at t=0 and face_detected=False.
    """
    return CropTimeline(
        keyframes=[CropKeyframe(time=0.0, center_x=source_width // 2)],
        source_width=source_width,
        source_height=source_height,
        crop_width=crop_w,
        face_detected=False,
    )
