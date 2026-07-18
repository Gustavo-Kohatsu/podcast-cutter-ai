"""Video renderer service — Step 7 of the pipeline.

Produces the final publication-ready MP4 for each clip:
  - Converts horizontal (16:9) video to vertical (9:16) via centre crop.
  - Scales to 1080x1920 (the standard resolution for Shorts/Reels/TikTok).
  - Burns the ASS subtitle file directly into the video stream.
  - Re-encodes with H.264 + AAC for maximum platform compatibility.

Technical decisions
-------------------
Crop strategy — centre crop
    For sports highlights, news, and podcast recordings the subject of interest
    is nearly always centred horizontally. A vertical slice through the middle
    of a 16:9 frame captures the action without complex subject-tracking.
    The crop is computed in Python from the probed dimensions rather than using
    FFmpeg expression syntax, so the filter string stays readable and debuggable.

Codec — H.264 (libx264) + AAC
    H.265 produces ~35% smaller files but is still rejected by some upload APIs
    and older mobile players. H.264 with CRF 23 and the *fast* preset gives
    excellent quality (~2-4 MB/min) that all target platforms accept without
    re-encoding.

Subtitle embedding — ASS filter
    The ``ass`` FFmpeg filter burns subtitles at the final output resolution.
    Because the subtitle files generated in Step 6 use PlayResX/Y = 1280x720,
    a vertical-adapted copy is created before rendering (PlayResX/Y = 1080x1920,
    font size 72, bottom margins adjusted). The temp copy is deleted afterwards.

Path escaping
    FFmpeg's filtergraph parser uses ``:`` as an option separator and ``\\`` as
    an escape character. On Windows the drive-letter colon (``C:``) must be
    written as ``C\\:`` inside filter option values. Backslashes in the rest
    of the path are converted to forward slashes, which FFmpeg handles on all
    platforms.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import static_ffmpeg

from app.config.settings import Settings
from app.schemas.clip import ClipDetectionResult
from app.schemas.crop import CropTimeline
from app.services.smart_crop import SmartCropService
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Target output dimensions — standard for Shorts, Reels and TikTok
_OUT_W = 1080
_OUT_H = 1920

# ASS style overrides for the 1080x1920 coordinate space.
# Font size 72 / 1920 = 3.75 % of height — roughly equivalent to the
# original 52 / 720 = 7.2 % adjusted for a taller canvas and the smaller
# visual weight of subtitles on a phone screen held vertically.
_V_FONTSIZE = 72
_V_MARGIN_L = 60
_V_MARGIN_R = 60
_V_MARGIN_V = 150  # pixels from bottom in 1920px coordinate space


class VideoRendererService:
    """Renders each viral clip into a vertical 9:16 MP4 with burnt-in subtitles.

    Args:
        settings: Application settings (output paths, encode parameters).

    Example:
        >>> renderer = VideoRendererService(settings)
        >>> paths = renderer.render_all(result, cut_paths, subtitle_pairs)
        >>> print(paths[0])   # storage/rendered/VIDEO_ID/cut_001_final.mp4
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        static_ffmpeg.add_paths()
        self._ffmpeg_bin = shutil.which("ffmpeg")
        self._ffprobe_bin = shutil.which("ffprobe")
        if not self._ffmpeg_bin or not self._ffprobe_bin:
            raise RuntimeError(
                "ffmpeg/ffprobe not found after static_ffmpeg.add_paths(). "
                "Try reinstalling static-ffmpeg."
            )
        self._smart_crop = SmartCropService(settings)

    # ── Public API ────────────────────────────────────────────────────────────

    def render_all(
        self,
        result: ClipDetectionResult,
        cut_paths: list[Path],
        subtitle_pairs: list[tuple[Path, Path]],
        crop_timelines: list[CropTimeline] | None = None,
    ) -> list[Path]:
        """Render all clips to vertical MP4 with embedded subtitles.

        Processes clips in the same order as ``result.sorted_by_score()`` so
        ``cut_001_final.mp4`` always corresponds to the highest-scored clip,
        matching the naming convention established in Step 5.

        Args:
            result: Clip detection result for metadata and video_id.
            cut_paths: Paths to the raw cut MP4s from Step 5.
            subtitle_pairs: ``(srt, ass)`` path pairs from Step 6.
            crop_timelines: Pre-computed CropTimeline objects from SmartCropStage.
                When provided, face analysis is skipped inside the renderer.
                When ``None``, the renderer computes timelines internally
                (backward-compatible fallback for direct service use).

        Returns:
            List of paths to the rendered final MP4 files, in rank order.
        """
        if len(cut_paths) != len(subtitle_pairs):
            logger.warning(
                "cut_paths (%d) and subtitle_pairs (%d) lengths differ — "
                "rendering only the aligned portion.",
                len(cut_paths),
                len(subtitle_pairs),
            )

        rendered: list[Path] = []
        pairs = list(zip(cut_paths, subtitle_pairs, strict=False))

        for i, (cut_path, (_srt, ass_path)) in enumerate(pairs, start=1):
            precomputed = crop_timelines[i - 1] if crop_timelines else None
            try:
                output = self._render_one(cut_path, ass_path, i, result.video_id, precomputed)
                rendered.append(output)
            except Exception as exc:
                logger.error(
                    "  [%d/%d] Render failed for %s: %s",
                    i,
                    len(pairs),
                    cut_path.name,
                    exc,
                )

        logger.info(
            "Rendering complete — %d/%d file(s) saved to: %s",
            len(rendered),
            len(pairs),
            self._settings.rendered_dir / result.video_id,
        )
        return rendered

    # ── Private ───────────────────────────────────────────────────────────────

    def _render_one(
        self,
        cut_path: Path,
        ass_path: Path,
        clip_index: int,
        video_id: str,
        crop_timeline: CropTimeline | None = None,
    ) -> Path:
        """Render a single clip.

        Args:
            cut_path: Raw cut MP4 from Step 5.
            ass_path: Subtitle file from Step 6.
            clip_index: 1-based clip number (for output filename and logs).
            video_id: Used for the output subdirectory.

        Returns:
            Path to the rendered MP4.
        """
        output_dir = self._settings.rendered_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"cut_{clip_index:03d}_final"
        output_path = output_dir / f"{stem}.mp4"

        if output_path.exists() and output_path.stat().st_size > 0:
            logger.info("  [%d] Render cache hit — skipping %s", clip_index, stem)
            return output_path

        if not cut_path.exists():
            raise FileNotFoundError(f"Cut file not found: {cut_path}")

        # Probe source dimensions
        width, height = self._probe_dimensions(cut_path)

        # ── Smart crop analysis ───────────────────────────────────────────────
        # Use a pre-computed timeline from SmartCropStage when available.
        # Fall back to on-the-fly analysis for direct service use (backwards compat).
        timeline = (
            crop_timeline if crop_timeline is not None else self._smart_crop.analyze(cut_path)
        )

        logger.info(
            "  [%d] %s | %dx%d → %dx%d | smart_crop=%s | keyframes=%d",
            clip_index,
            stem,
            width,
            height,
            _OUT_W,
            _OUT_H,
            timeline.face_detected,
            len(timeline.keyframes),
        )

        # ── Build FFmpeg filter chain ─────────────────────────────────────────
        input_ratio = width / height
        target_ratio = 9 / 16

        if abs(input_ratio - target_ratio) < 0.01:
            # Already 9:16 — no crop needed, just scale and burn subtitles
            crop_filter = f"scale={_OUT_W}:{_OUT_H}"
        else:
            # Dynamic piecewise-linear crop expression from the timeline
            crop_x_expr = timeline.to_ffmpeg_crop_x_expr()
            crop_filter = (
                f"crop={timeline.crop_width}:{height}:{crop_x_expr}:0,scale={_OUT_W}:{_OUT_H}"
            )

        # Create a temporary ASS file adapted for 9:16 playback, then clean up
        adapted_ass = self._adapt_ass_for_vertical(ass_path, output_dir, clip_index)
        try:
            ass_escaped = _escape_filter_path(adapted_ass)
            filter_chain = f"{crop_filter},ass={ass_escaped}"
            self._run_ffmpeg(cut_path, filter_chain, output_path)
        finally:
            adapted_ass.unlink(missing_ok=True)

        size_mb = output_path.stat().st_size / 1024**2
        logger.info("  [%d] %s done — %.1f MB", clip_index, stem, size_mb)
        return output_path

    def _probe_dimensions(self, video_path: Path) -> tuple[int, int]:
        """Return ``(width, height)`` of the first video stream via ffprobe.

        Args:
            video_path: Path to any video file.

        Returns:
            Integer ``(width, height)`` tuple.

        Raises:
            RuntimeError: If ffprobe fails or the stream has no dimension info.
        """
        cmd = [
            self._ffprobe_bin,
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
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ffprobe failed for {video_path.name}: {exc.stderr}") from exc

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video streams found in {video_path.name}")

        stream = streams[0]
        return int(stream["width"]), int(stream["height"])

    def _adapt_ass_for_vertical(self, ass_path: Path, output_dir: Path, clip_index: int) -> Path:
        """Create a temporary ASS file tuned for 1080x1920 vertical rendering.

        The original ASS from Step 6 targets a 1280x720 coordinate space.
        This method rewrites the ``PlayResX/Y`` header values and the Default
        style's font size and margins so that subtitles render at the right
        visual scale and position on the 1080x1920 output.

        The temp file is written to the output directory (avoids OS temp-dir
        paths that can contain special chars or spaces) and is deleted by the
        caller after FFmpeg finishes.

        Args:
            ass_path: Path to the original ASS subtitle file.
            output_dir: Directory to write the temporary file into.
            clip_index: Used to give the temp file a unique name.

        Returns:
            Path to the temporary adapted ASS file.
        """
        content = ass_path.read_text(encoding="utf-8")

        # Update virtual canvas dimensions
        content = re.sub(r"PlayResX:\s*\d+", f"PlayResX: {_OUT_W}", content)
        content = re.sub(r"PlayResY:\s*\d+", f"PlayResY: {_OUT_H}", content)

        # Patch the Default style line in-place.
        # ASS Style field order (0-indexed, comma-separated after "Style: "):
        #   0=Name  1=Fontname  2=Fontsize  ...  19=MarginL  20=MarginR  21=MarginV  22=Encoding
        def _patch_style(match: re.Match) -> str:
            fields = match.group(0).split(",")
            if len(fields) >= 23:
                fields[2] = str(_V_FONTSIZE)
                fields[19] = str(_V_MARGIN_L)
                fields[20] = str(_V_MARGIN_R)
                fields[21] = str(_V_MARGIN_V)
            return ",".join(fields)

        content = re.sub(r"^Style: Default,.*$", _patch_style, content, flags=re.MULTILINE)

        tmp_path = output_dir / f"_tmp_ass_{clip_index:03d}.ass"
        tmp_path.write_text(content, encoding="utf-8")
        return tmp_path

    def _run_ffmpeg(self, input_path: Path, filter_chain: str, output_path: Path) -> None:
        """Invoke FFmpeg to render one clip.

        Video pipeline:
            crop → scale → ass (subtitle burn-in)
        Audio pipeline:
            aac 128 k @ 44100 Hz (stereo)

        The ``-profile:v high -level:v 4.1`` flags ensure the output is
        playable on every modern device and accepted by all major upload APIs.
        ``-pix_fmt yuv420p`` disables 4:4:4 / 10-bit subsampling so the file
        plays on hardware decoders without issues.

        Args:
            input_path: Source cut MP4.
            filter_chain: Pre-built FFmpeg filtergraph string.
            output_path: Destination path for the rendered MP4.

        Raises:
            RuntimeError: Wraps CalledProcessError with the relevant stderr snippet.
        """
        cmd = [
            self._ffmpeg_bin,
            "-y",  # overwrite without prompt
            "-i",
            str(input_path),  # source cut MP4
            "-vf",
            filter_chain,  # crop + scale + subtitles
            "-c:v",
            "libx264",  # H.264 — maximum compatibility
            "-preset",
            self._settings.render_preset,
            "-crf",
            str(self._settings.render_crf),
            "-profile:v",
            "high",  # H.264 High Profile
            "-level:v",
            "4.1",  # compatible with all modern devices
            "-c:a",
            "aac",  # AAC audio
            "-b:a",
            "128k",
            "-ar",
            "44100",
            "-pix_fmt",
            "yuv420p",  # broad hardware decoder compat
            "-movflags",
            "+faststart",  # moov atom at front for streaming
            str(output_path),
        ]

        logger.debug("FFmpeg: %s", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            # Trim stderr to last 2000 chars to avoid flooding the log
            stderr_tail = (exc.stderr or "")[-2000:]
            logger.error("FFmpeg failed:\n%s", stderr_tail)
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"FFmpeg render failed for {output_path.name}. Last error: {stderr_tail[-300:]}"
            ) from exc


# ── Module-level helpers ──────────────────────────────────────────────────────


def _escape_filter_path(path: Path) -> str:
    """Escape a filesystem path for use as an FFmpeg filtergraph option value.

    FFmpeg's filtergraph parser treats ``:`` as an option separator and ``\\``
    as an escape character.  On Windows the drive-letter colon must be written
    as ``C\\:`` and all other backslashes converted to forward slashes.

    Args:
        path: The path to escape.

    Returns:
        A string safe for embedding directly after ``ass=`` in a filter chain.

    Examples:
        >>> _escape_filter_path(Path(r"C:\\Users\\foo\\sub.ass"))
        'C\\:/Users/foo/sub.ass'
        >>> _escape_filter_path(Path("/tmp/sub.ass"))
        '/tmp/sub.ass'
    """
    p = str(path).replace("\\", "/")
    # Escape the Windows drive-letter colon so FFmpeg doesn't treat it as
    # a filter option separator (e.g. "C:/" → "C\:/")
    p = re.sub(r"^([A-Za-z]):", r"\1\\:", p)
    return p
