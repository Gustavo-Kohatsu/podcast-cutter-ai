"""Video clip cutting service using FFmpeg.

Responsible for extracting individual viral clip segments from a full
downloaded video, based on the timestamps produced by the LLM detection step.

Each clip is extracted via stream-copy (no re-encoding), making cuts nearly
instantaneous regardless of clip position or video length.
"""

import math
import shutil
import subprocess
from pathlib import Path

import static_ffmpeg

from app.config.settings import Settings
from app.schemas.clip import ClipDetectionResult, ViralClip
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VideoCutter:
    """Cuts viral clip segments from a full video file using FFmpeg.

    Uses stream-copy (``-c:v copy -c:a copy``) so the operation is fast and
    lossless — no re-encoding occurs. The trade-off is that cut points are
    aligned to the nearest keyframe, which may cause clips to start up to
    ~2 seconds earlier than the requested timestamp. For podcast content,
    this level of imprecision is imperceptible.

    Clips are named sequentially by viral score rank:
    ``cut_001.mp4`` is the highest-scored clip, ``cut_002.mp4`` the second, etc.

    If a single clip fails (invalid timestamps, FFmpeg error, etc.), it is
    logged and skipped — remaining clips are still processed.

    Args:
        settings: Application settings containing the cuts storage path.

    Example:
        >>> cutter = VideoCutter(settings)
        >>> paths = cutter.cut(video_path, detection_result, "dQw4w9WgXcQ")
        >>> for p in paths:
        ...     print(p)
        storage/cuts/dQw4w9WgXcQ/cut_001.mp4
        storage/cuts/dQw4w9WgXcQ/cut_002.mp4

    Raises:
        RuntimeError: If the ffmpeg binary cannot be located after
            ``static_ffmpeg.add_paths()``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Inject ffmpeg/ffprobe binaries from the venv into the current process PATH.
        static_ffmpeg.add_paths()
        self._ffmpeg_bin = shutil.which("ffmpeg")
        if self._ffmpeg_bin is None:
            raise RuntimeError(
                "ffmpeg binary not found in PATH after static_ffmpeg.add_paths(). "
                "Try reinstalling static-ffmpeg."
            )

    def cut(
        self,
        video_path: Path,
        result: ClipDetectionResult,
        video_id: str,
    ) -> list[Path]:
        """Cut all detected clips from the source video.

        Processes clips in descending viral-score order so the numbering
        always reflects quality rank (cut_001 = best clip). Clips that
        already exist on disk are skipped without re-processing.

        Args:
            video_path: Path to the full downloaded video file.
            result: Clip detection result containing timestamps and scores.
            video_id: YouTube video ID used to build the output directory.

        Returns:
            Paths to all successfully created cut files, in rank order.
            The list may be shorter than ``result.total_clips_found`` when
            clips are skipped due to validation errors or FFmpeg failures.

        Raises:
            FileNotFoundError: If ``video_path`` does not exist on disk.
        """
        if not video_path.exists():
            raise FileNotFoundError(
                f"Source video not found: {video_path}\n"
                "Ensure Step 4 (Video Download) completed successfully."
            )

        if not result.clips:
            logger.warning("No clips in detection result — nothing to cut.")
            return []

        output_dir = self._settings.cuts_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        clips = result.sorted_by_score()
        total = len(clips)
        logger.info(
            "Cutting %d clip(s) from '%s' → %s",
            total,
            video_path.name,
            output_dir,
        )

        successful: list[Path] = []

        for idx, clip in enumerate(clips, start=1):
            label = f"cut_{idx:03d}"
            output_path = output_dir / f"{label}.mp4"

            if output_path.exists():
                logger.info(
                    "  [%d/%d] %s already exists, skipping.", idx, total, label
                )
                successful.append(output_path)
                continue

            if not self._is_valid(clip, idx, total):
                continue

            logger.info(
                "  [%d/%d] %s — '%s' (%.0fs → %.0fs, %.1fs)",
                idx,
                total,
                label,
                clip.title[:60],
                clip.start_time,
                clip.end_time,
                clip.duration,
            )

            try:
                self._run_ffmpeg(video_path, output_path, clip)
                size_mb = _file_size_mb(output_path)
                logger.info(
                    "  [%d/%d] %s — done (%.1f MB)", idx, total, label, size_mb
                )
                successful.append(output_path)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "  [%d/%d] %s — FFmpeg exited with code %d: %s",
                    idx,
                    total,
                    label,
                    exc.returncode,
                    (exc.stderr or "").strip() or "(no stderr captured)",
                )
                # Remove incomplete output file if it was partially written
                output_path.unlink(missing_ok=True)
            except Exception as exc:
                logger.error(
                    "  [%d/%d] %s — unexpected error: %s", idx, total, label, exc
                )
                output_path.unlink(missing_ok=True)

        logger.info(
            "Cutting complete — %d/%d clip(s) generated successfully.",
            len(successful),
            total,
        )
        return successful

    # ── Private helpers ────────────────────────────────────────────────────────

    def _is_valid(self, clip: ViralClip, idx: int, total: int) -> bool:
        """Return True if the clip's timestamps are safe to pass to FFmpeg.

        Pydantic already guarantees ``end_time > start_time >= 0``, so here
        we only guard against non-finite floats (NaN/Inf) that would cause
        FFmpeg to hang or produce garbage output.

        Args:
            clip: The clip to validate.
            idx: 1-based position in the current processing pass (for logs).
            total: Total number of clips in this pass (for logs).

        Returns:
            True if the clip should be processed; False if it should be skipped.
        """
        if not math.isfinite(clip.start_time) or not math.isfinite(clip.end_time):
            logger.warning(
                "  [%d/%d] Skipping '%s' — non-finite timestamp(s): "
                "start=%.2f end=%.2f",
                idx,
                total,
                clip.title,
                clip.start_time,
                clip.end_time,
            )
            return False

        if clip.duration <= 0:
            logger.warning(
                "  [%d/%d] Skipping '%s' — non-positive duration %.2fs.",
                idx,
                total,
                clip.title,
                clip.duration,
            )
            return False

        return True

    def _run_ffmpeg(
        self,
        input_path: Path,
        output_path: Path,
        clip: ViralClip,
    ) -> None:
        """Invoke FFmpeg to extract a single clip segment.

        Placing ``-ss`` before ``-i`` enables fast keyframe-level seeking so
        FFmpeg does not decode from the start of the file. ``-t`` specifies
        the clip duration (relative to the seek position). ``-avoid_negative_ts
        make_zero`` corrects presentation timestamps that can become negative
        after seeking, preventing playback issues.

        Args:
            input_path: Path to the full source video file.
            output_path: Destination path for the extracted clip.
            clip: Clip metadata providing start time and duration.

        Raises:
            subprocess.CalledProcessError: If FFmpeg exits with a non-zero
                return code. ``exc.stderr`` contains the FFmpeg error output.
        """
        cmd = [
            self._ffmpeg_bin,
            "-y",                               # overwrite output without prompt
            "-ss", str(clip.start_time),        # fast input seek (keyframe-aligned)
            "-i", str(input_path),              # source video
            "-t", str(clip.duration),           # duration relative to seek position
            "-c:v", "copy",                     # copy video stream — no re-encode
            "-c:a", "copy",                     # copy audio stream — no re-encode
            "-avoid_negative_ts", "make_zero",  # fix timestamps after seeking
            "-movflags", "+faststart",          # move moov atom to front (MP4 best practice)
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)


def _file_size_mb(path: Path) -> float:
    """Return the file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)
