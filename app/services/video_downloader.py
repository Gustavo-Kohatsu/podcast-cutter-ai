"""Video download service using yt-dlp.

Responsible for downloading the full video stream from a YouTube URL and
saving it as an MP4 file in the configured storage directory.

The audio-only download used during transcription is handled separately by
AudioDownloader, keeping the two concerns independent. This service is
intended to run after clip detection so the video is only fetched when there
are confirmed clips worth cutting.
"""

from pathlib import Path

import static_ffmpeg
import yt_dlp

from app.config.settings import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Video extensions that yt-dlp may produce, checked in preference order
_VIDEO_EXTENSIONS = ("mp4", "mkv", "webm")


class VideoDownloader:
    """Downloads video from YouTube URLs using yt-dlp.

    Downloads the combined video+audio stream at or below the configured
    quality (height in pixels). The output is always merged into an MP4
    container via FFmpeg. If the exact requested resolution is unavailable,
    yt-dlp falls back gracefully to the best option below the limit.

    Args:
        settings: Application settings containing storage path and video
            quality configuration.

    Example:
        >>> downloader = VideoDownloader(settings)
        >>> video_path = downloader.download("https://youtu.be/...", "dQw4w9WgXcQ")
        >>> print(video_path)
        storage/videos/dQw4w9WgXcQ/dQw4w9WgXcQ.mp4
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Inject ffmpeg/ffprobe binaries from the venv into the current process PATH.
        # This runs once per instance — no system installation required.
        static_ffmpeg.add_paths()

    def download(self, url: str, video_id: str) -> Path:
        """Download the video from the given YouTube URL.

        Creates a subdirectory under ``settings.video_dir`` named after the
        ``video_id`` and places the video file there. If a file already exists
        from a previous run, it is returned immediately without re-downloading.

        Args:
            url: The YouTube URL to download the video from.
            video_id: The YouTube video ID, used as directory and filename.

        Returns:
            Path to the downloaded video file.

        Raises:
            yt_dlp.utils.DownloadError: If yt-dlp fails to download the video.
            FileNotFoundError: If the video file cannot be located after
                download (indicates an FFmpeg or format issue).
        """
        output_dir = self._settings.video_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        cached = self._find_video_file(output_dir, video_id)
        if cached is not None:
            logger.info("Video already exists, skipping download: %s", cached)
            return cached

        quality = self._settings.video_quality
        logger.info("Downloading video for '%s' at up to %dp...", video_id, quality)
        logger.debug("Output directory: %s", output_dir)

        ydl_opts = self._build_ydl_options(output_dir, video_id, quality)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
        except yt_dlp.utils.DownloadError as exc:
            logger.error("yt-dlp failed to download video '%s': %s", video_id, exc)
            raise

        video_file = self._find_video_file(output_dir, video_id)
        if video_file is None:
            raise FileNotFoundError(
                f"Video file not found after download for video '{video_id}' "
                f"in directory: {output_dir}\n"
                "Ensure FFmpeg is installed and available in your PATH."
            )

        logger.info(
            "Video downloaded successfully: %s (%.1f MB)",
            video_file,
            _file_size_mb(video_file),
        )
        return video_file

    def _build_ydl_options(
        self, output_dir: Path, video_id: str, quality: int
    ) -> dict:
        """Build the yt-dlp options dictionary for video download.

        The format selector gives priority to MP4+M4A streams at or below the
        requested height, avoiding re-encoding on most devices. The fallback
        chain ensures we always get *something* even when the preferred codec
        or resolution is unavailable.

        Args:
            output_dir: Directory where the video file will be saved.
            video_id: Used to construct the output filename template.
            quality: Maximum video height in pixels (e.g. 720).

        Returns:
            Dictionary of yt-dlp options.
        """
        # Preference chain:
        # 1. Best MP4 video ≤ quality + M4A audio (no re-encoding needed)
        # 2. Best video ≤ quality + best audio (any codec, merged via FFmpeg)
        # 3. Best combined stream ≤ quality (pre-merged, lower quality)
        # 4. Absolute best available (last resort, ignores quality limit)
        fmt = (
            f"bestvideo[height<={quality}][ext=mp4]+bestaudio[ext=m4a]"
            f"/bestvideo[height<={quality}]+bestaudio"
            f"/best[height<={quality}]"
            f"/best"
        )

        opts: dict = {
            "format": fmt,
            # Always merge streams into a single MP4 via FFmpeg
            "merge_output_format": "mp4",
            # Output template: storage/videos/{video_id}/{video_id}.%(ext)s
            "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
            # Force single video download — ignore playlist parameters in URL
            "noplaylist": True,
            # Suppress yt-dlp's own stdout — we handle all logging
            "quiet": True,
            "no_warnings": True,
            "writethumbnail": False,
            # Retry on transient network errors
            "retries": 5,
            "fragment_retries": 5,
        }

        # Cookie injection — fixes HTTP 403 bot-detection on signed stream URLs.
        # Priority: cookies.txt file > browser extraction.
        cookies_file = self._settings.ytdlp_cookies_file.strip()
        browser = self._settings.ytdlp_cookie_browser.strip()
        if cookies_file and Path(cookies_file).exists():
            opts["cookiefile"] = cookies_file
            logger.debug("Using cookies file: %s", cookies_file)
        elif browser:
            opts["cookiesfrombrowser"] = (browser,)
            logger.debug("Using cookies from browser: %s", browser)

        return opts

    def _find_video_file(self, directory: Path, video_id: str) -> Path | None:
        """Locate the video file in a directory regardless of extension.

        yt-dlp may produce different container formats depending on source
        availability and FFmpeg support. This method checks all known video
        extensions in preference order.

        Args:
            directory: Directory to search in.
            video_id: Base filename (without extension) to look for.

        Returns:
            Path to the found file, or None if no video file exists yet.
        """
        for ext in _VIDEO_EXTENSIONS:
            candidate = directory / f"{video_id}.{ext}"
            if candidate.exists():
                return candidate
        return None


def _file_size_mb(path: Path) -> float:
    """Return the file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)
