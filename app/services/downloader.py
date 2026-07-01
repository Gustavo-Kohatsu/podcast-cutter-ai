"""Audio download service using yt-dlp.

Responsible for downloading the audio stream from a YouTube URL and saving
it as an audio file in the configured storage directory.

Only the audio stream is downloaded — the full video is never fetched at this
stage, keeping bandwidth and disk usage minimal during the transcription phase.
"""

from pathlib import Path

import static_ffmpeg
import yt_dlp

from app.config.settings import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Audio extensions that yt-dlp may produce, in preference order
_AUDIO_EXTENSIONS = ("mp3", "m4a", "opus", "webm", "ogg", "wav", "aac")


class AudioDownloader:
    """Downloads audio from YouTube URLs using yt-dlp.

    This service downloads only the audio stream of a video, optionally
    converting it to MP3 via FFmpeg postprocessing. If FFmpeg is not
    available, yt-dlp will fall back to the best available audio container.

    Args:
        settings: Application settings containing storage path configuration.

    Example:
        >>> downloader = AudioDownloader(settings)
        >>> audio_path = downloader.download("https://youtu.be/...", "dQw4w9WgXcQ")
        >>> print(audio_path)
        storage/audio/dQw4w9WgXcQ/dQw4w9WgXcQ.mp3
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Inject ffmpeg/ffprobe binaries from the venv into the current process PATH.
        # This runs once per instance — no system installation required.
        static_ffmpeg.add_paths()

    def download(self, url: str, video_id: str) -> Path:
        """Download the audio stream from the given YouTube URL.

        Creates a subdirectory under ``settings.audio_dir`` named after the
        ``video_id`` and places the audio file there. If the file already
        exists from a previous run, it is returned immediately without
        re-downloading.

        Args:
            url: The YouTube URL to download audio from.
            video_id: The YouTube video ID, used as directory and filename.

        Returns:
            Path to the downloaded audio file.

        Raises:
            yt_dlp.utils.DownloadError: If yt-dlp fails to download the video.
            FileNotFoundError: If the audio file cannot be located after download.
        """
        output_dir = self._settings.audio_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check for cached file from a previous run
        cached = self._find_audio_file(output_dir, video_id)
        if cached is not None:
            logger.info("Audio already exists, skipping download: %s", cached)
            return cached

        logger.info("Downloading audio for video '%s'...", video_id)
        logger.debug("Output directory: %s", output_dir)

        ydl_opts = self._build_ydl_options(output_dir, video_id)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = self._find_audio_file(output_dir, video_id)
        if audio_file is None:
            raise FileNotFoundError(
                f"Audio file not found after download for video '{video_id}' "
                f"in directory: {output_dir}\n"
                "Ensure FFmpeg is installed and available in your PATH."
            )

        logger.info("Audio downloaded successfully: %s (%.1f MB)", audio_file, _file_size_mb(audio_file))
        return audio_file

    def _build_ydl_options(self, output_dir: Path, video_id: str) -> dict:
        """Build the yt-dlp options dictionary.

        Args:
            output_dir: Directory where the audio file will be saved.
            video_id: Used to construct the output filename template.

        Returns:
            Dictionary of yt-dlp options.
        """
        opts: dict = {
            # Download best audio quality available
            "format": "bestaudio/best",
            # Output template: storage/audio/{video_id}/{video_id}.%(ext)s
            "outtmpl": str(output_dir / f"{video_id}.%(ext)s"),
            # Convert to MP3 via FFmpeg postprocessor
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            # Force single video download — ignore &list= playlist parameter in URL
            "noplaylist": True,
            # Suppress yt-dlp's own output — we handle logging
            "quiet": True,
            "no_warnings": True,
            "writethumbnail": False,
            # Retry on transient network errors
            "retries": 5,
            "fragment_retries": 5,
        }

        cookies_file = self._settings.ytdlp_cookies_file.strip()
        browser = self._settings.ytdlp_cookie_browser.strip()
        if cookies_file and Path(cookies_file).exists():
            opts["cookiefile"] = cookies_file
        elif browser:
            opts["cookiesfrombrowser"] = (browser,)

        return opts

    def _find_audio_file(self, directory: Path, video_id: str) -> Path | None:
        """Locate the audio file in a directory regardless of extension.

        yt-dlp may produce different extensions depending on whether FFmpeg
        is available and what the source format was. This method checks for
        all known audio extensions in preference order.

        Args:
            directory: Directory to search in.
            video_id: Base filename (without extension) to look for.

        Returns:
            Path to the found file, or None if no audio file exists yet.
        """
        for ext in _AUDIO_EXTENSIONS:
            candidate = directory / f"{video_id}.{ext}"
            if candidate.exists():
                return candidate
        return None


def _file_size_mb(path: Path) -> float:
    """Return the file size in megabytes."""
    return path.stat().st_size / (1024 * 1024)
