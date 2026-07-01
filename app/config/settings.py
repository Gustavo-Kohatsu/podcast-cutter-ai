"""Centralized application configuration.

All settings are loaded from environment variables or a .env file.
Defaults are chosen to work out-of-the-box on a CPU-only machine.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with automatic .env loading.

    All fields can be overridden via environment variables or a .env file.
    Field names map directly to environment variable names (case-insensitive).

    Example:
        WHISPER_MODEL=small OLLAMA_MODEL=gemma3:12b python main.py "..."
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Whisper ───────────────────────────────────────────────────────────────
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str | None = None

    # ── Video ─────────────────────────────────────────────────────────────────
    # Maximum video resolution height in pixels. Common values: 360, 480, 720, 1080
    video_quality: int = 720

    # ── yt-dlp ────────────────────────────────────────────────────────────────
    # Path to a Netscape-format cookies.txt exported from your browser.
    # Recommended fix for HTTP 403 errors — see README for export instructions.
    # Leave empty to skip cookie injection.
    ytdlp_cookies_file: str = ""

    # Fallback: extract cookies directly from a browser process.
    # Broken on Windows with Chrome/Brave 127+ (App-Bound Encryption).
    # Only reliable with Firefox. Leave empty to disable.
    ytdlp_cookie_browser: str = ""

    # ── Ollama ────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.3
    # 16k context fits comfortably for 15-min chunks (~3-4k tokens each)
    ollama_num_ctx: int = 16384
    # Seconds before a single LLM call is aborted (prevents zombie requests)
    ollama_timeout_seconds: int = 120

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_duration_minutes: int = 15
    chunk_overlap_seconds: int = 45
    candidates_per_chunk: int = 4

    # ── Pipeline ──────────────────────────────────────────────────────────────
    min_clip_duration: int = 60
    max_clip_duration: int = 120
    max_clips: int = 10

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_dir: Path = Path("storage")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_dir: Path = Path("logs")

    # ── Derived paths (read-only properties) ──────────────────────────────────

    @property
    def audio_dir(self) -> Path:
        """Directory where downloaded audio files are stored."""
        return self.storage_dir / "audio"

    @property
    def video_dir(self) -> Path:
        """Directory where downloaded video files are stored."""
        return self.storage_dir / "videos"

    @property
    def transcripts_dir(self) -> Path:
        """Directory where transcription JSON files are stored."""
        return self.storage_dir / "transcripts"

    @property
    def jobs_dir(self) -> Path:
        """Directory where final clip detection results are stored."""
        return self.storage_dir / "jobs"

    @property
    def cuts_dir(self) -> Path:
        """Directory where extracted clip video files are stored."""
        return self.storage_dir / "cuts"

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        normalized = value.upper()
        if normalized not in valid_levels:
            raise ValueError(f"Invalid log level '{value}'. Must be one of: {valid_levels}")
        return normalized

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, value: str) -> str:
        """Ensure Whisper device is either cpu or cuda."""
        valid_devices = {"cpu", "cuda"}
        normalized = value.lower()
        if normalized not in valid_devices:
            raise ValueError(f"Invalid device '{value}'. Must be one of: {valid_devices}")
        return normalized

    @field_validator("ollama_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure temperature is within valid range."""
        if not (0.0 <= value <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {value}")
        return value
