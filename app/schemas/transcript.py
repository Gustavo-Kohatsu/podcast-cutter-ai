"""Data models for audio transcription results.

These Pydantic models represent the output of the Whisper transcription step.
They are immutable value objects: once created, they are only read, never mutated.
"""

from pydantic import BaseModel, Field, field_validator


class TranscriptSegment(BaseModel):
    """A single timed segment from a Whisper transcription.

    Corresponds to one chunk of speech as identified by faster-whisper.
    Timestamps are in seconds from the start of the audio.

    Attributes:
        start: Start time of the segment in seconds.
        end: End time of the segment in seconds.
        text: Transcribed text content of this segment, stripped of whitespace.
    """

    start: float = Field(ge=0.0, description="Segment start time in seconds")
    end: float = Field(ge=0.0, description="Segment end time in seconds")
    text: str = Field(min_length=1, description="Transcribed text content")

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        """Remove leading/trailing whitespace from segment text."""
        return value.strip()

    @field_validator("end")
    @classmethod
    def end_must_be_after_start(cls, end: float, info) -> float:
        """Ensure the segment end time is after its start time."""
        start = info.data.get("start")
        if start is not None and end <= start:
            raise ValueError(
                f"Segment end ({end:.2f}s) must be greater than start ({start:.2f}s)"
            )
        return end

    def duration(self) -> float:
        """Return the duration of this segment in seconds."""
        return self.end - self.start

    def formatted_timestamp(self) -> str:
        """Return a human-readable timestamp string for display purposes.

        Returns:
            A string in the format '[MM:SS.s - MM:SS.s]'.

        Example:
            >>> seg = TranscriptSegment(start=65.3, end=72.8, text="Hello")
            >>> seg.formatted_timestamp()
            '[01:05.3 - 01:12.8]'
        """
        return f"[{_format_seconds(self.start)} - {_format_seconds(self.end)}]"


class Transcript(BaseModel):
    """Complete transcription result for a YouTube video.

    Contains all timed segments and metadata produced by the transcription step.

    Attributes:
        video_id: The YouTube video ID (11-character string).
        url: The original YouTube URL.
        language: ISO 639-1 language code detected by Whisper (e.g. "pt", "en").
        duration: Total audio duration in seconds.
        segments: Ordered list of transcribed segments.
    """

    video_id: str = Field(min_length=11, max_length=11)
    url: str
    language: str
    duration: float = Field(gt=0.0)
    segments: list[TranscriptSegment] = Field(default_factory=list)

    def to_plain_text(self) -> str:
        """Return the full transcript as a single plain text string.

        Returns:
            All segment texts joined by spaces.
        """
        return " ".join(seg.text for seg in self.segments)

    def to_timestamped_text(self) -> str:
        """Return the transcript formatted with timestamps per segment.

        This is the format sent to the LLM for clip detection.

        Returns:
            Multi-line string where each line is: '[start - end] text'

        Example:
            [00:00.0 - 00:05.3] Hello and welcome to the podcast.
            [00:05.3 - 00:12.1] Today we are talking about AI.
        """
        lines = [
            f"{seg.formatted_timestamp()} {seg.text}"
            for seg in self.segments
        ]
        return "\n".join(lines)

    @property
    def segment_count(self) -> int:
        """Return the total number of transcribed segments."""
        return len(self.segments)


def _format_seconds(seconds: float) -> str:
    """Convert a float seconds value to MM:SS.s format.

    Args:
        seconds: Time in seconds.

    Returns:
        Formatted string like '01:35.7'.
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes:02d}:{secs:04.1f}"
