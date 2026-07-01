"""Data models for viral clip detection results.

These models represent the structured output of the LLM clip detection step.
They are validated by Pydantic on construction, ensuring the LLM response
conforms to the expected contract before any downstream code touches it.
"""

from pydantic import BaseModel, Field, field_validator, model_validator


class ViralClip(BaseModel):
    """A single viral clip candidate identified by the LLM.

    Attributes:
        title: Engaging, shareable title for the clip.
        start_time: Clip start time in seconds from the beginning of the video.
        end_time: Clip end time in seconds from the beginning of the video.
        justification: Explanation of why this moment has viral potential.
        viral_score: Viralness score from 0.0 (not viral) to 10.0 (extremely viral).
    """

    title: str = Field(min_length=3, max_length=200, description="Engaging clip title")
    start_time: float = Field(ge=0.0, description="Clip start time in seconds")
    end_time: float = Field(ge=0.0, description="Clip end time in seconds")
    justification: str = Field(
        min_length=10,
        description="Why this moment has viral potential",
    )
    viral_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Viral potential score from 0.0 to 10.0",
    )

    @field_validator("title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        """Remove leading/trailing whitespace from title."""
        return value.strip()

    @model_validator(mode="after")
    def validate_clip_order(self) -> "ViralClip":
        """Ensure end_time is strictly greater than start_time."""
        if self.end_time <= self.start_time:
            raise ValueError(
                f"Clip end_time ({self.end_time:.2f}s) must be greater than "
                f"start_time ({self.start_time:.2f}s) for clip: '{self.title}'"
            )
        return self

    @property
    def duration(self) -> float:
        """Return the clip duration in seconds."""
        return self.end_time - self.start_time

    def formatted_duration(self) -> str:
        """Return the clip duration as a human-readable string.

        Returns:
            String in the format 'Xs' or 'Xm Ys'.

        Example:
            >>> clip.formatted_duration()
            '1m 23s'
        """
        total = int(self.duration)
        minutes, seconds = divmod(total, 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"


class ClipDetectionResult(BaseModel):
    """Complete result of the viral clip detection step for a single video.

    This is the final output of the pipeline for the current MVP.
    It is serialized as JSON and saved to storage/results/{video_id}_clips.json.

    Attributes:
        video_id: The YouTube video ID this result belongs to.
        url: The original YouTube URL.
        clips: List of viral clip candidates, sorted by viral_score descending.
        total_clips_found: Number of clips returned by the LLM.
    """

    video_id: str
    url: str
    clips: list[ViralClip] = Field(default_factory=list)

    @property
    def total_clips_found(self) -> int:
        """Return the number of detected clips."""
        return len(self.clips)

    def sorted_by_score(self) -> list[ViralClip]:
        """Return clips sorted by viral_score in descending order.

        Returns:
            New list of ViralClip instances ordered from highest to lowest score.
        """
        return sorted(self.clips, key=lambda c: c.viral_score, reverse=True)

    def summary(self) -> str:
        """Return a human-readable summary of the detection results.

        Returns:
            Multi-line string with title, score, and duration for each clip.
        """
        if not self.clips:
            return "No viral clips detected."

        lines = [f"Detected {self.total_clips_found} viral clip(s):\n"]
        for i, clip in enumerate(self.sorted_by_score(), start=1):
            lines.append(
                f"  {i:2}. [{clip.viral_score:4.1f}/10] "
                f"{clip.formatted_duration():>6}  —  {clip.title}"
            )
        return "\n".join(lines)
