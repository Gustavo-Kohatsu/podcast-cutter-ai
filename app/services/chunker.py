"""Transcript chunking service.

Splits a long transcript into smaller time-based windows so that each window
fits comfortably within the LLM's context window.

Design decisions:
- Each chunk extends slightly into the next window (overlap) to avoid missing
  viral moments that happen near a chunk boundary.
- Original timestamps from Whisper are preserved untouched — the LLM always
  receives and returns timestamps relative to the full video, not the chunk.
- Chunking is a pure transformation: same input always produces the same output.
"""

import math
from dataclasses import dataclass

from app.schemas.transcript import Transcript, TranscriptSegment
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    """Parameters that control how a transcript is split into chunks.

    Attributes:
        chunk_duration_s: Length of each chunk in seconds (default: 15 min).
        overlap_s: How many seconds each chunk extends into the next one,
            preventing clips near a boundary from being missed (default: 45s).
    """

    chunk_duration_s: float = 900.0
    overlap_s: float = 45.0


@dataclass
class TranscriptChunk:
    """A temporal slice of a full transcript.

    Contains only the segments that fall within [start_time, end_time + overlap].
    Timestamps inside the segments are always relative to the original video start.

    Attributes:
        index: Zero-based position of this chunk in the sequence.
        total: Total number of chunks the transcript was split into.
        segments: Transcript segments belonging to this chunk.
        start_time: Nominal start of this chunk's window (seconds).
        end_time: Nominal end of this chunk's window including overlap (seconds).
        video_id: YouTube video ID of the source video.
        url: Original YouTube URL.
    """

    index: int
    total: int
    segments: list[TranscriptSegment]
    start_time: float
    end_time: float
    video_id: str
    url: str

    @property
    def label(self) -> str:
        """Human-readable chunk identifier, e.g. 'Chunk 3/12'."""
        return f"Chunk {self.index + 1}/{self.total}"

    @property
    def segment_count(self) -> int:
        """Number of transcript segments in this chunk."""
        return len(self.segments)

    @property
    def duration(self) -> float:
        """Window duration in seconds."""
        return self.end_time - self.start_time

    def to_timestamped_text(self) -> str:
        """Return segments formatted as '[Xs - Ys] text' lines using float seconds.

        Timestamps are expressed in seconds (e.g. '[932.1s - 1005.3s]') so that
        the LLM naturally returns start_time/end_time as floats in its JSON
        response — matching exactly what the Pydantic schema expects.

        Using MM:SS format here would cause the model to mirror that format in
        the JSON output, breaking float validation downstream.

        Returns:
            Multi-line string with one segment per line.
        """
        return "\n".join(
            f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}" for seg in self.segments
        )

    def time_label(self) -> str:
        """Return a compact time range label for logging, e.g. '0m00s - 15m45s'."""
        return f"{_fmt_seconds(self.start_time)} - {_fmt_seconds(self.end_time)}"


class TranscriptChunker:
    """Splits a Transcript into time-based TranscriptChunk instances.

    Example:
        >>> chunker = TranscriptChunker()
        >>> config = ChunkConfig(chunk_duration_s=900, overlap_s=45)
        >>> chunks = chunker.chunk(transcript, config)
        >>> print(f"Split into {len(chunks)} chunks")
    """

    def chunk(self, transcript: Transcript, config: ChunkConfig) -> list[TranscriptChunk]:
        """Split a transcript into sequential chunks with overlap.

        Each chunk covers ``config.chunk_duration_s`` seconds of audio and
        extends an additional ``config.overlap_s`` seconds into the next chunk
        to ensure no viral moments at boundaries are missed.

        Args:
            transcript: Full transcript to split.
            config: Chunking strategy parameters.

        Returns:
            Ordered list of non-empty TranscriptChunk instances.
            Returns a single chunk wrapping the full transcript if it is
            shorter than one chunk duration.
        """
        if not transcript.segments:
            logger.warning("Transcript has no segments — returning empty chunk list.")
            return []

        total_duration = transcript.duration
        num_chunks = max(1, math.ceil(total_duration / config.chunk_duration_s))

        logger.info(
            "Splitting %.0f min transcript → %d chunks "
            "(~%.0f min each, %.0fs overlap)",
            total_duration / 60,
            num_chunks,
            config.chunk_duration_s / 60,
            config.overlap_s,
        )

        chunks: list[TranscriptChunk] = []

        for i in range(num_chunks):
            window_start = i * config.chunk_duration_s
            # Extend end into the next chunk by overlap_s to catch boundary clips
            window_end = min(
                (i + 1) * config.chunk_duration_s + config.overlap_s,
                total_duration,
            )

            chunk_segments = [
                seg
                for seg in transcript.segments
                if window_start <= seg.start < window_end
            ]

            if not chunk_segments:
                logger.debug("Chunk %d/%d is empty — skipping.", i + 1, num_chunks)
                continue

            chunks.append(
                TranscriptChunk(
                    index=i,
                    total=num_chunks,
                    segments=chunk_segments,
                    start_time=window_start,
                    end_time=window_end,
                    video_id=transcript.video_id,
                    url=transcript.url,
                )
            )

        logger.info("Created %d non-empty chunk(s).", len(chunks))
        return chunks


def _fmt_seconds(seconds: float) -> str:
    """Convert seconds to compact 'Xm00s' format for log messages."""
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m}m{s:02d}s"
