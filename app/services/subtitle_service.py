"""Subtitle generation service — Step 6 of the pipeline.

Generates SRT and ASS subtitle files for each detected clip using the
original Whisper transcription as the source of timed text segments.

Design decisions:
- Timestamps are re-anchored to clip-relative time so that the first
  subtitle of every clip always starts near 0:00:00,000.
- Segments are word-wrapped to a 42-character column limit with at most
  2 display lines per block. Segments that overflow 2 lines are split
  into multiple timed blocks with proportional time allocation.
- The SRT file is widely compatible (players, ffmpeg, DaVinci Resolve).
- The ASS file uses a production-ready Default style (1280x720, Arial 52,
  white text with black outline, bottom-centre alignment) so that future
  animation overrides ({fad}, {move}, {an}) can be added per-line
  without needing to restructure the file.
- Both files are cached: if both already exist the step is skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config.settings import Settings
from app.schemas.clip import ClipDetectionResult, ViralClip
from app.schemas.transcript import Transcript
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Maximum characters per subtitle line before wrapping.
# 42 chars fits comfortably on a 1280x720 frame at font size 52.
_MAX_LINE_CHARS = 42

# ASS file header template. PlayResX/Y set the virtual coordinate space;
# the subtitle renderer scales this to the actual video resolution.
_ASS_HEADER = """\
[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: 1280
PlayResY: 720

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,52,&H00FFFFFF,&H000000FF,&H00000000,&HA0000000,0,0,0,0,100,100,0,0,1,3,1,2,80,80,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


@dataclass
class SubtitleEntry:
    """A single timed subtitle block.

    Attributes:
        index: 1-based sequential number (SRT requires this).
        start: Start time in seconds, relative to the clip start.
        end: End time in seconds, relative to the clip start.
        text: Display text, with ``\\n`` separating display lines.
    """

    index: int
    start: float
    end: float
    text: str


class SubtitleService:
    """Generates SRT and ASS subtitle files for each viral clip.

    Uses the original Whisper transcript segments as the timed text source.
    Timestamps are converted to clip-relative time so subtitles sync with
    the standalone cut MP4, not the full source video.

    Args:
        settings: Application settings (used for output directory paths).

    Example:
        >>> service = SubtitleService(settings)
        >>> pairs = service.generate_all(result, transcript)
        >>> print(pairs[0])  # (PosixPath('storage/subtitles/…/cut_001.srt'), …)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_all(
        self,
        result: ClipDetectionResult,
        transcript: Transcript,
    ) -> list[tuple[Path, Path]]:
        """Generate subtitle files for every clip in a detection result.

        Clips are processed in descending viral score order so that
        ``cut_001`` always corresponds to the highest-scoring clip,
        matching the naming convention of the VideoCutter step.

        Args:
            result: Detection result containing the list of viral clips.
            transcript: Full video transcript used as the subtitle source.

        Returns:
            List of ``(srt_path, ass_path)`` tuples, one per clip.
        """
        clips = result.sorted_by_score()
        pairs: list[tuple[Path, Path]] = []

        for i, clip in enumerate(clips, start=1):
            pair = self._generate_one(clip, i, transcript, result.video_id)
            pairs.append(pair)

        logger.info(
            "Subtitle generation complete — %d pair(s) saved to: %s",
            len(pairs),
            self._settings.subtitles_dir / result.video_id,
        )
        return pairs

    # ── Private ───────────────────────────────────────────────────────────────

    def _generate_one(
        self,
        clip: ViralClip,
        clip_index: int,
        transcript: Transcript,
        video_id: str,
    ) -> tuple[Path, Path]:
        """Generate SRT and ASS files for a single clip.

        Args:
            clip: The viral clip to generate subtitles for.
            clip_index: 1-based clip number, used for the output filename.
            transcript: Full video transcript.
            video_id: YouTube video ID, used for the output subdirectory.

        Returns:
            ``(srt_path, ass_path)`` for this clip.
        """
        output_dir = self._settings.subtitles_dir / video_id
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = f"cut_{clip_index:03d}"
        srt_path = output_dir / f"{stem}.srt"
        ass_path = output_dir / f"{stem}.ass"

        if srt_path.exists() and ass_path.exists():
            logger.info(
                "Subtitle cache hit — skipping %s ('%s')",
                stem,
                clip.title[:50],
            )
            return srt_path, ass_path

        entries = self._extract_entries(clip, transcript)

        if not entries:
            logger.warning(
                "No transcript segments found for clip '%s' [%.1fs - %.1fs]. "
                "Writing empty subtitle files.",
                clip.title[:50],
                clip.start_time,
                clip.end_time,
            )

        srt_path.write_text(self._build_srt(entries), encoding="utf-8")
        ass_path.write_text(self._build_ass(entries, clip), encoding="utf-8")

        logger.info(
            "  %s — %d subtitle block(s) → %s / %s",
            stem,
            len(entries),
            srt_path.name,
            ass_path.name,
        )
        return srt_path, ass_path

    def _extract_entries(
        self,
        clip: ViralClip,
        transcript: Transcript,
    ) -> list[SubtitleEntry]:
        """Extract and re-anchor transcript segments that overlap the clip window.

        Only segments whose time window intersects ``[clip.start_time, clip.end_time)``
        are included. Their timestamps are offset by ``-clip.start_time`` so the
        first subtitle always appears near 0:00:00,000.

        Args:
            clip: The target clip defining the time window.
            transcript: Full transcript to extract segments from.

        Returns:
            Ordered list of SubtitleEntry instances ready for serialisation.
        """
        relevant = [
            seg
            for seg in transcript.segments
            if seg.end > clip.start_time and seg.start < clip.end_time
        ]

        entries: list[SubtitleEntry] = []
        global_index = 1

        for seg in relevant:
            # Re-anchor to clip-relative time, clamped to [0, clip.duration]
            rel_start = max(seg.start - clip.start_time, 0.0)
            rel_end = min(seg.end - clip.start_time, clip.duration)

            if rel_end <= rel_start:
                continue

            new_entries = self._segment_to_entries(
                global_index,
                rel_start,
                rel_end,
                seg.text.strip(),
            )
            entries.extend(new_entries)
            global_index += len(new_entries)

        # Re-number sequentially after potential multi-entry splits
        for i, entry in enumerate(entries, start=1):
            entry.index = i

        return entries

    def _segment_to_entries(
        self,
        index_start: int,
        start: float,
        end: float,
        text: str,
    ) -> list[SubtitleEntry]:
        """Convert one transcript segment into one or more subtitle entries.

        Text is wrapped to ``_MAX_LINE_CHARS`` characters per line.  If the
        result has more than 2 display lines the segment is divided into
        groups of 2 lines each, with time allocated proportionally to word
        count so faster speech gets more time.

        Args:
            index_start: 1-based index for the first generated entry.
            start: Clip-relative start time in seconds.
            end: Clip-relative end time in seconds.
            text: Segment text to display.

        Returns:
            One or more SubtitleEntry instances covering ``[start, end]``.
        """
        lines = _wrap_lines(text, max_chars=_MAX_LINE_CHARS)

        if len(lines) <= 2:
            return [
                SubtitleEntry(
                    index=index_start,
                    start=round(start, 3),
                    end=round(end, 3),
                    text="\n".join(lines),
                )
            ]

        # Overflow: split into groups of 2 lines, time-distributed by word count
        groups = [lines[i : i + 2] for i in range(0, len(lines), 2)]
        word_counts = [sum(len(line.split()) for line in g) for g in groups]
        total_words = sum(word_counts) or 1
        duration = end - start

        entries: list[SubtitleEntry] = []
        current_time = start

        for i, (group, wc) in enumerate(zip(groups, word_counts, strict=True)):
            is_last = i == len(groups) - 1
            fraction = wc / total_words
            entry_end = end if is_last else round(current_time + fraction * duration, 3)
            entries.append(
                SubtitleEntry(
                    index=index_start + i,
                    start=round(current_time, 3),
                    end=entry_end,
                    text="\n".join(group),
                )
            )
            current_time = entry_end

        return entries

    # ── Serialisers ───────────────────────────────────────────────────────────

    def _build_srt(self, entries: list[SubtitleEntry]) -> str:
        """Serialise entries to the SRT format.

        Each block is:
            <index>
            HH:MM:SS,mmm --> HH:MM:SS,mmm
            <text>
        Blocks are separated by a blank line.
        """
        blocks: list[str] = []
        for entry in entries:
            start_ts = _srt_timestamp(entry.start)
            end_ts = _srt_timestamp(entry.end)
            blocks.append(f"{entry.index}\n{start_ts} --> {end_ts}\n{entry.text}")
        return "\n\n".join(blocks) + ("\n" if blocks else "")

    def _build_ass(self, entries: list[SubtitleEntry], clip: ViralClip) -> str:
        """Serialise entries to the ASS (Advanced SubStation Alpha) format.

        The header defines a single Default style. Each subtitle block
        becomes one Dialogue line. Multi-line text uses the ASS hard
        line-break tag ``{\\N}``.

        The resulting file is ready for future per-line animation tags
        without structural changes.
        """
        header = _ASS_HEADER.format(title=_escape_ass(clip.title))
        dialogue_lines: list[str] = []

        for entry in entries:
            start_ts = _ass_timestamp(entry.start)
            end_ts = _ass_timestamp(entry.end)
            # Replace Python newline with ASS hard line-break override tag
            text = entry.text.replace("\n", r"{\N}")
            dialogue_lines.append(f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{text}")

        return header + "\n".join(dialogue_lines) + ("\n" if dialogue_lines else "")


# ── Module-level helpers ──────────────────────────────────────────────────────


def _wrap_lines(text: str, max_chars: int = _MAX_LINE_CHARS) -> list[str]:
    """Break ``text`` into lines of at most ``max_chars`` characters.

    Breaks only occur at word boundaries.  Single words longer than
    ``max_chars`` are placed on their own line without truncation.

    Args:
        text: Input text to wrap.
        max_chars: Maximum characters per output line.

    Returns:
        List of line strings (no trailing newlines).
    """
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        # Adding a space before the word: +1 (unless it's the first word)
        space = 1 if current else 0
        if current and current_len + space + word_len > max_chars:
            lines.append(" ".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += space + word_len

    if current:
        lines.append(" ".join(current))

    return lines


def _srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp: ``HH:MM:SS,mmm``."""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _ass_timestamp(seconds: float) -> str:
    """Convert seconds to ASS timestamp: ``H:MM:SS.cc`` (centiseconds)."""
    total_cs = int(seconds * 100)
    cs = total_cs % 100
    total_s = total_cs // 100
    h = total_s // 3600
    m = (total_s % 3600) // 60
    s = total_s % 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape_ass(text: str) -> str:
    """Escape characters that have special meaning in ASS Script Info fields."""
    return text.replace("{", "").replace("}", "")
