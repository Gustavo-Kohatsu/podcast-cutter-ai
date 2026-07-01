"""Prompt templates for the two-phase viral clip detection pipeline.

Phase 1 — Candidate detection:
    One prompt per transcript chunk. Asks the LLM for the top N candidates
    within that 15-minute window. Produces a small, focused JSON per chunk.

Phase 2 — Final ranking:
    One prompt with all deduplicated candidates. Asks the LLM to select the
    best final clips considering diversity and overall viral potential.

Design principles:
- Prompts are pure functions: same input → same output. No side effects.
- The JSON schema is embedded in every prompt so the model knows the exact
  contract without relying on format="json" alone.
- Instructions are written in English for maximum cross-model compatibility.
- All timestamps reference the original video (not chunk-relative) so the
  downstream video cutter can use them directly without any offset math.
"""

from app.config.settings import Settings
from app.schemas.clip import ViralClip
from app.services.chunker import TranscriptChunk

# ── Shared components ──────────────────────────────────────────────────────────

_VIRAL_CRITERIA = """\
## WHAT MAKES A GREAT VIRAL CLIP
Prioritize moments that contain ONE OR MORE of the following:
- Surprising, counterintuitive, or shocking revelations
- Strong emotional reactions (genuine laughter, anger, inspiration, vulnerability)
- Highly actionable, immediately applicable tips or advice
- Controversial or thought-provoking statements people will debate
- Personal stories with a clear emotional arc
- "Aha" moments, epiphanies, or paradigm shifts
- Quotable "mic drop" phrases
- Moments of high energy or dramatic tension\
"""

_JSON_SCHEMA = """\
Return ONLY a valid JSON object — no markdown fences, no explanations, no extra text.

CRITICAL DURATION RULE: end_time - start_time MUST be between {min_duration}s and {max_duration}s.
  VALID example:   start_time: 932.1, end_time: 992.1  (60s clip - CORRECT)
  INVALID example: start_time: 932.1, end_time: 934.1  (2s clip - TOO SHORT, REJECT)
  INVALID example: start_time: 0.0,   end_time: 945.0  (945s clip - TOO LONG, REJECT)

JSON format:
{{
  "clips": [
    {{
      "title": "Título real e criativo do momento viral",
      "start_time": 932.1,
      "end_time": 992.1,
      "justification": "Explicação clara de por que este momento tem potencial viral.",
      "viral_score": 8.5
    }}
  ]
}}\
"""

# ── Phase 1: Candidate detection ──────────────────────────────────────────────

_CANDIDATE_PROMPT = """\
You are an expert viral content curator for YouTube Shorts, Instagram Reels, and TikTok.

Analyze the TRANSCRIPT SEGMENT below and identify the {n_candidates} moments \
with the highest viral potential.

{viral_criteria}

## CLIP RULES (MANDATORY)
- Duration: between {min_duration}s and {max_duration}s
- Must be self-contained: a viewer with zero context must understand it
- Start slightly before the key moment to provide minimal context
- End at a natural pause, punchline, or conclusion — NEVER mid-sentence
- Timestamps MUST fall within this segment window: {start_time:.1f}s - {end_time:.1f}s
- Titles must be written in the SAME LANGUAGE as the transcript

## OUTPUT FORMAT
{json_schema}

## TRANSCRIPT SEGMENT
{label} | {time_label} | {segment_count} segments

{transcript_text}\
"""


def build_candidate_prompt(chunk: TranscriptChunk, settings: Settings) -> str:
    """Build the Phase 1 prompt to extract viral candidates from a single chunk.

    Args:
        chunk: The transcript chunk to analyze.
        settings: Application settings for clip duration constraints.

    Returns:
        Fully formatted prompt string ready to send to the LLM.
    """
    schema = _JSON_SCHEMA.format(
        min_duration=settings.min_clip_duration,
        max_duration=settings.max_clip_duration,
    )
    return _CANDIDATE_PROMPT.format(
        n_candidates=settings.candidates_per_chunk,
        viral_criteria=_VIRAL_CRITERIA,
        min_duration=settings.min_clip_duration,
        max_duration=settings.max_clip_duration,
        start_time=chunk.start_time,
        end_time=chunk.end_time,
        json_schema=schema,
        label=chunk.label,
        time_label=chunk.time_label(),
        segment_count=chunk.segment_count,
        transcript_text=chunk.to_timestamped_text(),
    )


# ── Phase 2: Final ranking ─────────────────────────────────────────────────────

_RANKING_PROMPT = """\
You are an expert viral content curator for YouTube Shorts, Instagram Reels, and TikTok.

Below are {n_candidates} clip candidates extracted from a {duration_label} podcast/live stream.

Your task: select the {max_clips} BEST clips for maximum social media virality.

{viral_criteria}

## SELECTION RULES
- Maximize viral potential — prefer clips with higher viral_score
- Ensure topic diversity — avoid selecting two very similar clips
- Prefer clips that are fully self-contained
- IMPORTANT: keep the original start_time, end_time, and justification exactly as-is
  Do NOT invent new timestamps or modify existing ones

## OUTPUT FORMAT
{json_schema}

## CANDIDATES ({n_candidates} total)
{candidates_text}\
"""


def build_ranking_prompt(
    candidates: list[ViralClip],
    settings: Settings,
    total_duration: float,
) -> str:
    """Build the Phase 2 prompt to rank all candidates and select the final clips.

    Args:
        candidates: All deduplicated ViralClip candidates from Phase 1.
        settings: Application settings (max_clips, duration bounds).
        total_duration: Full video duration in seconds (used for display only).

    Returns:
        Fully formatted prompt string ready to send to the LLM.
    """
    schema = _JSON_SCHEMA.format(
        min_duration=settings.min_clip_duration,
        max_duration=settings.max_clip_duration,
    )
    return _RANKING_PROMPT.format(
        n_candidates=len(candidates),
        duration_label=_fmt_duration(total_duration),
        max_clips=settings.max_clips,
        viral_criteria=_VIRAL_CRITERIA,
        json_schema=schema,
        candidates_text=_format_candidates(candidates),
    )


def _format_candidates(candidates: list[ViralClip]) -> str:
    """Format candidate clips as a numbered, human-readable list.

    Args:
        candidates: List of ViralClip instances to format.

    Returns:
        Multi-line string with one candidate per block.
    """
    blocks = []
    for i, clip in enumerate(candidates, start=1):
        blocks.append(
            f"[{i:02d}] Score {clip.viral_score:.1f}/10 | "
            f"{clip.start_time:.1f}s → {clip.end_time:.1f}s | "
            f"{clip.formatted_duration()}\n"
            f"     Title: {clip.title}\n"
            f"     Why:   {clip.justification}"
        )
    return "\n\n".join(blocks)


# ── Shared formatting helpers ──────────────────────────────────────────────────


def _fmt_duration(total_seconds: float) -> str:
    """Convert seconds to a human-readable duration string."""
    total = int(total_seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
