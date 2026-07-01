"""Viral clip detection service — two-phase pipeline using local Ollama LLM.

Architecture: Two-Phase Detection
──────────────────────────────────
Phase 1 — Candidate extraction (one LLM call per chunk):
    The transcript is split into ~15-minute chunks by TranscriptChunker.
    Each chunk is sent independently to Ollama, which returns 3-5 candidates.
    A short context window (16k) is used — chunks fit with room to spare.
    If one chunk fails, it is skipped and the pipeline continues.

Phase 2 — Final ranking (one LLM call total):
    All candidates from Phase 1 are deduplicated (overlap-based) and sent
    in a compact format to the LLM for a final ranking pass.
    The LLM selects the best N clips considering virality and diversity.
    This step is skipped if the candidate count is already ≤ max_clips.

Why two phases?
    - Better quality: the ranking model sees all candidates side-by-side
      and can enforce diversity, which per-chunk calls cannot.
    - Better performance: each individual LLM call is small and fast.
    - Better resilience: a single bad chunk doesn't kill the pipeline.
"""

import json
import time
from pathlib import Path

import ollama
from pydantic import ValidationError

from app.config.settings import Settings
from app.prompts.clip_detection import build_candidate_prompt, build_ranking_prompt
from app.schemas.clip import ClipDetectionResult, ViralClip
from app.schemas.transcript import Transcript
from app.services.chunker import ChunkConfig, TranscriptChunk, TranscriptChunker
from app.utils.logger import get_logger

logger = get_logger(__name__)


class ClipDetector:
    """Detects viral clip candidates using a two-phase LLM pipeline.

    Args:
        settings: Application settings containing Ollama and chunking config.

    Example:
        >>> detector = ClipDetector(settings)
        >>> result = detector.detect(transcript)
        >>> print(result.summary())
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        # Timeout prevents zombie requests (e.g. model stuck in generation loop).
        # Each chunk should complete in well under 120s on CPU for a 3B model.
        self._client = ollama.Client(
            host=settings.ollama_base_url,
            timeout=settings.ollama_timeout_seconds,
        )
        self._chunker = TranscriptChunker()

    def detect(self, transcript: Transcript) -> ClipDetectionResult:
        """Run the full two-phase viral clip detection on a transcript.

        If a result file already exists at ``storage/jobs/{video_id}.json``
        from a previous run, it is loaded and returned immediately — the
        Ollama connection and all LLM calls are skipped entirely.

        Args:
            transcript: The complete transcription to analyze.

        Returns:
            A ClipDetectionResult with the final selected viral clips,
            already saved to storage/jobs/{video_id}.json.

        Raises:
            ConnectionError: If the Ollama server is not running.
            RuntimeError: If the configured model is not available in Ollama.
        """
        cached_path = self._settings.jobs_dir / f"{transcript.video_id}.json"
        if cached_path.exists():
            logger.info(
                "Job cache found, skipping detection — loading from: %s", cached_path
            )
            return ClipDetectionResult.model_validate_json(
                cached_path.read_text(encoding="utf-8")
            )

        self._verify_ollama_connection()

        total_start = time.perf_counter()
        config = ChunkConfig(
            chunk_duration_s=self._settings.chunk_duration_minutes * 60,
            overlap_s=self._settings.chunk_overlap_seconds,
        )

        chunks = self._chunker.chunk(transcript, config)
        if not chunks:
            logger.warning("No chunks to process — returning empty result.")
            return self._build_and_save_result(transcript, [])

        # ── Phase 1: Extract candidates from each chunk ───────────────────────
        logger.info(
            "Phase 1 — Candidate extraction: %d chunks, model '%s', num_ctx=%d",
            len(chunks),
            self._settings.ollama_model,
            self._settings.ollama_num_ctx,
        )

        all_candidates: list[ViralClip] = []
        for chunk in chunks:
            candidates = self._extract_candidates(chunk, transcript.duration)
            all_candidates.extend(candidates)

        logger.info(
            "Phase 1 complete — %d raw candidates from %d chunks.",
            len(all_candidates),
            len(chunks),
        )

        # ── Deduplication ─────────────────────────────────────────────────────
        unique = self._deduplicate(all_candidates)
        logger.info(
            "After deduplication: %d unique candidates (removed %d overlapping).",
            len(unique),
            len(all_candidates) - len(unique),
        )

        # ── Phase 2: Final ranking ─────────────────────────────────────────────
        if len(unique) > self._settings.max_clips:
            logger.info(
                "Phase 2 — Final ranking: selecting %d from %d candidates...",
                self._settings.max_clips,
                len(unique),
            )
            final_clips = self._rank_candidates(unique, transcript)
        else:
            logger.info(
                "Phase 2 — Skipped (%d candidates ≤ max_clips=%d), "
                "using Phase 1 results directly.",
                len(unique),
                self._settings.max_clips,
            )
            final_clips = sorted(unique, key=lambda c: c.viral_score, reverse=True)

        final_clips = final_clips[: self._settings.max_clips]

        elapsed = time.perf_counter() - total_start
        logger.info(
            "Clip detection complete — %d final clips selected in %.1fs.",
            len(final_clips),
            elapsed,
        )

        return self._build_and_save_result(transcript, final_clips)

    # ── Phase 1 ───────────────────────────────────────────────────────────────

    def _extract_candidates(
        self, chunk: TranscriptChunk, total_duration: float
    ) -> list[ViralClip]:
        """Send one chunk to Ollama and return the detected candidates.

        Failures are logged as warnings and return an empty list so the
        pipeline continues with the remaining chunks.

        Args:
            chunk: The transcript chunk to analyze.
            total_duration: Full video duration in seconds (used to validate
                clip end_time against the entire video, not just the chunk
                window, so clips that slightly exceed the chunk boundary are
                still accepted).

        Returns:
            List of ViralClip candidates (may be empty on failure).
        """
        prompt = build_candidate_prompt(chunk, self._settings)
        step_start = time.perf_counter()

        logger.info(
            "  %s | %s | %d segments | ~%d chars",
            chunk.label,
            chunk.time_label(),
            chunk.segment_count,
            len(prompt),
        )

        try:
            response = self._client.chat(
                model=self._settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": self._settings.ollama_temperature,
                    "num_ctx": self._settings.ollama_num_ctx,
                },
            )
        except Exception as exc:
            logger.warning(
                "  %s — LLM call failed, skipping: %s", chunk.label, exc
            )
            return []

        elapsed = time.perf_counter() - step_start
        candidates = self._parse_clips(
            raw_json=response.message.content,
            max_duration=total_duration + 5,
        )

        logger.info(
            "  %s — %d candidate(s) found in %.1fs.",
            chunk.label,
            len(candidates),
            elapsed,
        )
        return candidates

    # ── Phase 2 ───────────────────────────────────────────────────────────────

    def _rank_candidates(
        self,
        candidates: list[ViralClip],
        transcript: Transcript,
    ) -> list[ViralClip]:
        """Send all deduplicated candidates to the LLM for final ranking.

        Args:
            candidates: All unique candidate clips from Phase 1.
            transcript: Source transcript (used for duration metadata).

        Returns:
            Final ranked list of ViralClip instances.
            Falls back to score-sorted candidates on LLM failure.
        """
        prompt = build_ranking_prompt(candidates, self._settings, transcript.duration)
        step_start = time.perf_counter()

        logger.info(
            "  Ranking %d candidates (~%d chars)...",
            len(candidates),
            len(prompt),
        )

        try:
            response = self._client.chat(
                model=self._settings.ollama_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": self._settings.ollama_temperature,
                    "num_ctx": self._settings.ollama_num_ctx,
                },
            )
        except Exception as exc:
            logger.warning(
                "  Phase 2 ranking failed (%s) — falling back to score sort.", exc
            )
            return sorted(candidates, key=lambda c: c.viral_score, reverse=True)

        elapsed = time.perf_counter() - step_start
        ranked = self._parse_clips(
            raw_json=response.message.content,
            max_duration=transcript.duration + 5,
        )

        logger.info(
            "  Phase 2 complete — %d clips selected in %.1fs.",
            len(ranked),
            elapsed,
        )

        # Fallback: if ranking returned nothing, use score-sorted candidates
        if not ranked:
            logger.warning("  Phase 2 returned no clips — using score-sorted fallback.")
            return sorted(candidates, key=lambda c: c.viral_score, reverse=True)

        return ranked

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _parse_clips(self, raw_json: str, max_duration: float) -> list[ViralClip]:
        """Parse and validate the LLM JSON response into ViralClip objects.

        Clips below ``min_clip_duration`` are automatically expanded around
        their detected center point rather than discarded. Clips above
        ``max_clip_duration`` or with invalid timestamps are still discarded.

        Args:
            raw_json: Raw JSON string from the LLM.
            max_duration: Upper bound for clip end_time validation (seconds).
                Typically ``video_duration + 5`` to give a small tolerance for
                clips that slightly overshoot chunk boundaries.

        Returns:
            List of valid ViralClip instances, sorted by viral_score descending.
        """
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON: %s", exc)
            return []

        raw_clips: list[dict] = data.get("clips", [])
        if not raw_clips:
            return []

        # Fix nested array structure: [[{...}], [{...}]] → [{...}, {...}]
        if raw_clips and isinstance(raw_clips[0], list):
            raw_clips = [item for sublist in raw_clips for item in sublist if isinstance(item, dict)]
            logger.warning("Fixed nested array structure in LLM response (%d clips recovered).", len(raw_clips))

        valid: list[ViralClip] = []
        discarded = 0

        for i, raw in enumerate(raw_clips):
            if not isinstance(raw, dict):
                discarded += 1
                continue

            # Normalize field name casing — model sometimes returns 'Justification' (capital J)
            raw = {k.lower(): v for k, v in raw.items()}

            # Provide safe defaults for fields the model occasionally omits
            raw.setdefault("justification", "Momento com potencial viral identificado pelo modelo.")
            raw.setdefault("viral_score", 7.0)

            # Reject clips where the model echoed the schema placeholder as the title
            title = str(raw.get("title", "")).strip()
            if not title or "título real" in title.lower() or "engaging" in title.lower():
                logger.warning("Clip %d has placeholder/empty title — discarding.", i + 1)
                discarded += 1
                continue

            try:
                clip = ViralClip(**raw)
            except (ValidationError, TypeError) as exc:
                logger.warning(
                    "Clip %d schema error — %s | raw: %s",
                    i + 1,
                    exc,
                    str(raw)[:200],
                )
                discarded += 1
                continue

            if clip.end_time > max_duration:
                logger.warning(
                    "Clip %d '%s' end_time %.1fs exceeds video duration %.1fs — discarding.",
                    i + 1,
                    clip.title[:60],
                    clip.end_time,
                    max_duration,
                )
                discarded += 1
                continue

            # Expand clips that are too short instead of discarding them.
            # The LLM reliably finds the right moment but often under-estimates
            # how long the segment should be. We expand symmetrically around
            # the detected center to meet the minimum duration requirement.
            if clip.duration < self._settings.min_clip_duration:
                original_duration = clip.duration
                # max_duration includes a +5s tolerance buffer; strip it for expansion
                video_end = max(0.0, max_duration - 5.0)
                clip = self._expand_clip(clip, video_end)
                logger.debug(
                    "Clip %d '%s' expanded %.1fs → %.1fs.",
                    i + 1,
                    clip.title[:60],
                    original_duration,
                    clip.duration,
                )

            # After expansion, the clip might still be too short (e.g. detected
            # near the very start or end of an extremely short video).
            if clip.duration < self._settings.min_clip_duration:
                logger.warning(
                    "Clip %d '%s' too short even after expansion (%.1fs < %ds) — discarding.",
                    i + 1,
                    clip.title[:60],
                    clip.duration,
                    self._settings.min_clip_duration,
                )
                discarded += 1
                continue

            if clip.duration > self._settings.max_clip_duration:
                logger.warning(
                    "Clip %d '%s' duration %.1fs exceeds max [%ds] — discarding.",
                    i + 1,
                    clip.title[:60],
                    clip.duration,
                    self._settings.max_clip_duration,
                )
                discarded += 1
                continue

            valid.append(clip)

        if discarded:
            logger.warning(
                "Discarded %d invalid/out-of-bounds clip(s) from LLM response.",
                discarded,
            )

        if not valid and raw_clips:
            logger.warning(
                "LLM returned %d clip(s) but ALL failed validation. "
                "Raw response sample: %.300s",
                len(raw_clips),
                json.dumps(raw_clips[:2]),
            )

        return sorted(valid, key=lambda c: c.viral_score, reverse=True)

    def _expand_clip(self, clip: ViralClip, video_end: float) -> ViralClip:
        """Expand a clip symmetrically to meet the minimum duration requirement.

        The original detected center ``(start + end) / 2`` is preserved as
        the anchor. Expansion is symmetric: half the extra time is added
        before, half after. If the result would overshoot a boundary (0 or
        ``video_end``), it is clamped and the slack is taken from the other
        side, so the output duration is always exactly ``min_clip_duration``
        when there is enough video around the clip.

        Args:
            clip: The clip to expand. Must have ``duration < min_clip_duration``.
            video_end: Last valid timestamp in seconds (video duration).

        Returns:
            A new ViralClip with updated ``start_time`` and ``end_time``.
        """
        target = float(self._settings.min_clip_duration)
        if clip.duration >= target:
            return clip

        center = (clip.start_time + clip.end_time) / 2.0
        half = target / 2.0

        new_start = max(0.0, center - half)
        new_end = new_start + target

        # If end overshoots the video, anchor to the end and pull start back
        if new_end > video_end:
            new_end = video_end
            new_start = max(0.0, new_end - target)

        return clip.model_copy(
            update={
                "start_time": round(new_start, 1),
                "end_time": round(new_end, 1),
            }
        )

    def _deduplicate(self, clips: list[ViralClip]) -> list[ViralClip]:
        """Remove overlapping clips, keeping the one with the higher viral_score.

        Uses a greedy algorithm: sort by score descending, then accept each
        clip only if it does not overlap more than 50% with an already-accepted
        clip.

        Args:
            clips: Raw list of candidates, possibly with overlapping timestamps.

        Returns:
            Deduplicated list of ViralClip instances.
        """
        sorted_clips = sorted(clips, key=lambda c: c.viral_score, reverse=True)
        accepted: list[ViralClip] = []

        for clip in sorted_clips:
            if not any(_overlap_ratio(clip, kept) > 0.5 for kept in accepted):
                accepted.append(clip)

        return accepted

    def _build_and_save_result(
        self,
        transcript: Transcript,
        clips: list[ViralClip],
    ) -> ClipDetectionResult:
        """Construct ClipDetectionResult, save it to disk and return it.

        Args:
            transcript: Source transcript for metadata.
            clips: Final selected viral clips.

        Returns:
            The saved ClipDetectionResult.
        """
        result = ClipDetectionResult(
            video_id=transcript.video_id,
            url=transcript.url,
            clips=clips,
        )
        self._save(result)
        return result

    def _save(self, result: ClipDetectionResult) -> Path:
        """Persist the ClipDetectionResult as JSON to storage/jobs/.

        Args:
            result: The result to persist.

        Returns:
            Path to the saved JSON file.
        """
        output_dir = self._settings.jobs_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{result.video_id}.json"
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

        logger.info("Results saved: %s", output_path)
        return output_path

    def _verify_ollama_connection(self) -> None:
        """Verify the Ollama server is reachable and the model is available.

        Raises:
            ConnectionError: If the Ollama server is not running.
            RuntimeError: If the configured model has not been pulled.
        """
        try:
            available = self._client.list()
            model_names = [m.model for m in available.models]
            normalized = [name.split(":")[0] for name in model_names]
            requested = self._settings.ollama_model

            if requested not in model_names and requested not in normalized:
                raise RuntimeError(
                    f"Model '{requested}' is not available in Ollama.\n"
                    f"Available: {model_names}\n"
                    f"Pull it with: ollama pull {requested}"
                )

            logger.debug(
                "Ollama connection verified. Model '%s' is available.", requested
            )

        except Exception as exc:
            if "connect" in str(exc).lower() or isinstance(exc, ConnectionError):
                raise ConnectionError(
                    f"Cannot connect to Ollama at '{self._settings.ollama_base_url}'.\n"
                    "Make sure Ollama is running: ollama serve"
                ) from exc
            raise


# ── Module-level helpers ──────────────────────────────────────────────────────

def _overlap_ratio(a: ViralClip, b: ViralClip) -> float:
    """Calculate the overlap between two clips as a fraction of the shorter one.

    Args:
        a: First clip.
        b: Second clip.

    Returns:
        Float in [0.0, 1.0]. 0.0 means no overlap, 1.0 means full overlap.
    """
    overlap_start = max(a.start_time, b.start_time)
    overlap_end = min(a.end_time, b.end_time)

    if overlap_end <= overlap_start:
        return 0.0

    overlap_duration = overlap_end - overlap_start
    shorter = min(a.duration, b.duration)

    if shorter == 0:
        return 0.0

    return overlap_duration / shorter
