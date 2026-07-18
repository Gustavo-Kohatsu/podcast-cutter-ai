"""Backward-compatibility shim for the old Pipeline class.

The orchestration logic has been moved to ``app.pipeline.orchestrator``.
This module re-exports ``Pipeline`` under the same name so that any
existing imports (``from app.core.pipeline import Pipeline``) continue to
work without modification.

New code should import directly from the new location:

    from app.pipeline.orchestrator import PipelineOrchestrator, create_default_pipeline
"""

from pathlib import Path

from app.config.settings import Settings
from app.pipeline.orchestrator import PipelineOrchestrator, create_default_pipeline


class Pipeline:
    """Thin wrapper around PipelineOrchestrator for backward compatibility.

    Accepts the same ``Pipeline(settings)`` constructor call as before and
    delegates directly to the new orchestrator.  All behaviour is identical.

    Args:
        settings: Application settings shared across all stages.

    Example:
        >>> pipeline = Pipeline(settings)
        >>> result_path = pipeline.run("https://youtu.be/...")
    """

    def __init__(self, settings: Settings) -> None:
        self._orchestrator = create_default_pipeline(settings)

    def run(self, url: str) -> Path:
        """Run the full pipeline and return the result-file path."""
        return self._orchestrator.run(url)


__all__ = ["Pipeline", "PipelineOrchestrator", "create_default_pipeline"]
