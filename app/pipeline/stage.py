"""Stage ABC — the behavioural contract for all pipeline stages.

Every stage in the pipeline must implement this interface.  The orchestrator
works exclusively against this abstraction; it never imports concrete stage
classes or service classes.

Design rationale — Abstract Base Class vs. Protocol
----------------------------------------------------
Python's ``typing.Protocol`` provides structural (duck-typed) subtyping: any
class with a matching ``name`` property and ``run()`` method satisfies the
contract without declaring it.  This is convenient for tests and third-party
extensions.

``ABC`` with ``@abstractmethod`` provides **nominal** (declared) subtyping:
the interpreter raises ``TypeError`` at instantiation if a required method is
missing, giving a clear error *before* the pipeline ever runs a broken stage.

For a pipeline where a forgotten ``run()`` implementation would silently pass
context through unchanged and corrupt downstream data, the explicit enforcement
of ABC is the safer default.  Structural typing can always be added on top via
``Protocol`` if external plugin extensibility is needed in the future.

Usage example
-------------
::

    class MyStage(Stage):
        @property
        def name(self) -> str:
            return "My Custom Stage"

        def run(self, ctx: PipelineContext) -> None:
            # Read inputs
            transcript = ctx.require_transcript()
            # Do work …
            # Write output
            ctx.clip_result = my_service.process(transcript)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.pipeline.context import PipelineContext


class Stage(ABC):
    """Abstract base class that all pipeline stages must inherit from.

    A stage is the **unit of work** in the pipeline.  It encapsulates one
    cohesive responsibility, reads typed inputs from the shared context, and
    writes typed outputs back into it.

    Rules:
    - A stage MUST NOT call another stage directly.
    - A stage MUST NOT import from ``app.pipeline.orchestrator``.
    - A stage SHOULD raise ``ValueError`` when a required context field is
      missing, using the ``ctx.require_*`` guard helpers.
    - A stage MAY be tested in isolation by constructing a ``PipelineContext``
      and calling ``run(ctx)`` directly.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name used in log messages and progress display."""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:
        """Execute the stage.

        Reads required inputs from ``ctx`` and writes outputs back into it.
        Any unhandled exception propagates to the orchestrator, which logs it
        and aborts the pipeline.

        Args:
            ctx: The shared pipeline context.  Fields populated by earlier
                stages are guaranteed to be set; later stages' fields may be
                ``None``.

        Raises:
            ValueError: If a required predecessor field is not set in ``ctx``.
            Exception: Any other exception aborts the pipeline.
        """
