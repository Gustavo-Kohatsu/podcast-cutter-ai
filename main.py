"""Entry point for the Podcast Cutter AI pipeline.

Usage:
    python main.py "<youtube_url>"

Examples:
    python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python main.py "https://youtu.be/dQw4w9WgXcQ"
    python main.py "https://www.youtube.com/live/dQw4w9WgXcQ"

Exit codes:
    0 — Pipeline completed successfully.
    1 — Invalid arguments or URL.
    2 — Pipeline failed (download, transcription, or LLM error).
"""

import sys

from app.config.settings import Settings
from app.core.pipeline import Pipeline
from app.utils.logger import get_logger, setup_logging
from app.utils.validators import validate_youtube_url


def main() -> None:
    """Parse CLI arguments and run the pipeline."""
    if len(sys.argv) != 2:
        print(
            "Usage: python main.py <youtube_url>\n"
            "\nExamples:\n"
            '  python main.py "https://www.youtube.com/watch?v=VIDEO_ID"\n'
            '  python main.py "https://youtu.be/VIDEO_ID"',
            file=sys.stderr,
        )
        sys.exit(1)

    url = sys.argv[1].strip()

    # Bootstrap settings before anything else (needed for logging config)
    settings = Settings()
    setup_logging(settings)

    logger = get_logger(__name__)

    # Validate URL before starting any I/O-intensive work
    try:
        validate_youtube_url(url)
    except ValueError as exc:
        logger.error("Invalid YouTube URL: %s", exc)
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = Pipeline(settings)
        result_path = pipeline.run(url)

        print(f"\nDone! Results saved to: {result_path}")

    except ConnectionError as exc:
        logger.error("Ollama connection error: %s", exc)
        sys.exit(2)

    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        sys.exit(2)

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(2)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(0)

    except Exception as exc:
        logger.critical("Unexpected pipeline error: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
