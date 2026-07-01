"""YouTube URL validation and video ID extraction.

Supports all common YouTube URL formats:
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID
- https://www.youtube.com/live/VIDEO_ID
- https://www.youtube.com/shorts/VIDEO_ID
- https://m.youtube.com/watch?v=VIDEO_ID
"""

import re
from urllib.parse import parse_qs, urlparse

_YOUTUBE_DOMAINS = frozenset({
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "youtu.be",
    "www.youtu.be",
})

_VIDEO_ID_PATTERN = re.compile(
    r"(?:v=|youtu\.be/|/live/|/shorts/|/embed/)([a-zA-Z0-9_-]{11})"
)


def validate_youtube_url(url: str) -> None:
    """Validate that the given string is a recognizable YouTube URL.

    Performs two checks:
    1. The URL's domain belongs to the set of known YouTube domains.
    2. A valid 11-character video ID can be extracted from it.

    Args:
        url: The URL string to validate.

    Raises:
        ValueError: If the URL is not a valid YouTube URL or no video ID
            could be extracted.

    Example:
        >>> validate_youtube_url("https://youtu.be/dQw4w9WgXcQ")  # OK
        >>> validate_youtube_url("https://vimeo.com/123")  # raises ValueError
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")

    parsed = urlparse(url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL must use http or https scheme, got: '{parsed.scheme}'"
        )

    if parsed.netloc not in _YOUTUBE_DOMAINS:
        raise ValueError(
            f"Not a YouTube URL. Domain '{parsed.netloc}' is not recognized.\n"
            f"Expected one of: {sorted(_YOUTUBE_DOMAINS)}"
        )

    # Attempt to extract video ID to catch malformed YouTube URLs early
    extract_video_id(url)


def extract_video_id(url: str) -> str:
    """Extract the 11-character video ID from a YouTube URL.

    Tries multiple strategies in order:
    1. Query string parameter ``?v=VIDEO_ID``
    2. Path segments for youtu.be, /live/, /shorts/, /embed/
    3. Regex fallback for non-standard formats

    Args:
        url: A YouTube URL string.

    Returns:
        The 11-character video ID string.

    Raises:
        ValueError: If no valid video ID could be extracted.

    Example:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    parsed = urlparse(url.strip())

    # Strategy 1: ?v= query parameter (standard watch URLs)
    query_params = parse_qs(parsed.query)
    if "v" in query_params:
        video_id = query_params["v"][0]
        if _is_valid_video_id(video_id):
            return video_id

    # Strategy 2: Path-based formats (youtu.be, /live/, /shorts/, /embed/)
    path = parsed.path.lstrip("/")
    path_segments = path.split("/")

    if path_segments:
        candidate = path_segments[-1]
        # Remove any query string that may have been left in the path
        candidate = candidate.split("?")[0]
        if _is_valid_video_id(candidate):
            return candidate

    # Strategy 3: Regex fallback for non-standard or shortened URLs
    match = _VIDEO_ID_PATTERN.search(url)
    if match:
        video_id = match.group(1)
        if _is_valid_video_id(video_id):
            return video_id

    raise ValueError(
        f"Could not extract a valid YouTube video ID from URL: '{url}'\n"
        "Make sure the URL points to a specific video, live, or short."
    )


def _is_valid_video_id(value: str) -> bool:
    """Check if a string matches the YouTube video ID format (11 alphanumeric chars)."""
    return bool(value) and len(value) == 11 and re.match(r"^[a-zA-Z0-9_-]{11}$", value)
