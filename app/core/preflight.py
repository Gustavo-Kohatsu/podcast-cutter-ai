"""Pre-flight checks — tool and environment verification.

Runs before the pipeline starts and reports the status of every external
dependency.  Checks are classified as:

  OK      — tool found and functional.
  WARNING — tool missing or misconfigured, but the pipeline degrades
            gracefully (e.g. MediaPipe unavailable → falls back to centre crop).
  ERROR   — tool is required; the pipeline cannot proceed without it.

``PreflightChecker.run_all()`` returns ``True`` if no ERROR-level check
failed, meaning the pipeline may start.  If any ERROR is present, ``main.py``
exits immediately so the user can fix the problem before wasting time on
a long download or transcription.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal

import static_ffmpeg

from app.config.settings import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum free disk space before a WARNING is emitted (bytes)
_MIN_FREE_BYTES = 3 * 1024**3  # 3 GB

Status = Literal["ok", "warning", "error"]

_ICONS: dict[Status, str] = {
    "ok": "✓",
    "warning": "⚠",
    "error": "✗",
}


@dataclass
class CheckResult:
    """Result of a single pre-flight check.

    Attributes:
        name: Short human-readable tool/check name.
        status: ``ok`` | ``warning`` | ``error``.
        detail: One-line description of what was found or went wrong.
        fix: Optional actionable hint shown when status is not ``ok``.
    """

    name: str
    status: Status
    detail: str
    fix: str | None = None


class PreflightChecker:
    """Verifies all external dependencies before the pipeline starts.

    Args:
        settings: Application settings (needed for Ollama URL/model,
            cookies path, storage directory, etc.).

    Example:
        >>> checker = PreflightChecker(settings)
        >>> if not checker.run_all():
        ...     sys.exit(2)
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self) -> bool:
        """Run every check and log a formatted summary.

        Returns:
            ``True`` if the pipeline may proceed (no ERROR-level failures).
            ``False`` if at least one ERROR was found.
        """
        checks = [
            self._check_ollama(),
            self._check_ffmpeg(),
            self._check_ytdlp(),
            self._check_mediapipe(),
            self._check_cookies(),
            self._check_disk_space(),
        ]

        self._print_summary(checks)

        return not any(c.status == "error" for c in checks)

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_ollama(self) -> CheckResult:
        """Verify that Ollama is running and the configured model is available."""
        import ollama  # already a project dependency

        client = ollama.Client(host=self._settings.ollama_base_url)
        model = self._settings.ollama_model

        try:
            response = client.list()
        except Exception:
            return CheckResult(
                name="Ollama",
                status="error",
                detail=f"servidor não está respondendo em {self._settings.ollama_base_url}",
                fix="Inicie o Ollama com: ollama serve",
            )

        available = [m.model for m in response.models]

        # Normalize: Ollama sometimes appends ":latest"
        available_normalized = {m.split(":")[0] for m in available}
        model_base = model.split(":")[0]

        if model_base not in available_normalized:
            available_str = ", ".join(sorted(available_normalized)) or "(nenhum)"
            return CheckResult(
                name="Ollama",
                status="error",
                detail=f"modelo '{model}' não encontrado. Disponíveis: {available_str}",
                fix=f"Baixe o modelo com: ollama pull {model}",
            )

        return CheckResult(
            name="Ollama",
            status="ok",
            detail=f"rodando em {self._settings.ollama_base_url}  |  modelo={model}",
        )

    def _check_ffmpeg(self) -> CheckResult:
        """Verify that FFmpeg and FFprobe binaries are accessible."""
        try:
            static_ffmpeg.add_paths()
        except Exception as exc:
            return CheckResult(
                name="FFmpeg",
                status="error",
                detail=f"static_ffmpeg falhou ao configurar os binários: {exc}",
                fix="Execute: pip install --upgrade static-ffmpeg",
            )

        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")

        if not ffmpeg or not ffprobe:
            missing = "ffmpeg" if not ffmpeg else "ffprobe"
            return CheckResult(
                name="FFmpeg",
                status="error",
                detail=f"{missing} não encontrado no PATH após static_ffmpeg.add_paths()",
                fix="Execute: pip install --upgrade static-ffmpeg",
            )

        return CheckResult(
            name="FFmpeg",
            status="ok",
            detail=f"ok  ({ffmpeg})",
        )

    def _check_ytdlp(self) -> CheckResult:
        """Verify that yt-dlp is importable and report its version."""
        try:
            import yt_dlp

            version = getattr(yt_dlp, "version", None)
            version_str = getattr(version, "__version__", "versão desconhecida")
            return CheckResult(
                name="yt-dlp",
                status="ok",
                detail=f"ok  (v{version_str})",
            )
        except ImportError:
            return CheckResult(
                name="yt-dlp",
                status="error",
                detail="não instalado",
                fix="Execute: uv add yt-dlp",
            )

    def _check_mediapipe(self) -> CheckResult:
        """Verify that MediaPipe is importable (needed for smart crop)."""
        try:
            import mediapipe

            version = getattr(mediapipe, "__version__", "versão desconhecida")
            return CheckResult(
                name="MediaPipe",
                status="ok",
                detail=f"ok  (v{version})  |  smart crop ativo",
            )
        except ImportError:
            return CheckResult(
                name="MediaPipe",
                status="warning",
                detail="não instalado  |  smart crop desabilitado (fallback: crop central fixo)",
                fix="Execute: uv add mediapipe",
            )

    def _check_cookies(self) -> CheckResult:
        """Verify cookies configuration for yt-dlp authentication."""
        cookies_file = self._settings.ytdlp_cookies_file
        cookie_browser = self._settings.ytdlp_cookie_browser

        if cookies_file:
            from pathlib import Path

            path = Path(cookies_file)
            if not path.exists():
                return CheckResult(
                    name="cookies.txt",
                    status="warning",
                    detail=f"arquivo configurado mas não encontrado: {cookies_file}",
                    fix="Exporte os cookies do browser e salve no caminho configurado",
                )
            return CheckResult(
                name="cookies.txt",
                status="ok",
                detail=f"encontrado  ({path.name})",
            )

        if cookie_browser:
            return CheckResult(
                name="cookies.txt",
                status="warning",
                detail=(
                    f"extração via browser '{cookie_browser}' configurada  "
                    "(pode falhar no Windows com Chrome/Brave 127+)"
                ),
            )

        return CheckResult(
            name="cookies.txt",
            status="warning",
            detail="não configurado  |  downloads de vídeos privados podem falhar (HTTP 403)",
            fix="Configure YTDLP_COOKIES_FILE=cookies.txt no .env  (veja README)",
        )

    def _check_disk_space(self) -> CheckResult:
        """Warn if there is less than 3 GB of free disk space."""
        try:
            target = self._settings.storage_dir
            target.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(target)
            free_gb = usage.free / 1024**3

            if usage.free < _MIN_FREE_BYTES:
                return CheckResult(
                    name="Espaço livre",
                    status="warning",
                    detail=f"{free_gb:.1f} GB livres em {target}  (mínimo recomendado: 3 GB)",
                    fix="Libere espaço em disco antes de continuar",
                )

            return CheckResult(
                name="Espaço livre",
                status="ok",
                detail=f"{free_gb:.1f} GB livres em {target}",
            )
        except Exception as exc:
            return CheckResult(
                name="Espaço livre",
                status="warning",
                detail=f"não foi possível verificar espaço em disco: {exc}",
            )

    # ── Formatting ────────────────────────────────────────────────────────────

    def _print_summary(self, checks: list[CheckResult]) -> None:
        """Log the formatted pre-flight summary table."""
        sep = "═" * 62
        name_width = max(len(c.name) for c in checks) + 2

        logger.info(sep)
        logger.info("  Verificação pré-pipeline")
        logger.info(sep)

        for check in checks:
            icon = _ICONS[check.status]
            line = f"  {icon}  {check.name:<{name_width}} {check.detail}"

            if check.status == "ok":
                logger.info(line)
            elif check.status == "warning":
                logger.warning(line)
                if check.fix:
                    logger.warning("     → %s", check.fix)
            else:
                logger.error(line)
                if check.fix:
                    logger.error("     → %s", check.fix)

        errors = sum(1 for c in checks if c.status == "error")
        warnings = sum(1 for c in checks if c.status == "warning")

        logger.info(sep)
        if errors:
            logger.error(
                "  %d erro(s) encontrado(s) — pipeline abortado. Corrija os erros acima.",
                errors,
            )
        else:
            status_str = f"{warnings} aviso(s)" if warnings else "tudo ok"
            logger.info("  Pronto para iniciar  (%s)", status_str)
        logger.info(sep)
