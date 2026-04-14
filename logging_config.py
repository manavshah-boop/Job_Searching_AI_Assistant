"""
logging_config.py — Loguru sink configuration for the job agent.

Call configure_logging(profile, debug) once at startup from any entry point
(main.py, dashboard.py, worker/run_pipeline.py). All other modules just do:

    from loguru import logger

and use logger directly — the sinks are wired here.
"""

import sys
from pathlib import Path

from loguru import logger

_PROJECT_ROOT   = Path(__file__).parent
_configured_for: str = ""   # module-level cache — prevents duplicate setup


def configure_logging(profile: str = "default", debug: bool = False) -> None:
    """
    Configure loguru with two sinks:
      - stderr  : INFO+ normally, DEBUG+ when debug=True
      - log file: DEBUG always, 10 MB rotation, 7-day retention, zip compression

    Idempotent: re-calling with the same (profile, debug) pair is a no-op.
    Re-calling with different args (e.g. profile switch in dashboard) reconfigures.
    """
    global _configured_for

    cache_key = f"{profile}:{debug}"
    if _configured_for == cache_key:
        return
    _configured_for = cache_key

    # Remove all existing sinks (including loguru's default stderr sink)
    logger.remove()

    # ── Console sink ──────────────────────────────────────────────────────────
    # Clean format — no timestamp so terminal output stays readable
    console_level = "DEBUG" if debug else "INFO"
    logger.add(
        sys.stderr,
        level=console_level,
        format="<level>{level:<8}</level> {message}",
        colorize=True,
    )

    # ── File sink ─────────────────────────────────────────────────────────────
    log_dir = _PROJECT_ROOT / "logs" / profile
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "agent.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} — {message}",
        encoding="utf-8",
    )

    logger.debug(f"Logging configured — profile={profile!r} debug={debug} log_dir={log_dir}")
