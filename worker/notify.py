"""
worker/notify.py - Failure notifier invoked by systemd OnFailure=.

Wired by deploy/systemd/jobagent-failure-notify@.service (one per profile).
The active alerting channel for this deployment is dashboard-banner-only —
the dashboard surfaces a staleness banner when .last_run grows old, and no
external notification path is configured. This script exists as a no-op
hook so a future channel (webhook, email, Pushover, etc.) can be wired in
without touching systemd.

Behavior:
- Reads the failed profile from --profile (matches the %i template).
- Reads the .last_run JSON for the profile to include error context, when
  available.
- If no notification channel env var is set, logs one info line and exits 0.
  This means non-deployed environments and the banner-only configuration
  cannot break the systemd OnFailure path.
- Any failure inside the notifier itself is swallowed (exit 0) — the only
  thing worse than a missed alert is the alert path itself crashing in a
  loop and masking the real failure.

Channel env vars (all optional):
- JOBAGENT_NOTIFY_WEBHOOK   - generic JSON POST URL (Slack-compatible)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_PROFILES_DIR = _PROJECT_ROOT / "profiles"
_STATUS_NAME = ".last_run"


def _read_last_run(profile: str) -> dict | None:
    path = _PROFILES_DIR / profile / _STATUS_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_message(profile: str, last_run: dict | None) -> str:
    if not last_run:
        return f"[jobagent] worker failed for profile '{profile}' (no .last_run found)"
    status = last_run.get("status", "unknown")
    error = last_run.get("error_message") or "(no error message)"
    finished = last_run.get("finished_at", "?")
    return (
        f"[jobagent] worker failed for profile '{profile}' "
        f"at {finished} (status={status}): {error}"
    )


def _post_webhook(url: str, message: str) -> None:
    import urllib.request

    data = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=10).read()


def main() -> int:
    parser = argparse.ArgumentParser(description="Worker failure notifier")
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    profile = args.profile
    message = _format_message(profile, _read_last_run(profile))

    webhook = os.environ.get("JOBAGENT_NOTIFY_WEBHOOK", "").strip()
    if not webhook:
        # Banner-only deployment, dev box, or unconfigured channel.
        # Print to stderr so the systemd journal still has a breadcrumb.
        print(message, file=sys.stderr)
        return 0

    try:
        _post_webhook(webhook, message)
    except Exception as exc:
        print(f"[jobagent.notify] webhook delivery failed: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
