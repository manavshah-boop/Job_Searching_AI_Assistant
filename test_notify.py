"""Tests for worker/notify.py — the systemd OnFailure hook.

The notifier must exit 0 in unconfigured environments (dev boxes, the
banner-only deployment, missing .last_run files) so that systemd OnFailure
invocations cannot themselves crash and mask the real worker failure.
"""

from worker import notify


def test_notify_exits_zero_when_webhook_env_missing(monkeypatch, capsys):
    monkeypatch.delenv("JOBAGENT_NOTIFY_WEBHOOK", raising=False)
    monkeypatch.setattr("sys.argv", ["notify.py", "--profile", "manav"])
    monkeypatch.setattr(notify, "_read_last_run", lambda profile: None)

    rc = notify.main()
    assert rc == 0
    err = capsys.readouterr().err
    assert "manav" in err
    assert "no .last_run" in err


def test_notify_exits_zero_when_webhook_env_blank(monkeypatch):
    monkeypatch.setenv("JOBAGENT_NOTIFY_WEBHOOK", "   ")
    monkeypatch.setattr("sys.argv", ["notify.py", "--profile", "sister"])
    monkeypatch.setattr(notify, "_read_last_run", lambda profile: None)

    posted = []
    monkeypatch.setattr(notify, "_post_webhook", lambda url, msg: posted.append((url, msg)))

    rc = notify.main()
    assert rc == 0
    assert posted == []


def test_notify_includes_error_message_when_last_run_present(monkeypatch, capsys):
    monkeypatch.delenv("JOBAGENT_NOTIFY_WEBHOOK", raising=False)
    monkeypatch.setattr("sys.argv", ["notify.py", "--profile", "manav"])
    monkeypatch.setattr(
        notify,
        "_read_last_run",
        lambda profile: {
            "status": "failed",
            "finished_at": "2026-05-07T04:01:23+00:00",
            "error_message": "scraper raised ConnectionError",
        },
    )

    rc = notify.main()
    assert rc == 0
    err = capsys.readouterr().err
    assert "scraper raised ConnectionError" in err
    assert "manav" in err


def test_notify_swallows_webhook_failures(monkeypatch):
    monkeypatch.setenv("JOBAGENT_NOTIFY_WEBHOOK", "https://example.invalid/hook")
    monkeypatch.setattr("sys.argv", ["notify.py", "--profile", "manav"])
    monkeypatch.setattr(notify, "_read_last_run", lambda profile: {"status": "failed"})

    def boom(url, msg):
        raise RuntimeError("network down")

    monkeypatch.setattr(notify, "_post_webhook", boom)

    assert notify.main() == 0
