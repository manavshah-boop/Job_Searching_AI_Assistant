import dashboard
from pathlib import Path
import shutil
import app
import ui_theme


def test_lines_to_list_strips_blank_lines_and_whitespace():
    assert dashboard._lines_to_list("  Python  \n\n AWS\n   \nLLM systems  ") == [
        "Python",
        "AWS",
        "LLM systems",
    ]


def test_build_jobs_table_frame_returns_expected_columns_and_values():
    frame = dashboard.build_jobs_table_frame(
        [
            {
                "id": "job-1",
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "",
                "source_label": "Greenhouse",
                "status_label": "Applied",
                "score_state": "Scored",
                "fit_score": 88,
                "ats_score": 76,
                "one_liner": "Strong match for Python backend work.",
                "url": "https://example.com/job-1",
            }
        ]
    )

    assert list(frame.columns) == [
        "id",
        "Title",
        "Company",
        "Location",
        "Source",
        "Job status",
        "Score state",
        "Fit",
        "ATS",
        "Summary",
        "Posting",
    ]
    assert frame.iloc[0].to_dict() == {
        "id": "job-1",
        "Title": "Backend Engineer",
        "Company": "Acme",
        "Location": "Location not listed",
        "Source": "Greenhouse",
        "Job status": "Applied",
        "Score state": "Scored",
        "Fit": 88,
        "ATS": 76,
        "Summary": "Strong match for Python backend work.",
        "Posting": "https://example.com/job-1",
    }
    assert str(frame["Fit"].dtype) == "Int64"
    assert str(frame["ATS"].dtype) == "Int64"


def test_summarize_run_errors_collapses_long_lists():
    summary = dashboard.summarize_run_errors(
        [
            "This is a very long error message that should be truncated for table display readability.",
            "Second error",
            "Third error",
        ],
        preview_chars=42,
    )

    assert summary == "This is a very long error message that sh… (+2 more)"


def test_build_jobs_filter_chips_formats_active_state():
    chips = dashboard.build_jobs_filter_chips(
        selected_sources=["Greenhouse", "Lever"],
        source_options=["Greenhouse", "Lever", "HN Who's Hiring"],
        selected_statuses=["Applied"],
        status_options=["Applied", "New", "Skipped"],
        selected_score_states=["Scored", "Pending", "Needs retry"],
        score_state_options=["Scored", "Pending", "Needs retry", "Failed"],
        min_fit=70,
        search="python backend",
        include_full_text=True,
    )

    assert chips == [
        "Sources: Greenhouse, Lever",
        "Status: Applied",
        "Scoring: Scored, Pending +1",
        "Min fit: 70+",
        "Search: python backend",
        "Full text search: On",
    ]


def test_resolve_job_table_columns_preserves_base_order():
    columns = dashboard.resolve_job_table_columns(["Summary", "ATS", "Posting"])

    assert columns == [
        "Title",
        "Company",
        "Job status",
        "Score state",
        "Fit",
        "ATS",
        "Summary",
        "Posting",
    ]


def test_effective_config_summary_surfaces_core_settings():
    config = {
        "llm": {"provider": "groq"},
        "scoring": {"min_display_score": 65},
        "preferences": {
            "titles": ["Backend Engineer", "AI Engineer"],
            "desired_skills": ["Python", "AWS", "LLM"],
            "location": {"remote_ok": True, "preferred_locations": ["Remote", "San Francisco, CA"]},
            "compensation": {"min_salary": 150000},
        },
    }
    raw_config = {
        "sources": {
            "greenhouse": {"enabled": True},
            "lever": {"enabled": False},
            "hn": {"enabled": True},
        }
    }

    summary = dashboard.effective_config_summary(config, raw_config)

    assert "Provider: Groq" in summary
    assert "Minimum display score: 65" in summary
    assert "Remote OK" in summary
    assert "Preferred locations: Remote, San Francisco, CA" in summary
    assert "Target titles: 2" in summary
    assert "Desired skills: 3" in summary
    assert "Sources enabled: Greenhouse, HN Who's Hiring" in summary
    assert "Minimum salary: $150,000" in summary


def test_get_theme_bootstrap_state_reports_missing_theme_from_cwd(monkeypatch):
    root = (Path.cwd() / ".tmp_streamlit_test" / "theme_debug_missing").resolve()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    monkeypatch.chdir(root)

    state = dashboard.get_theme_bootstrap_state()

    assert state["entrypoint"] == "dashboard.py"
    assert state["cwd"] == str(root)
    assert state["theme_config_path"] == root / ".streamlit" / "config.toml"
    assert state["theme_config_exists"] is False


def test_get_theme_bootstrap_state_reports_theme_when_present(monkeypatch):
    root = (Path.cwd() / ".tmp_streamlit_test" / "theme_debug_present").resolve()
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    theme_path = root / ".streamlit" / "config.toml"
    theme_path.parent.mkdir(parents=True)
    theme_path.write_text("[theme]\nbase='light'\n", encoding="utf-8")
    monkeypatch.chdir(root)

    state = dashboard.get_theme_bootstrap_state()

    assert state["theme_config_exists"] is True
    assert state["theme_config_path"] == theme_path


def test_app_entrypoint_delegates_to_dashboard(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "dashboard_main", lambda: calls.append("dashboard"))

    app.main()

    assert calls == ["dashboard"]


def test_inject_global_css_emits_styles_on_every_call(monkeypatch):
    calls = []
    monkeypatch.setattr(ui_theme.st, "markdown", lambda content, unsafe_allow_html=False: calls.append((content, unsafe_allow_html)))

    ui_theme.inject_global_css()
    ui_theme.inject_global_css()

    assert len(calls) == 2
    assert all(unsafe_allow_html is True for _, unsafe_allow_html in calls)
    assert all("<style>" in content for content, _ in calls)
