"""
dashboard_ui.py — Modern SaaS-style dashboard components.

Streamlit-based UI for displaying job search progress with real-time updates.
"""

import streamlit as st
from ui_shell import badge, callout, empty_state, panel, stat_row
from progress_tracker import ProgressTracker, Stage, StageStatus, ActivityType


def render_progress_header(tracker: ProgressTracker) -> None:
    """Render the main progress bar and overall metrics."""
    progress_pct = tracker.overall_progress_pct
    elapsed = tracker.elapsed_time
    mins, secs = divmod(int(elapsed.total_seconds()), 60)
    eta = tracker.eta
    eta_label = "—"
    if eta:
        eta_mins, eta_secs = divmod(int(eta.total_seconds()), 60)
        eta_label = f"~{eta_mins}m {eta_secs}s"

    stat_row(
        [
            ("Progress", f"{int(progress_pct)}%"),
            ("Elapsed", f"{mins}m {secs}s"),
            ("ETA", eta_label),
        ]
    )
    st.progress(progress_pct / 100, text=f"{int(progress_pct)}% complete")


def render_pipeline_stages(tracker: ProgressTracker) -> None:
    """Render pipeline stages with status indicators."""
    stages_info = [
        (Stage.DISCOVERING, "Company Discovery"),
        (Stage.FETCHING, "Job Board Fetch"),
        (Stage.SCRAPING, "Job Scraping"),
        (Stage.SCORING, "Scoring & Filtering"),
        (Stage.EMBEDDING, "Semantic Embeddings"),
        (Stage.FINALIZING, "Results Finalization"),
    ]

    with panel("Pipeline stages", subtitle="Clear stage boundaries help explain where time is being spent"):
        for stage, label in stages_info:
            prog = tracker.stages[stage]
            col1, col2, col3 = st.columns([1, 3, 1])

            with col1:
                st.markdown(
                    badge(
                        tracker.get_stage_status_emoji(stage),
                        "success" if prog.status == StageStatus.COMPLETE else "info" if prog.status == StageStatus.RUNNING else "danger" if prog.status == StageStatus.FAILED else "neutral",
                    ),
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(f"**{label}**")
                if prog.status == StageStatus.RUNNING:
                    st.caption(f"Running for {int(prog.duration or 0)}s")
                elif prog.status == StageStatus.COMPLETE:
                    st.caption(f"Completed in {int(prog.duration or 0)}s")
                elif prog.status == StageStatus.FAILED:
                    st.caption("Failed before completion")

            with col3:
                if prog.metrics:
                    metric_strs = [f"{k}: {v}" for k, v in prog.metrics.items()]
                    st.caption(" · ".join(metric_strs))


def render_source_progress(tracker: ProgressTracker) -> None:
    """Render per-source job scraping progress."""
    if not tracker.sources:
        return

    with panel("Source progress", subtitle="Per-source progress builds trust while the pipeline is running"):
        for source_name, source in tracker.sources.items():
            col1, col2, col3 = st.columns([2.1, 1, 1], gap="medium")
            with col1:
                st.write(f"**{source_name}**")
                st.progress(source.progress_pct / 100 if source.companies_total else 0.0, text=f"{int(source.progress_pct)}%")
            with col2:
                st.caption("Coverage")
                st.write(f"{source.companies_processed}/{source.companies_total or 0}")
            with col3:
                st.caption("New jobs")
                st.write(str(source.jobs_new))


def render_activity_feed(tracker: ProgressTracker, limit: int = 15) -> None:
    """Render the live activity feed."""
    activities = tracker.get_recent_activities(limit)

    with panel("Activity feed", subtitle="Human-readable updates make the run easier to follow"):
        if not activities:
            empty_state("Waiting for activity", "Progress messages will appear here as the run advances.")
            return

        for activity in reversed(activities):
            if activity.type == ActivityType.ERROR:
                callout("error", activity.time_str(), activity.message)
            elif activity.type == ActivityType.WARNING:
                callout("warning", activity.time_str(), activity.message)
            elif activity.type == ActivityType.METRIC_UPDATE:
                callout("success", activity.time_str(), activity.message)
            else:
                st.write(f"**{activity.time_str()}** — {activity.message}")


def render_summary(tracker: ProgressTracker) -> None:
    """Render final summary when complete."""
    with panel("Run summary", subtitle="Compact wrap-up after the pipeline finishes"):
        jobs_per_min = (
            (tracker.total_jobs_new / tracker.elapsed_time.total_seconds()) * 60
            if tracker.elapsed_time.total_seconds() > 0
            else 0
        )
        stat_row(
            [
                ("Total jobs found", tracker.total_jobs_new),
                ("Total time", f"{int(tracker.elapsed_time.total_seconds())}s"),
                ("Rate", f"{jobs_per_min:.1f} jobs/min"),
                ("Sources", len(tracker.sources)),
            ]
        )


def render_debug_logs(tracker: ProgressTracker) -> None:
    """Render collapsible debug logs section."""
    with st.expander("🔧 Debug Logs (for troubleshooting)"):
        st.caption(f"Total activities logged: {len(tracker.activities)}")

        # Show raw activity log
        for activity in tracker.activities:
            log_entry = f"[{activity.time_str()}] {activity.type.value}: {activity.message}"
            if activity.details:
                log_entry += f" | {activity.details}"

            if activity.type == ActivityType.ERROR:
                st.write(f":red[{log_entry}]")
            elif activity.type == ActivityType.WARNING:
                st.write(f":orange[{log_entry}]")
            else:
                st.write(log_entry)
