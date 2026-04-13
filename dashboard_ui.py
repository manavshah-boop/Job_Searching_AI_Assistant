"""
dashboard_ui.py — Modern SaaS-style dashboard components.

Streamlit-based UI for displaying job search progress with real-time updates.
"""

import streamlit as st
from typing import Optional, List, Dict, Any
from progress_tracker import ProgressTracker, Stage, StageStatus, ActivityType


def render_progress_header(tracker: ProgressTracker) -> None:
    """Render the main progress bar and overall metrics."""
    st.markdown("---")

    # Main progress bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        progress_pct = tracker.overall_progress_pct
        st.metric("Progress", f"{int(progress_pct)}%")
        st.progress(progress_pct / 100, text=f"{int(progress_pct)}%")
    with col2:
        elapsed = tracker.elapsed_time
        mins, secs = divmod(int(elapsed.total_seconds()), 60)
        st.metric("Elapsed", f"{mins}m {secs}s")
    with col3:
        eta = tracker.eta
        if eta:
            mins, secs = divmod(int(eta.total_seconds()), 60)
            st.metric("ETA", f"~{mins}m {secs}s")
        else:
            st.metric("ETA", "—")

    st.markdown("---")


def render_pipeline_stages(tracker: ProgressTracker) -> None:
    """Render pipeline stages with status indicators."""
    st.subheader("📋 Pipeline Stages")

    stages_info = [
        (Stage.DISCOVERING, "Company Discovery"),
        (Stage.FETCHING, "Job Board Fetch"),
        (Stage.SCRAPING, "Job Scraping"),
        (Stage.SCORING, "Scoring & Filtering"),
        (Stage.FINALIZING, "Results Finalization"),
    ]

    for stage, label in stages_info:
        prog = tracker.stages[stage]
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            status_emoji = tracker.get_stage_status_emoji(stage)
            st.write(status_emoji)

        with col2:
            # Stage name
            st.markdown(f"**{label}**")
            if prog.status == StageStatus.RUNNING:
                st.caption(f"Duration: {int(prog.duration or 0)}s")
            elif prog.status == StageStatus.COMPLETE:
                st.caption(f"✓ Completed in {int(prog.duration or 0)}s")

        with col3:
            if prog.metrics:
                # Show brief metrics
                metric_strs = [f"{v}" for k, v in prog.metrics.items()]
                st.caption(" · ".join(metric_strs))

    st.markdown("---")


def render_source_progress(tracker: ProgressTracker) -> None:
    """Render per-source job scraping progress."""
    if not tracker.sources:
        return

    st.subheader("🏢 Job Board Progress")

    for source_name, source in tracker.sources.items():
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                status_icon = {
                    StageStatus.PENDING: "⏹️",
                    StageStatus.RUNNING: "⏳",
                    StageStatus.COMPLETE: "✅",
                    StageStatus.FAILED: "❌",
                }.get(source.status, "•")

                progress_pct = source.progress_pct
                st.markdown(f"{status_icon} **{source_name}**")
                st.progress(progress_pct / 100, text=f"{int(progress_pct)}%")

            with col2:
                st.metric(
                    "Companies",
                    f"{source.companies_processed}/{source.companies_total}",
                )

            with col3:
                st.metric("Jobs Found", source.jobs_new)

            with col4:
                if source.eta:
                    mins, secs = divmod(int(source.eta.total_seconds()), 60)
                    st.metric("ETA", f"~{mins}m {secs}s")
                else:
                    st.metric("ETA", "—")

        st.markdown("---")


def render_activity_feed(tracker: ProgressTracker, limit: int = 15) -> None:
    """Render the live activity feed."""
    st.subheader("📝 Activity Feed")

    activities = tracker.get_recent_activities(limit)

    if not activities:
        st.info("Waiting for activity…")
        return

    for activity in reversed(activities):
        # Color code by type
        if activity.type == ActivityType.ERROR:
            st.error(f"**{activity.time_str()}** — {activity.message}")
        elif activity.type == ActivityType.WARNING:
            st.warning(f"**{activity.time_str()}** — {activity.message}")
        elif activity.type == ActivityType.STAGE_UPDATE:
            st.info(f"**{activity.time_str()}** — {activity.message}")
        elif activity.type == ActivityType.METRIC_UPDATE:
            st.success(f"**{activity.time_str()}** — {activity.message}")
        else:
            st.write(f"**{activity.time_str()}** — {activity.message}")


def render_summary(tracker: ProgressTracker) -> None:
    """Render final summary when complete."""
    st.markdown("---")
    st.subheader("✅ Search Complete")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Jobs Found", tracker.total_jobs_new)
    with col2:
        st.metric("Total Time", f"{int(tracker.elapsed_time.total_seconds())}s")
    with col3:
        jobs_per_min = (
            (tracker.total_jobs_new / tracker.elapsed_time.total_seconds())
            * 60
            if tracker.elapsed_time.total_seconds() > 0
            else 0
        )
        st.metric("Rate", f"{jobs_per_min:.1f} jobs/min")
    with col4:
        st.metric("Sources", len(tracker.sources))


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
