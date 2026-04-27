"""
progress_tracker.py — Manages job search progress tracking and events.

Converts pipeline activity into user-friendly events for real-time dashboard display.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta


class Stage(Enum):
    """Pipeline stages."""
    DISCOVERING = "🔍 Discovering companies"
    FETCHING = "🏢 Fetching job boards"
    SCRAPING = "📥 Scraping jobs"
    SCORING = "🧠 Scoring and filtering jobs"
    EMBEDDING = "🧩 Generating job embeddings"
    FINALIZING = "📊 Finalizing and saving results"


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ActivityType(Enum):
    """Types of user-visible activities."""
    STAGE_UPDATE = "stage_update"
    METRIC_UPDATE = "metric_update"
    ACTIVITY_LOG = "activity_log"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class StageProgress:
    """Progress info for a single pipeline stage."""
    stage: Stage
    status: StageStatus = StageStatus.PENDING
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Return duration in seconds, or None if not finished."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    def start(self):
        """Mark stage as running."""
        self.status = StageStatus.RUNNING
        self.started_at = time.time()

    def complete(self):
        """Mark stage as complete."""
        self.status = StageStatus.COMPLETE
        self.completed_at = time.time()

    def fail(self):
        """Mark stage as failed."""
        self.status = StageStatus.FAILED
        self.completed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "status": self.status.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StageProgress":
        stage = next(s for s in Stage if s.value == data["stage"])
        return cls(
            stage=stage,
            status=StageStatus(data["status"]),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            metrics=data.get("metrics", {}),
        )


@dataclass
class Activity:
    """Single activity event for the user-visible feed."""
    type: ActivityType
    timestamp: float = field(default_factory=time.time)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def time_str(self) -> str:
        """Return human-readable time."""
        dt = datetime.fromtimestamp(self.timestamp)
        return dt.strftime("%H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "timestamp": self.timestamp,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Activity":
        return cls(
            type=ActivityType(data["type"]),
            timestamp=data["timestamp"],
            message=data["message"],
            details=data.get("details", {}),
        )


@dataclass
class SourceProgress:
    """Progress tracking for a single source (Greenhouse, Lever, etc.)."""
    name: str  # "Greenhouse", "Lever", etc.
    status: StageStatus = StageStatus.PENDING
    companies_total: int = 0
    companies_processed: int = 0
    jobs_found: int = 0
    jobs_new: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    @property
    def progress_pct(self) -> float:
        """Return percentage progress (0-100)."""
        if self.companies_total == 0:
            return 0.0
        return (self.companies_processed / self.companies_total) * 100

    @property
    def duration(self) -> Optional[float]:
        """Return duration in seconds."""
        if self.started_at is None:
            return None
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def eta(self) -> Optional[timedelta]:
        """Estimate time to completion based on rate."""
        if not self.duration or self.companies_processed == 0:
            return None
        rate_per_company = self.duration / self.companies_processed
        remaining = self.companies_total - self.companies_processed
        remaining_seconds = rate_per_company * remaining
        return timedelta(seconds=remaining_seconds)

    def start(self):
        """Mark source as starting."""
        self.status = StageStatus.RUNNING
        self.started_at = time.time()

    def complete(self):
        """Mark source as complete."""
        self.status = StageStatus.COMPLETE
        self.completed_at = time.time()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status.value,
            "companies_total": self.companies_total,
            "companies_processed": self.companies_processed,
            "jobs_found": self.jobs_found,
            "jobs_new": self.jobs_new,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceProgress":
        obj = cls(name=data["name"])
        obj.status = StageStatus(data["status"])
        obj.companies_total = data.get("companies_total", 0)
        obj.companies_processed = data.get("companies_processed", 0)
        obj.jobs_found = data.get("jobs_found", 0)
        obj.jobs_new = data.get("jobs_new", 0)
        obj.started_at = data.get("started_at")
        obj.completed_at = data.get("completed_at")
        return obj


class ProgressTracker:
    """Central progress tracking for the entire job search pipeline."""

    def __init__(self):
        self.stages: Dict[Stage, StageProgress] = {
            stage: StageProgress(stage=stage) for stage in Stage
        }
        self.sources: Dict[str, SourceProgress] = {}
        self.activities: List[Activity] = []
        self.start_time = time.time()
        self.total_jobs_processed = 0
        self.total_jobs_new = 0

    def log_activity(
        self,
        message: str,
        activity_type: ActivityType = ActivityType.ACTIVITY_LOG,
        details: Optional[Dict[str, Any]] = None,
    ) -> Activity:
        """Log a user-visible activity."""
        activity = Activity(
            type=activity_type,
            message=message,
            details=details or {},
        )
        self.activities.append(activity)
        return activity

    def start_stage(self, stage: Stage) -> StageProgress:
        """Mark a stage as started."""
        prog = self.stages[stage]
        prog.start()
        self.log_activity(
            f"Starting: {stage.value}",
            ActivityType.STAGE_UPDATE,
            {"stage": stage.name},
        )
        return prog

    def complete_stage(self, stage: Stage):
        """Mark a stage as complete."""
        prog = self.stages[stage]
        prog.complete()
        self.log_activity(
            f"Completed: {stage.value}",
            ActivityType.STAGE_UPDATE,
            {"stage": stage.name},
        )

    def set_stage_metrics(self, stage: Stage, **metrics: Any):
        """Attach lightweight stage metrics for UI display."""
        self.stages[stage].metrics.update(metrics)

    def fail_stage(self, stage: Stage, error: str):
        """Mark a stage as failed."""
        prog = self.stages[stage]
        prog.fail()
        self.log_activity(
            f"Failed: {stage.value} — {error}",
            ActivityType.ERROR,
            {"stage": stage.name, "error": error},
        )

    def register_source(self, name: str, companies_total: int) -> SourceProgress:
        """Register a new source to track."""
        source = SourceProgress(
            name=name,
            companies_total=companies_total,
        )
        self.sources[name] = source
        return source

    def start_source(self, name: str):
        """Mark a source as starting."""
        if name not in self.sources:
            self.register_source(name, 0)
        self.sources[name].start()
        self.log_activity(
            f"Processing {name}…",
            ActivityType.ACTIVITY_LOG,
        )

    def update_source(
        self,
        name: str,
        companies_processed: Optional[int] = None,
        jobs_found: Optional[int] = None,
    ):
        """Update source progress."""
        if name not in self.sources:
            self.register_source(name, 0)
        source = self.sources[name]
        if companies_processed is not None:
            source.companies_processed = companies_processed
        if jobs_found is not None:
            source.jobs_found = jobs_found
            source.jobs_new = jobs_found

    def complete_source(self, name: str, jobs_found: int = 0):
        """Mark a source as complete."""
        if name not in self.sources:
            self.register_source(name, 0)
        source = self.sources[name]
        source.jobs_new = jobs_found
        source.complete()
        self.total_jobs_new += jobs_found
        self.log_activity(
            f"Found {jobs_found} new jobs at {name}",
            ActivityType.METRIC_UPDATE,
            {"source": name, "jobs": jobs_found},
        )

    def add_activity_log(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Add a human-readable activity to the feed."""
        self.log_activity(message, ActivityType.ACTIVITY_LOG, details)

    def add_warning(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a warning."""
        self.log_activity(message, ActivityType.WARNING, details)

    def add_error(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Log an error."""
        self.log_activity(message, ActivityType.ERROR, details)

    @property
    def overall_progress_pct(self) -> float:
        """Calculate overall progress 0-100%."""
        # Equal weight to each stage
        stage_pct = sum(
            1.0 if prog.status == StageStatus.COMPLETE else (0.5 if prog.status == StageStatus.RUNNING else 0.0)
            for prog in self.stages.values()
        ) / len(self.stages)

        # Source-level progress
        if self.sources:
            source_pct = sum(
                source.progress_pct for source in self.sources.values()
            ) / len(self.sources) / 100.0
        else:
            source_pct = 0.0

        # 60% from stages, 40% from sources
        return (stage_pct * 60 + source_pct * 40)

    @property
    def elapsed_time(self) -> timedelta:
        """Return elapsed time since start."""
        return timedelta(seconds=time.time() - self.start_time)

    @property
    def eta(self) -> Optional[timedelta]:
        """Estimate time to completion."""
        # Average ETA from all sources
        source_etas = [s.eta for s in self.sources.values() if s.eta]
        if not source_etas:
            return None
        avg_eta_seconds = sum(eta.total_seconds() for eta in source_etas) / len(
            source_etas
        )
        return timedelta(seconds=avg_eta_seconds)

    def get_recent_activities(self, limit: int = 10) -> List[Activity]:
        """Get most recent N activities."""
        return self.activities[-limit:]

    def get_stage_status_emoji(self, stage: Stage) -> str:
        """Get emoji for stage status."""
        prog = self.stages[stage]
        if prog.status == StageStatus.COMPLETE:
            return "✅"
        elif prog.status == StageStatus.RUNNING:
            return "⏳"
        elif prog.status == StageStatus.FAILED:
            return "❌"
        else:
            return "⏹️"

    def to_dict(self) -> dict:
        return {
            "stages": {stage.value: prog.to_dict() for stage, prog in self.stages.items()},
            "sources": {name: src.to_dict() for name, src in self.sources.items()},
            "activities": [a.to_dict() for a in self.activities],
            "start_time": self.start_time,
            "total_jobs_processed": self.total_jobs_processed,
            "total_jobs_new": self.total_jobs_new,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProgressTracker":
        tracker = cls()
        stages_data = data.get("stages", {})
        for stage in Stage:
            if stage.value in stages_data:
                tracker.stages[stage] = StageProgress.from_dict(stages_data[stage.value])
        tracker.sources = {
            name: SourceProgress.from_dict(src)
            for name, src in data.get("sources", {}).items()
        }
        tracker.activities = [Activity.from_dict(a) for a in data.get("activities", [])]
        tracker.start_time = data.get("start_time", time.time())
        tracker.total_jobs_processed = data.get("total_jobs_processed", 0)
        tracker.total_jobs_new = data.get("total_jobs_new", 0)
        return tracker
