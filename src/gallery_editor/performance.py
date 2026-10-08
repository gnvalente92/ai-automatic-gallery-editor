"""Durable wall-clock profile for gallery runs and local model requests."""

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter

from .storage import write_json


class PerformanceProfile:
    def __init__(self, root, operation):
        self.root = root
        self.started = perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.operation = operation
        self.status = "running"
        self.stages = []
        self.model_calls = []
        self.photos = []
        self.operations = []
        self._stage_name = None
        self._stage_started = None
        self._photo_started = None
        self._photo_name = None

    def stage(self, name):
        now = perf_counter()
        if self._stage_name and self._stage_name != name:
            self.stages.append({"name": self._stage_name, "seconds": round(now - self._stage_started, 3)})
        if self._stage_name != name:
            self._stage_name, self._stage_started = name, now
            self.checkpoint()

    def photo_start(self, filename):
        self._photo_name, self._photo_started = filename, perf_counter()

    def photo_end(self):
        if self._photo_started is not None:
            self.photos.append(
                {"file": self._photo_name, "seconds": round(perf_counter() - self._photo_started, 3)}
            )
            self._photo_name = self._photo_started = None
            self.checkpoint()

    def model_event(self, event):
        self.model_calls.append(dict(event))
        self.checkpoint()

    @contextmanager
    def measure(self, name):
        started = perf_counter()
        try:
            yield
        finally:
            self.operations.append({"name": name, "seconds": round(perf_counter() - started, 3)})
            self.checkpoint()

    def snapshot(self):
        elapsed = perf_counter() - self.started
        stages = list(self.stages)
        if self._stage_name:
            stages.append(
                {
                    "name": self._stage_name,
                    "seconds": round(perf_counter() - self._stage_started, 3),
                    "active": True,
                }
            )
        grouped = defaultdict(list)
        for event in self.model_calls:
            grouped[(event.get("role", "unknown"), event.get("status", "unknown"))].append(
                event.get("seconds", 0)
            )
        summary = [
            {
                "role": role,
                "status": status,
                "calls": len(durations),
                "seconds_total": round(sum(durations), 3),
                "seconds_max": round(max(durations, default=0), 3),
            }
            for (role, status), durations in sorted(grouped.items())
        ]
        grouped_operations = defaultdict(list)
        for event in self.operations:
            grouped_operations[event["name"]].append(event["seconds"])
        return {
            "operation": self.operation,
            "status": self.status,
            "started_at": self.started_at,
            "elapsed_seconds": round(elapsed, 3),
            "stages": stages,
            "model_summary": summary,
            "model_calls": list(self.model_calls),
            "operation_summary": [
                {
                    "name": name,
                    "count": len(durations),
                    "seconds_total": round(sum(durations), 3),
                    "seconds_max": round(max(durations, default=0), 3),
                }
                for name, durations in sorted(grouped_operations.items())
            ],
            "operations": list(self.operations),
            "photos": list(self.photos),
        }

    def checkpoint(self):
        write_json(self.root / "reports" / "performance.json", self.snapshot())

    def finish(self, status="complete"):
        self.photo_end()
        if self._stage_name:
            self.stages.append(
                {"name": self._stage_name, "seconds": round(perf_counter() - self._stage_started, 3)}
            )
            self._stage_name = self._stage_started = None
        self.status = status
        self.checkpoint()
