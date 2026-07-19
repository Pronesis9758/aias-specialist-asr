from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .utils import sha256_file, utc_now

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    project_name TEXT NOT NULL,
    config_path TEXT NOT NULL,
    backend TEXT NOT NULL,
    model_repo TEXT NOT NULL,
    model_revision TEXT,
    git_sha TEXT,
    run_dir TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    PRIMARY KEY (run_id, scope, metric_name),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS artifacts (
    run_id TEXT NOT NULL,
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    PRIMARY KEY (run_id, path),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
"""


class ExperimentStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def start_run(self, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, started_at, status, project_name, config_path, backend,
                    model_repo, git_sha, run_dir
                ) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"],
                    payload["started_at"],
                    payload["project_name"],
                    payload["config_path"],
                    payload["backend"],
                    payload["model_repo"],
                    payload.get("git_sha"),
                    payload["run_dir"],
                ),
            )

    def event(self, run_id: str, stage: str, status: str, message: str = "") -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO events (run_id, occurred_at, stage, status, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, utc_now().isoformat(), stage, status, message),
            )

    def finish_run(
        self,
        run_id: str,
        status: str,
        model_revision: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE runs
                SET completed_at = ?, status = ?, model_revision = ?, error = ?
                WHERE run_id = ?
                """,
                (utc_now().isoformat(), status, model_revision, error, run_id),
            )

    def add_metrics(self, run_id: str, scope: str, metrics: dict[str, Any]) -> None:
        numeric = [
            (run_id, scope, name, float(value))
            for name, value in metrics.items()
            if isinstance(value, int | float)
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO metrics (run_id, scope, metric_name, metric_value)
                VALUES (?, ?, ?, ?)
                """,
                numeric,
            )

    def add_artifact(self, run_id: str, artifact_type: str, path: Path, sha256: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts (run_id, artifact_type, path, sha256)
                VALUES (?, ?, ?, ?)
                """,
                (run_id, artifact_type, str(path), sha256),
            )

    def history(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, started_at, completed_at, status, backend, model_repo,
                       model_revision, run_dir, error
                FROM runs ORDER BY started_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]


def register_run_artifacts(store: ExperimentStore, run_id: str, run_dir: Path) -> None:
    """Register every immutable file produced inside a run directory."""
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        artifact_type = path.suffix.lstrip(".") or "file"
        store.add_artifact(run_id, artifact_type, path, sha256_file(path))
