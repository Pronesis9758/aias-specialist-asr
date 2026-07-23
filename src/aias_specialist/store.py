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
CREATE TABLE IF NOT EXISTS experiment_groups (
    group_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    config_path TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS experiment_group_members (
    group_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    run_id TEXT,
    model_id TEXT NOT NULL,
    variant_id TEXT NOT NULL,
    status TEXT NOT NULL,
    config_path TEXT NOT NULL,
    error TEXT,
    PRIMARY KEY (group_id, member_id),
    FOREIGN KEY (group_id) REFERENCES experiment_groups(group_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS experiment_selections (
    selection_id TEXT PRIMARY KEY,
    group_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    selected_at TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    reason TEXT NOT NULL,
    selection_path TEXT NOT NULL,
    FOREIGN KEY (group_id) REFERENCES experiment_groups(group_id),
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

    def start_group(
        self,
        *,
        group_id: str,
        kind: str,
        name: str,
        config_path: Path,
        output_dir: Path,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO experiment_groups (
                    group_id, kind, name, started_at, status, config_path, output_dir
                ) VALUES (?, ?, ?, ?, 'running', ?, ?)
                """,
                (
                    group_id,
                    kind,
                    name,
                    utc_now().isoformat(),
                    str(config_path),
                    str(output_dir),
                ),
            )
            connection.execute(
                """
                UPDATE experiment_groups
                SET status = 'running', completed_at = NULL, error = NULL
                WHERE group_id = ? AND status != 'completed'
                """,
                (group_id,),
            )

    def finish_group(self, group_id: str, status: str, error: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE experiment_groups
                SET completed_at = ?, status = ?, error = ?
                WHERE group_id = ?
                """,
                (utc_now().isoformat(), status, error, group_id),
            )

    def upsert_group_member(
        self,
        *,
        group_id: str,
        member_id: str,
        model_id: str,
        variant_id: str,
        status: str,
        config_path: Path,
        run_id: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO experiment_group_members (
                    group_id, member_id, run_id, model_id, variant_id,
                    status, config_path, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(group_id, member_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    model_id = excluded.model_id,
                    variant_id = excluded.variant_id,
                    status = excluded.status,
                    config_path = excluded.config_path,
                    error = excluded.error
                """,
                (
                    group_id,
                    member_id,
                    run_id,
                    model_id,
                    variant_id,
                    status,
                    str(config_path),
                    error,
                ),
            )

    def group_members(self, group_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT group_id, member_id, run_id, model_id, variant_id,
                       status, config_path, error
                FROM experiment_group_members
                WHERE group_id = ?
                ORDER BY member_id
                """,
                (group_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def add_selection(
        self,
        *,
        selection_id: str,
        group_id: str,
        member_id: str,
        run_id: str,
        reviewer: str,
        reason: str,
        selection_path: Path,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiment_selections (
                    selection_id, group_id, member_id, run_id, selected_at,
                    reviewer, reason, selection_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selection_id,
                    group_id,
                    member_id,
                    run_id,
                    utc_now().isoformat(),
                    reviewer,
                    reason,
                    str(selection_path),
                ),
            )


def register_run_artifacts(store: ExperimentStore, run_id: str, run_dir: Path) -> None:
    """Register every immutable file produced inside a run directory."""
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        artifact_type = path.suffix.lstrip(".") or "file"
        store.add_artifact(run_id, artifact_type, path, sha256_file(path))
