from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelVersion:
    version: str
    artifact_path: str
    sha256: str
    created_at: str
    metrics: dict[str, float]
    stage: str = "candidate"


class ModelRegistry:
    """Tiny local model registry used by the demo platform.

    It deliberately models the concepts that matter in a production registry:
    immutable versions, artifact checksums, metrics, promotion stages and an
    auditable promotion history. The storage backend is SQLite so the complete
    project remains runnable without external services.
    """

    def __init__(self, db_path: str | Path = "artifacts/registry.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_versions (
                    version TEXT PRIMARY KEY,
                    artifact_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    stage TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS promotion_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL,
                    old_stage TEXT,
                    new_stage TEXT NOT NULL,
                    promoted_at TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def checksum(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def register(
        self,
        version: str,
        artifact_path: str | Path,
        metrics: dict[str, float],
        stage: str = "candidate",
    ) -> ModelVersion:
        path = Path(artifact_path)
        record = ModelVersion(
            version=version,
            artifact_path=str(path),
            sha256=self.checksum(path),
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            stage=stage,
        )
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO model_versions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.version,
                    record.artifact_path,
                    record.sha256,
                    record.created_at,
                    json.dumps(record.metrics, sort_keys=True),
                    record.stage,
                ),
            )
        return record

    def get(self, version: str) -> ModelVersion:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
        if row is None:
            raise KeyError(version)
        return ModelVersion(
            version=row["version"],
            artifact_path=row["artifact_path"],
            sha256=row["sha256"],
            created_at=row["created_at"],
            metrics=json.loads(row["metrics_json"]),
            stage=row["stage"],
        )

    def latest(self, stage: str | None = None) -> ModelVersion:
        query = "SELECT version FROM model_versions"
        params: tuple[Any, ...] = ()
        if stage is not None:
            query += " WHERE stage = ?"
            params = (stage,)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as con:
            row = con.execute(query, params).fetchone()
        if row is None:
            raise KeyError(f"no model found for stage={stage!r}")
        return self.get(row["version"])

    def promote(self, version: str, new_stage: str, reason: str) -> ModelVersion:
        current = self.get(version)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            if new_stage == "production":
                con.execute(
                    "UPDATE model_versions SET stage='archived' WHERE stage='production'"
                )
            con.execute(
                "UPDATE model_versions SET stage = ? WHERE version = ?",
                (new_stage, version),
            )
            con.execute(
                "INSERT INTO promotion_log(version, old_stage, new_stage, promoted_at, reason) VALUES (?, ?, ?, ?, ?)",
                (version, current.stage, new_stage, now, reason),
            )
        return self.get(version)

    def verify_artifact(self, version: str) -> bool:
        record = self.get(version)
        path = Path(record.artifact_path)
        return path.exists() and self.checksum(path) == record.sha256

    def list_versions(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT version FROM model_versions ORDER BY created_at DESC"
            ).fetchall()
        return [asdict(self.get(row["version"])) for row in rows]
