from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VERSION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MODEL_STAGES = frozenset({"candidate", "staging", "production", "archived"})
MUTABLE_MODEL_VERSIONS = frozenset(
    {"current", "latest", "main", "master", "production"}
)
STAGE_TRANSITIONS = {
    "candidate": frozenset({"staging", "production", "archived"}),
    "staging": frozenset({"production", "archived"}),
    "production": frozenset({"archived"}),
    "archived": frozenset(),
}


class ModelRegistryError(ValueError):
    """Stable, machine-readable model registry rejection."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


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
                CREATE UNIQUE INDEX IF NOT EXISTS one_production_model
                ON model_versions(stage) WHERE stage = 'production';
                """
            )

    @staticmethod
    def _validate_version(version: str) -> None:
        if (
            not isinstance(version, str)
            or not VERSION_PATTERN.fullmatch(version)
            or version.casefold() in MUTABLE_MODEL_VERSIONS
        ):
            raise ModelRegistryError("INVALID_MODEL_VERSION")

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if stage not in MODEL_STAGES:
            raise ModelRegistryError("INVALID_MODEL_STAGE")

    @staticmethod
    def _validated_metrics(metrics: dict[str, float]) -> dict[str, float]:
        if not isinstance(metrics, dict) or not metrics:
            raise ModelRegistryError("INVALID_MODEL_METRICS")
        normalized: dict[str, float] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip() or isinstance(value, bool):
                raise ModelRegistryError("INVALID_MODEL_METRICS")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ModelRegistryError("INVALID_MODEL_METRICS") from exc
            if not math.isfinite(numeric):
                raise ModelRegistryError("INVALID_MODEL_METRICS")
            normalized[name] = numeric
        return normalized

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ModelVersion:
        return ModelVersion(
            version=row["version"],
            artifact_path=row["artifact_path"],
            sha256=row["sha256"],
            created_at=row["created_at"],
            metrics=json.loads(row["metrics_json"]),
            stage=row["stage"],
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
        self._validate_version(version)
        self._validate_stage(stage)
        if stage not in {"candidate", "staging"}:
            raise ModelRegistryError("INVALID_INITIAL_STAGE")
        normalized_metrics = self._validated_metrics(metrics)
        path = Path(artifact_path)
        if not path.is_file():
            raise ModelRegistryError("MODEL_ARTIFACT_NOT_FOUND")
        record = ModelVersion(
            version=version,
            artifact_path=str(path),
            sha256=self.checksum(path),
            created_at=datetime.now(UTC).isoformat(),
            metrics=normalized_metrics,
            stage=stage,
        )
        try:
            with self._connect() as con:
                con.execute(
                    "INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.version,
                        record.artifact_path,
                        record.sha256,
                        record.created_at,
                        json.dumps(record.metrics, sort_keys=True),
                        record.stage,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ModelRegistryError("MODEL_VERSION_ALREADY_EXISTS") from exc
        return record

    def get(self, version: str) -> ModelVersion:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM model_versions WHERE version = ?", (version,)
            ).fetchone()
        if row is None:
            raise KeyError(version)
        return self._record_from_row(row)

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
        self._validate_version(version)
        self._validate_stage(new_stage)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1_000:
            raise ModelRegistryError("INVALID_PROMOTION_REASON")
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE")
                row = con.execute(
                    "SELECT * FROM model_versions WHERE version = ?", (version,)
                ).fetchone()
                if row is None:
                    raise KeyError(version)
                current = self._record_from_row(row)
                if new_stage not in STAGE_TRANSITIONS[current.stage]:
                    raise ModelRegistryError("INVALID_STAGE_TRANSITION")

                path = Path(current.artifact_path)
                if not path.is_file() or self.checksum(path) != current.sha256:
                    raise ModelRegistryError("MODEL_ARTIFACT_INTEGRITY_FAILED")

                if new_stage == "production":
                    con.execute(
                        "UPDATE model_versions SET stage='archived' "
                        "WHERE stage='production' AND version != ?",
                        (version,),
                    )
                updated = con.execute(
                    "UPDATE model_versions SET stage = ? "
                    "WHERE version = ? AND stage = ?",
                    (new_stage, version, current.stage),
                )
                if updated.rowcount != 1:
                    raise ModelRegistryError("CONCURRENT_PROMOTION_CONFLICT")
                con.execute(
                    "INSERT INTO promotion_log(version, old_stage, new_stage, promoted_at, reason) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (version, current.stage, new_stage, now, reason.strip()),
                )
        except sqlite3.Error as exc:
            raise ModelRegistryError("PROMOTION_TRANSACTION_FAILED") from exc
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

    def promotion_history(self, version: str | None = None) -> list[dict[str, Any]]:
        if version is not None:
            self._validate_version(version)
        query = "SELECT version, old_stage, new_stage, promoted_at, reason FROM promotion_log"
        params: tuple[Any, ...] = ()
        if version is not None:
            query += " WHERE version = ?"
            params = (version,)
        query += " ORDER BY id"
        with self._connect() as con:
            rows = con.execute(query, params).fetchall()
        return [dict(row) for row in rows]
