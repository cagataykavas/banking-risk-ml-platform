from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

MANIFEST_SCHEMA_VERSION = "banking-risk-release/v1"
SIGNATURE_SCHEMA_VERSION = "banking-risk-release-signature/v1"
RECEIPT_SCHEMA_VERSION = "banking-risk-release-verification/v1"
MAX_DOCUMENT_BYTES = 8_192
MAX_KEYRING_BYTES = 16_384
MAX_KEYS = 16
MAX_FUTURE_SKEW_SECONDS = 300
MANIFEST_FIELDS = {
    "schema_version",
    "model_version",
    "artifact_sha256",
    "validation_report_sha256",
    "source_snapshot_sha256",
    "promotion_record_sha256",
    "issued_at",
}
SIGNATURE_FIELDS = {
    "schema_version",
    "algorithm",
    "key_id",
    "manifest_sha256",
    "signature_b64",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PROMOTION_DOMAIN = b"banking-risk-ml-platform/promotion-record/v1\0"


class ReleaseManifestError(ValueError):
    """Stable, non-sensitive release verification failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestError("DOCUMENT_INVALID") from exc


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseManifestError("DOCUMENT_DUPLICATE_FIELD")
        result[key] = value
    return result


def _parse_canonical_json(payload: bytes, *, code: str) -> dict[str, object]:
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ReleaseManifestError(code)
            ),
        )
    except ReleaseManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(code) from exc
    if not isinstance(document, dict) or _canonical_json(document) != payload:
        raise ReleaseManifestError(code)
    return document


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_digest(value: object, code: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ReleaseManifestError(code)
    return value


def _identifier(value: object, code: str) -> str:
    if not isinstance(value, str) or IDENTIFIER_RE.fullmatch(value) is None:
        raise ReleaseManifestError(code)
    return value


def _timestamp(value: object, code: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise ReleaseManifestError(code)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseManifestError(code) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseManifestError(code)
    return parsed.astimezone(UTC)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ReleaseManifestError("ISSUED_AT_INVALID")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def promotion_record_sha256(
    history: Sequence[Mapping[str, object]], model_version: str
) -> str:
    """Commit to the one production transition for an immutable model version."""

    version = _identifier(model_version, "MODEL_VERSION_INVALID")
    matches: list[dict[str, object]] = []
    expected_fields = {
        "version",
        "old_stage",
        "new_stage",
        "promoted_at",
        "reason",
    }
    for raw in history:
        record = dict(raw)
        if set(record) != expected_fields:
            raise ReleaseManifestError("PROMOTION_RECORD_INVALID")
        if record["version"] == version and record["new_stage"] == "production":
            if not isinstance(record["reason"], str) or not record["reason"]:
                raise ReleaseManifestError("PROMOTION_RECORD_INVALID")
            _timestamp(record["promoted_at"], "PROMOTION_RECORD_INVALID")
            matches.append(record)
    if len(matches) != 1:
        raise ReleaseManifestError("PROMOTION_RECORD_INVALID")
    return _digest(_PROMOTION_DOMAIN + _canonical_json(matches[0]))


def build_release_manifest(
    *,
    model_version: str,
    artifact_sha256: str,
    validation_report_sha256: str,
    source_snapshot_sha256: str,
    promotion_record_sha256: str,
    issued_at: datetime,
) -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_version": _identifier(model_version, "MODEL_VERSION_INVALID"),
        "artifact_sha256": _required_digest(artifact_sha256, "ARTIFACT_DIGEST_INVALID"),
        "validation_report_sha256": _required_digest(
            validation_report_sha256, "VALIDATION_DIGEST_INVALID"
        ),
        "source_snapshot_sha256": _required_digest(
            source_snapshot_sha256, "SOURCE_SNAPSHOT_DIGEST_INVALID"
        ),
        "promotion_record_sha256": _required_digest(
            promotion_record_sha256, "PROMOTION_DIGEST_INVALID"
        ),
        "issued_at": _format_timestamp(issued_at),
    }


def sign_release_manifest(
    manifest: Mapping[str, object], *, key_id: str, private_key: bytes
) -> tuple[bytes, bytes]:
    """Return canonical manifest and detached Ed25519 signature documents."""

    key_name = _identifier(key_id, "KEY_ID_INVALID")
    if not isinstance(private_key, bytes) or len(private_key) != 32:
        raise ReleaseManifestError("PRIVATE_KEY_INVALID")
    manifest_payload = _canonical_json(dict(manifest))
    parsed = _parse_manifest(manifest_payload)
    manifest_payload = _canonical_json(parsed)
    signature = Ed25519PrivateKey.from_private_bytes(private_key).sign(manifest_payload)
    envelope = {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "algorithm": "ed25519",
        "key_id": key_name,
        "manifest_sha256": _digest(manifest_payload),
        "signature_b64": base64.urlsafe_b64encode(signature).decode("ascii"),
    }
    return manifest_payload, _canonical_json(envelope)


def load_public_keyring(raw: str) -> dict[str, Ed25519PublicKey]:
    if (
        not isinstance(raw, str)
        or not raw
        or len(raw.encode("utf-8")) > MAX_KEYRING_BYTES
    ):
        raise ReleaseManifestError("KEYRING_INVALID")
    try:
        document = json.loads(raw, object_pairs_hook=_reject_duplicates)
    except ReleaseManifestError:
        raise
    except json.JSONDecodeError as exc:
        raise ReleaseManifestError("KEYRING_INVALID") from exc
    if not isinstance(document, dict):
        raise ReleaseManifestError("KEYRING_INVALID")
    if not 1 <= len(document) <= MAX_KEYS:
        raise ReleaseManifestError("KEYRING_INVALID")
    keyring: dict[str, Ed25519PublicKey] = {}
    for key_id, encoded in document.items():
        _identifier(key_id, "KEYRING_INVALID")
        if not isinstance(encoded, str) or len(encoded) > 64:
            raise ReleaseManifestError("KEYRING_INVALID")
        try:
            public_bytes = base64.b64decode(encoded, altchars=b"-_", validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ReleaseManifestError("KEYRING_INVALID") from exc
        if len(public_bytes) != 32:
            raise ReleaseManifestError("KEYRING_INVALID")
        keyring[key_id] = Ed25519PublicKey.from_public_bytes(public_bytes)
    return keyring


def _parse_manifest(payload: bytes) -> dict[str, object]:
    document = _parse_canonical_json(payload, code="MANIFEST_INVALID")
    if (
        set(document) != MANIFEST_FIELDS
        or document["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ReleaseManifestError("MANIFEST_INVALID")
    normalized = build_release_manifest(
        model_version=document["model_version"],
        artifact_sha256=document["artifact_sha256"],
        validation_report_sha256=document["validation_report_sha256"],
        source_snapshot_sha256=document["source_snapshot_sha256"],
        promotion_record_sha256=document["promotion_record_sha256"],
        issued_at=_timestamp(document["issued_at"], "ISSUED_AT_INVALID"),
    )
    if document != normalized:
        raise ReleaseManifestError("MANIFEST_INVALID")
    return document


def _parse_signature(payload: bytes) -> dict[str, object]:
    document = _parse_canonical_json(payload, code="SIGNATURE_INVALID")
    if (
        set(document) != SIGNATURE_FIELDS
        or document["schema_version"] != SIGNATURE_SCHEMA_VERSION
        or document["algorithm"] != "ed25519"
    ):
        raise ReleaseManifestError("SIGNATURE_INVALID")
    _identifier(document["key_id"], "SIGNATURE_INVALID")
    _required_digest(document["manifest_sha256"], "SIGNATURE_INVALID")
    return document


def verify_release_manifest(
    manifest_payload: bytes,
    signature_payload: bytes,
    *,
    keyring: Mapping[str, Ed25519PublicKey],
    expected_model_version: str,
    expected_artifact_sha256: str,
    expected_promotion_record_sha256: str,
    now: datetime,
    max_age_seconds: int,
) -> dict[str, object]:
    if not 0 < max_age_seconds <= 365 * 86_400:
        raise ReleaseManifestError("MAX_AGE_INVALID")
    manifest = _parse_manifest(manifest_payload)
    signature = _parse_signature(signature_payload)
    key_id = signature["key_id"]
    public_key = keyring.get(key_id) if isinstance(key_id, str) else None
    if public_key is None:
        raise ReleaseManifestError("SIGNING_KEY_UNKNOWN")
    manifest_sha256 = _digest(manifest_payload)
    if not hmac.compare_digest(signature["manifest_sha256"], manifest_sha256):
        raise ReleaseManifestError("MANIFEST_DIGEST_MISMATCH")
    try:
        signature_bytes = base64.b64decode(
            signature["signature_b64"], altchars=b"-_", validate=True
        )
        public_key.verify(signature_bytes, manifest_payload)
    except (TypeError, ValueError, binascii.Error, InvalidSignature) as exc:
        raise ReleaseManifestError("SIGNATURE_INVALID") from exc

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ReleaseManifestError("NOW_INVALID")
    issued_at = _timestamp(manifest["issued_at"], "ISSUED_AT_INVALID")
    age_seconds = (now.astimezone(UTC) - issued_at).total_seconds()
    if age_seconds < -MAX_FUTURE_SKEW_SECONDS:
        raise ReleaseManifestError("MANIFEST_FROM_FUTURE")
    if age_seconds > max_age_seconds:
        raise ReleaseManifestError("MANIFEST_EXPIRED")

    expected_version = _identifier(expected_model_version, "MODEL_VERSION_INVALID")
    expected_artifact = _required_digest(
        expected_artifact_sha256, "ARTIFACT_DIGEST_INVALID"
    )
    expected_promotion = _required_digest(
        expected_promotion_record_sha256, "PROMOTION_DIGEST_INVALID"
    )
    if manifest["model_version"] != expected_version:
        raise ReleaseManifestError("MODEL_VERSION_MISMATCH")
    if not hmac.compare_digest(manifest["artifact_sha256"], expected_artifact):
        raise ReleaseManifestError("ARTIFACT_DIGEST_MISMATCH")
    if not hmac.compare_digest(manifest["promotion_record_sha256"], expected_promotion):
        raise ReleaseManifestError("PROMOTION_DIGEST_MISMATCH")

    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "verified": True,
        "algorithm": "ed25519",
        "key_id": key_id,
        "manifest_sha256": manifest_sha256,
        "model_version": expected_version,
        "artifact_sha256": expected_artifact,
        "validation_report_sha256": manifest["validation_report_sha256"],
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "promotion_record_sha256": expected_promotion,
        "issued_at": manifest["issued_at"],
    }
    receipt["receipt_sha256"] = _digest(_canonical_json(receipt))
    return receipt


def _read_document(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReleaseManifestError("DOCUMENT_UNAVAILABLE") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or not 1 <= metadata.st_size <= MAX_DOCUMENT_BYTES
        ):
            raise ReleaseManifestError("DOCUMENT_INVALID")
        payload = os.read(descriptor, MAX_DOCUMENT_BYTES + 1)
        after = os.fstat(descriptor)
        if (
            len(payload) != metadata.st_size
            or (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            )
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or metadata.st_ctime_ns != after.st_ctime_ns
        ):
            raise ReleaseManifestError("DOCUMENT_CHANGED")
        return payload
    finally:
        os.close(descriptor)


def verify_release_manifest_files(
    manifest_path: Path,
    signature_path: Path,
    **kwargs: object,
) -> dict[str, object]:
    return verify_release_manifest(
        _read_document(manifest_path),
        _read_document(signature_path),
        **kwargs,
    )
