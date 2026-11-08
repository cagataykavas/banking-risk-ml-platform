from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import HTTPException

from app import api
from src.release_manifest import (
    ReleaseManifestError,
    build_release_manifest,
    load_public_keyring,
    promotion_record_sha256,
    sign_release_manifest,
    verify_release_manifest,
    verify_release_manifest_files,
)

NOW = datetime(2026, 11, 8, 19, 45, tzinfo=UTC)
ARTIFACT_DIGEST = "a" * 64
VALIDATION_DIGEST = "b" * 64
SOURCE_DIGEST = "c" * 64
MODEL_VERSION = "risk-20261108-194500"


def promotion_history() -> list[dict[str, object]]:
    return [
        {
            "version": MODEL_VERSION,
            "old_stage": "candidate",
            "new_stage": "production",
            "promoted_at": "2026-11-08T19:44:00+00:00",
            "reason": "point-in-time validation accepted",
        }
    ]


def key_material() -> tuple[bytes, str]:
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private_bytes, base64.urlsafe_b64encode(public_bytes).decode("ascii")


def signed_bundle(*, issued_at: datetime = NOW):
    private_key, public_key = key_material()
    promotion_digest = promotion_record_sha256(promotion_history(), MODEL_VERSION)
    manifest = build_release_manifest(
        model_version=MODEL_VERSION,
        artifact_sha256=ARTIFACT_DIGEST,
        validation_report_sha256=VALIDATION_DIGEST,
        source_snapshot_sha256=SOURCE_DIGEST,
        promotion_record_sha256=promotion_digest,
        issued_at=issued_at,
    )
    manifest_payload, signature_payload = sign_release_manifest(
        manifest,
        key_id="release-2026-q4",
        private_key=private_key,
    )
    keyring = load_public_keyring(json.dumps({"release-2026-q4": public_key}))
    return manifest_payload, signature_payload, keyring, promotion_digest, public_key


def verify(
    manifest_payload: bytes,
    signature_payload: bytes,
    keyring,
    promotion_digest: str,
    **overrides,
):
    arguments = {
        "expected_model_version": MODEL_VERSION,
        "expected_artifact_sha256": ARTIFACT_DIGEST,
        "expected_promotion_record_sha256": promotion_digest,
        "now": NOW,
        "max_age_seconds": 86_400,
    }
    arguments.update(overrides)
    return verify_release_manifest(
        manifest_payload,
        signature_payload,
        keyring=keyring,
        **arguments,
    )


def test_signed_release_binds_model_validation_source_and_promotion() -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle()

    receipt = verify(manifest, signature, keyring, promotion_digest)

    assert receipt["verified"] is True
    assert receipt["algorithm"] == "ed25519"
    assert receipt["model_version"] == MODEL_VERSION
    assert receipt["artifact_sha256"] == ARTIFACT_DIGEST
    assert receipt["validation_report_sha256"] == VALIDATION_DIGEST
    assert receipt["source_snapshot_sha256"] == SOURCE_DIGEST
    assert receipt["promotion_record_sha256"] == promotion_digest
    assert receipt["receipt_sha256"]
    assert "point-in-time validation accepted" not in json.dumps(receipt)


@pytest.mark.parametrize(
    ("override", "code"),
    [
        ({"expected_model_version": "risk-other"}, "MODEL_VERSION_MISMATCH"),
        ({"expected_artifact_sha256": "d" * 64}, "ARTIFACT_DIGEST_MISMATCH"),
        (
            {"expected_promotion_record_sha256": "e" * 64},
            "PROMOTION_DIGEST_MISMATCH",
        ),
    ],
)
def test_registry_and_promotion_substitution_fails_closed(override, code: str) -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle()

    with pytest.raises(ReleaseManifestError, match=code):
        verify(manifest, signature, keyring, promotion_digest, **override)


def test_manifest_or_signature_tampering_fails_closed() -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle()
    changed_manifest = bytearray(manifest)
    changed_manifest[-2] ^= 1
    changed_signature = bytearray(signature)
    changed_signature[-2] ^= 1

    with pytest.raises(ReleaseManifestError):
        verify(bytes(changed_manifest), signature, keyring, promotion_digest)
    with pytest.raises(ReleaseManifestError):
        verify(manifest, bytes(changed_signature), keyring, promotion_digest)


def test_unknown_or_wrong_signing_key_fails_closed() -> None:
    manifest, signature, _, promotion_digest, _ = signed_bundle()
    _, other_public = key_material()

    with pytest.raises(ReleaseManifestError, match="SIGNING_KEY_UNKNOWN"):
        verify(
            manifest,
            signature,
            load_public_keyring(json.dumps({"other": other_public})),
            promotion_digest,
        )
    with pytest.raises(ReleaseManifestError, match="SIGNATURE_INVALID"):
        verify(
            manifest,
            signature,
            load_public_keyring(json.dumps({"release-2026-q4": other_public})),
            promotion_digest,
        )


@pytest.mark.parametrize(
    ("issued_at", "now", "code"),
    [
        (NOW - timedelta(days=2), NOW, "MANIFEST_EXPIRED"),
        (NOW + timedelta(seconds=301), NOW, "MANIFEST_FROM_FUTURE"),
    ],
)
def test_manifest_freshness_is_enforced(issued_at, now, code: str) -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle(
        issued_at=issued_at
    )

    with pytest.raises(ReleaseManifestError, match=code):
        verify(
            manifest,
            signature,
            keyring,
            promotion_digest,
            now=now,
            max_age_seconds=86_400,
        )


def test_canonical_documents_and_keyring_are_strict() -> None:
    manifest, signature, keyring, promotion_digest, public_key = signed_bundle()

    with pytest.raises(ReleaseManifestError, match="MANIFEST_INVALID"):
        verify(manifest + b"\n", signature, keyring, promotion_digest)
    with pytest.raises(ReleaseManifestError, match="DOCUMENT_DUPLICATE_FIELD"):
        load_public_keyring(
            '{"release-2026-q4":"'
            + public_key
            + '","release-2026-q4":"'
            + public_key
            + '"}'
        )
    with pytest.raises(ReleaseManifestError, match="KEYRING_INVALID"):
        load_public_keyring(json.dumps({"release-2026-q4": "short"}))


def test_manifest_timestamp_must_use_normalized_utc_form() -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle()
    document = json.loads(manifest)
    document["issued_at"] = document["issued_at"].replace("Z", "+00:00")
    non_normalized = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode()

    with pytest.raises(ReleaseManifestError, match="MANIFEST_INVALID"):
        verify(non_normalized, signature, keyring, promotion_digest)


def test_promotion_commitment_requires_exactly_one_production_transition() -> None:
    with pytest.raises(ReleaseManifestError, match="PROMOTION_RECORD_INVALID"):
        promotion_record_sha256([], MODEL_VERSION)
    duplicate = promotion_history() * 2
    with pytest.raises(ReleaseManifestError, match="PROMOTION_RECORD_INVALID"):
        promotion_record_sha256(duplicate, MODEL_VERSION)


def test_file_verifier_rejects_symlinks(tmp_path: Path) -> None:
    manifest, signature, keyring, promotion_digest, _ = signed_bundle()
    real_manifest = tmp_path / "release.json"
    linked_manifest = tmp_path / "linked.json"
    signature_path = tmp_path / "release.sig.json"
    real_manifest.write_bytes(manifest)
    signature_path.write_bytes(signature)
    linked_manifest.symlink_to(real_manifest)

    with pytest.raises(ReleaseManifestError, match="DOCUMENT_UNAVAILABLE"):
        verify_release_manifest_files(
            linked_manifest,
            signature_path,
            keyring=keyring,
            expected_model_version=MODEL_VERSION,
            expected_artifact_sha256=ARTIFACT_DIGEST,
            expected_promotion_record_sha256=promotion_digest,
            now=NOW,
            max_age_seconds=86_400,
        )


def test_readiness_requires_valid_manifest_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    issued_at = datetime.now(UTC)
    manifest, signature, _, _, public_key = signed_bundle(issued_at=issued_at)
    manifest_path = tmp_path / "release.json"
    signature_path = tmp_path / "release.sig.json"
    manifest_path.write_bytes(manifest)
    signature_path.write_bytes(signature)
    record = SimpleNamespace(version=MODEL_VERSION, sha256=ARTIFACT_DIGEST)

    monkeypatch.setattr(api, "RELEASE_MANIFEST_PATH", str(manifest_path))
    monkeypatch.setattr(api, "RELEASE_SIGNATURE_PATH", str(signature_path))
    monkeypatch.setattr(
        api,
        "RELEASE_PUBLIC_KEYS",
        json.dumps({"release-2026-q4": public_key}),
    )
    monkeypatch.setattr(api, "RELEASE_MAX_AGE_SECONDS", "86400")
    monkeypatch.setattr(api.REGISTRY, "latest", lambda _stage: record)
    monkeypatch.setattr(api.REGISTRY, "verify_artifact", lambda _version: True)
    monkeypatch.setattr(
        api.REGISTRY, "promotion_history", lambda _version: promotion_history()
    )

    ready = api.readiness()
    assert ready["ready"] is True
    assert ready["release_manifest_verified"] is True
    assert ready["release_manifest_sha256"]

    signature_path.write_bytes(signature[:-1] + b"x")
    degraded = api.readiness()
    assert degraded["ready"] is False
    assert degraded["artifact_verified"] is True
    assert degraded["release_manifest_verified"] is False
    with pytest.raises(HTTPException, match="release manifest"):
        api._load_production_model()


def test_incomplete_release_configuration_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = SimpleNamespace(version=MODEL_VERSION, sha256=ARTIFACT_DIGEST)
    monkeypatch.setattr(api, "RELEASE_MANIFEST_PATH", "/run/release/release.json")
    monkeypatch.setattr(api, "RELEASE_SIGNATURE_PATH", None)
    monkeypatch.setattr(api, "RELEASE_PUBLIC_KEYS", None)
    monkeypatch.setattr(api.REGISTRY, "latest", lambda _stage: record)
    monkeypatch.setattr(api.REGISTRY, "verify_artifact", lambda _version: True)

    ready = api.readiness()

    assert ready["ready"] is False
    assert ready["artifact_verified"] is True
    assert ready["release_manifest_verified"] is False
