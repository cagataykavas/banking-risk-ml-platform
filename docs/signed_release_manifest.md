# Signed release manifest

The registry proves that a promoted model still has the bytes registered under its immutable
version. It does not, by itself, prove that those bytes were released with the point-in-time
validation evidence and source snapshot reviewed for that promotion.

`src/release_manifest.py` adds that release boundary. An offline release process signs a canonical
manifest with Ed25519. The serving process receives only a public-key rotation set and verifies the
manifest before deserializing the production model.

## What is bound

The signed manifest contains exactly these fields:

| Field | Commitment |
|---|---|
| `model_version` | Immutable registry version selected for serving |
| `artifact_sha256` | Exact registered model bytes |
| `validation_report_sha256` | Point-in-time validation evidence reviewed for release |
| `source_snapshot_sha256` | Training/validation source snapshot identifier or manifest |
| `promotion_record_sha256` | Domain-separated digest of the one committed production transition |
| `issued_at` | UTC release time used for the freshness policy |

The detached signature envelope also commits to the canonical manifest digest and identifies the
public key used. Verification rejects unknown keys, malformed or non-canonical documents, stale or
future-dated manifests, artifact/version substitutions, and promotion-history substitutions.
Manifest files are read as bounded regular files without following symbolic links.

## Signing outside the serving service

The signer should run only after validation and production promotion succeed. The private key must
come from a KMS, HSM, or short-lived release secret; it must not be stored in this repository,
container image, or serving environment. This abbreviated example shows the data flow, not a key
management design:

```python
from datetime import UTC, datetime

from src.release_manifest import (
    build_release_manifest,
    promotion_record_sha256,
    sign_release_manifest,
)

promotion_digest = promotion_record_sha256(
    registry.promotion_history(model.version),
    model.version,
)
manifest = build_release_manifest(
    model_version=model.version,
    artifact_sha256=model.sha256,
    validation_report_sha256=sha256_file(validation_report),
    source_snapshot_sha256=sha256_file(source_snapshot_manifest),
    promotion_record_sha256=promotion_digest,
    issued_at=datetime.now(UTC),
)
manifest_bytes, signature_bytes = sign_release_manifest(
    manifest,
    key_id="release-2026-q4",
    private_key=release_private_key_from_secret_manager(),
)
```

Write the two returned byte strings without reformatting them. Reformatting changes the canonical
bytes and invalidates the signature.

## Enforcing verification at the serving boundary

Provide all three required settings to activate enforcement:

```bash
export MODEL_RELEASE_MANIFEST_PATH=/run/release/model-release.json
export MODEL_RELEASE_SIGNATURE_PATH=/run/release/model-release.sig.json
export MODEL_RELEASE_PUBLIC_KEYS='{"release-2026-q4":"<URL-safe-base64-raw-Ed25519-public-key>"}'
export MODEL_RELEASE_MAX_AGE_SECONDS=2592000
```

`/ready` then reports `release_manifest_verified: true` and the verified manifest digest. Missing,
incomplete, stale, tampered, or mismatched release evidence makes readiness degraded and blocks
`/score` with HTTP 503 before `joblib.load` runs. Multiple public keys may be present during a
rotation. Removing a key revokes manifests signed by that key on the next verification.

For local development, omitting all three required settings preserves the unsigned demo flow.
Supplying only some of them fails closed. A production deployment should make these settings and
read-only mounted documents mandatory in its admission policy.

## Deliberate limits

- The signature authenticates a release only to the extent that the private key and approval
  workflow are controlled. This repository intentionally contains neither.
- Validation and source-snapshot digests are authenticated commitments. The API does not fetch or
  independently re-evaluate those external evidence files at request time.
- Public-key distribution, revocation status, and clock synchronization remain deployment
  responsibilities.
- A valid signature does not make Python pickle/joblib deserialization safe. Only trusted model
  artifacts should be registered, and production storage should be immutable.
- The local filesystem still has a verification-to-load race. Immutable object versions or an
  already-open verified descriptor are required for a stronger production boundary.

The next increment is a KMS/HSM-backed release command that publishes the two canonical documents
and a deployment admission check that requires verification settings for every production revision.
