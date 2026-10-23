# Model registry integrity boundary

`src/model_registry.py` is a small SQLite reference implementation of the controls required around
a credit-risk model promotion. It is intentionally stricter than a mutable demo registry.

## Registration contract

A model version is an immutable identifier, not an alias. Registration therefore:

- accepts only bounded identifiers containing letters, digits, `.`, `_` and `-`;
- rejects mutable aliases such as `latest`, `current` and `production`;
- requires a real artifact file and finite, non-empty metrics;
- records the artifact's SHA-256 digest;
- uses `INSERT`, never `INSERT OR REPLACE`, so a duplicate identifier cannot overwrite evidence;
- permits only candidate or staging as an initial state.

The registry stores an artifact path for this local reference architecture. Production object
storage should additionally enforce versioning or write-once retention; a database digest does not
prevent someone with filesystem access from replacing the file.

## Promotion transaction

Promotion starts a SQLite `BEGIN IMMEDIATE` transaction. While holding that write reservation it:

1. reloads the current model record;
2. validates the requested state transition;
3. recomputes and compares the artifact SHA-256 digest;
4. archives any previous production model;
5. promotes the selected version with an optimistic stage predicate;
6. writes the reason and old/new stages to the promotion log.

A partial unique index on `stage = 'production'` is a final database-level invariant. If any step,
including audit logging, fails, SQLite rolls back the previous-model archive and candidate
promotion together. `promotion_history()` exposes the committed audit sequence.

Stable `ModelRegistryError` codes distinguish duplicate versions, missing or modified artifacts,
invalid metadata and transitions, concurrency conflicts, and transactional storage failures.

## Deliberate limits

- SHA-256 detects changed bytes but does not authenticate who registered them.
- The file may still change after a successful promotion; serving calls should verify the artifact
  before loading or use immutable object storage.
- SQLite serializes writers on one database file. Multi-region deployment needs an external
  transactional registry or consensus-backed control plane.
- Promotion checks artifact integrity and lifecycle state, not business-model quality. Calibration,
  discrimination, fairness and operational gates remain separate inputs to the decision.

The next increment is to bind the point-in-time calibration report, source snapshot digest and
promotion decision into one signed release manifest.
