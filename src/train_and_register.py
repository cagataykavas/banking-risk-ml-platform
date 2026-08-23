from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.model_registry import ModelRegistry
from src.risk_pipeline import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train, register and optionally promote a synthetic credit-risk model")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/model"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.db"))
    parser.add_argument("--rows", type=int, default=12000)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--min-roc-auc", type=float, default=0.68)
    parser.add_argument("--max-brier", type=float, default=0.23)
    args = parser.parse_args()

    metrics = train(args.artifact_dir, rows=args.rows)
    version = datetime.now(timezone.utc).strftime("risk-%Y%m%d-%H%M%S")
    registry = ModelRegistry(args.registry)
    record = registry.register(
        version=version,
        artifact_path=args.artifact_dir / "credit_risk.joblib",
        metrics=metrics,
    )

    promoted = False
    if args.promote:
        gates = metrics["roc_auc"] >= args.min_roc_auc and metrics["brier_score"] <= args.max_brier
        if gates:
            record = registry.promote(
                version,
                "production",
                reason=f"quality gates passed: roc_auc>={args.min_roc_auc}, brier<={args.max_brier}",
            )
            promoted = True

    print(json.dumps({"record": record.__dict__, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
