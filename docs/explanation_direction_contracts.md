# Governed explanation-direction audit

A plausible-looking local explanation can contradict a domain contract even when aggregate model quality remains acceptable. In credit risk, increasing `late_payments` should not reduce predicted risk, while increasing account history should not increase it under an explicitly governed monotonic policy.

`src.explanation_direction_audit` evaluates paired baseline/perturbed predictions and their local attribution deltas before an explanation method is promoted.

Each pair must move the feature in its declared **risk-increasing** direction: upward for `nondecreasing_risk`, downward for `nonincreasing_risk`. The expected probability and attribution deltas are therefore non-negative in both cases.

## What the gate checks

- the artifact is bound to exact model and cohort SHA-256 identities;
- a server-owned policy declares each feature's expected risk direction and minimum perturbation step;
- every declared feature has sufficient paired-case coverage;
- model probability changes follow the expected direction within a bounded tolerance;
- attribution deltas follow the same direction and reconcile with the observed probability delta;
- an explanation cannot pass by producing mostly immaterial, flat perturbations;
- stale, future-dated, duplicate, non-finite, undeclared or oversized evidence fails closed;
- reports contain aggregate feature statistics and digests, not case IDs or feature values.

The policy is separate from the evidence artifact so the evidence producer cannot widen its own acceptance thresholds. Example feature contracts:

```json
{
  "schema_version": 1,
  "required_explanation_method": "one_feature_counterfactual_sensitivity",
  "min_cases_per_feature": 20,
  "max_cases": 20000,
  "max_features": 128,
  "max_artifact_bytes": 4000000,
  "max_age_seconds": 86400,
  "max_future_skew_seconds": 300,
  "direction_tolerance": 1e-9,
  "max_attribution_error": 1e-6,
  "minimum_material_effect": 1e-4,
  "max_model_direction_breach_fraction": 0.05,
  "max_attribution_direction_breach_fraction": 0.05,
  "max_attribution_mismatch_fraction": 0.0,
  "max_flat_effect_fraction": 0.8,
  "features": [
    {"feature": "late_payments", "direction": "nondecreasing_risk", "minimum_step": 1},
    {"feature": "account_age_months", "direction": "nonincreasing_risk", "minimum_step": 12}
  ]
}
```

Run the audit with a reproducible evaluation time:

```bash
python -m src.explanation_direction_audit evidence.json \
  --policy direction-policy.json \
  --now 2026-09-27T12:00:00Z
```

Exit code `0` means accepted, `2` means well-formed evidence violated policy, and `3` means the policy or evidence artifact was malformed.

## Trust boundary

Monotonic direction is a governed domain assumption, not a universal causal fact. The audit tests the supplied perturbation cohort and cannot prove global monotonicity between sampled points. Correlated or infeasible counterfactuals can also mislead; production evidence should therefore come from a versioned, feasibility-aware sampler.

This gate does not certify fairness, calibration or causal recourse. The next increment is to generate the artifact directly from the registered model and cohort manifest, add conditional-feasibility constraints, and bind the accepted evidence digest to model promotion.
