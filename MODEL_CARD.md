# Model Card — Synthetic Banking Risk Decision Model

> Portfolio reference implementation. All data is synthetic. This is not a regulatory credit model, IFRS 9 engine, or production underwriting policy.

## Intended use

Estimate a probability of adverse outcome for a synthetic banking application and demonstrate how probability estimates connect to controlled business decisions.

The model is designed to support discussions around:

- probability estimation rather than raw class labels;
- calibration and discrimination;
- temporal validation;
- decision thresholds and review capacity;
- expected-loss / cost-sensitive policy design;
- model registry and promotion controls;
- human review and override monitoring;
- drift and segment stability;
- explanation methods and their limitations.

## Out-of-scope use

Do not use this repository to make real credit, employment, insurance, housing, or other high-impact decisions. The synthetic data, costs, thresholds, and policy assumptions are illustrative only.

## Data

Synthetic applications contain demographic-neutral financial and account-behavior-style features such as income, debt, utilization, late-payment counts, account age, and employment/housing categories.

### Leakage controls

- Temporal examples use ordered / out-of-time validation rather than random future leakage.
- Rolling transaction features use prior-event windows where practical.
- Target-derived features are prohibited.
- Feature timestamps should be no later than the decision timestamp.

## Candidate models

Current public examples include:

- HistGradientBoostingClassifier baseline;
- logistic-regression baselines for interpretability and imbalance experiments;
- optional XGBoost and LightGBM challengers;
- Optuna tuning in the advanced research layer.

The purpose of multiple candidates is not leaderboard chasing. It is to demonstrate champion/challenger governance and the trade-off between discrimination, calibration, latency, review burden, and expected business cost.

## Evaluation

### Statistical metrics

- ROC-AUC
- average precision / PR-AUC
- Brier score
- expected calibration error
- KS statistic
- lift@K

### Operational metrics

- auto-approval rate
- auto-decline rate
- human-review rate
- bad rate among approved cases
- captured adverse-event rate
- expected cost per application
- p95 inference latency

### Stability

- rolling out-of-time validation
- PSI and KS drift checks
- country / segment metric breakdowns
- feature-importance stability across folds/runs
- rank consistency and sign consistency

## Decision policy

The score is not treated as a business decision by itself. A two-threshold policy creates three routes:

```text
low estimated risk      -> auto-approve candidate
uncertain middle band   -> human review
high estimated risk     -> auto-decline candidate
```

Thresholds can be searched under constraints for review capacity, minimum approval rate, default capture, and an explicit cost model.

## Explainability

Two intentionally different explanation methods are present:

1. `src/explainability.py`: model-agnostic local perturbation / sensitivity analysis. It is **not** labeled SHAP.
2. `advanced/shap_explain.py`: actual SHAP TreeExplainer utilities for supported tree models.

Explanations are diagnostic aids, not causal proofs.

## Champion / challenger governance

A challenger is not promoted just because its AUC is higher. Promotion gates can include:

- minimum ROC-AUC or AP improvement;
- bounded Brier/calibration regression;
- expected-cost improvement;
- p95 latency ceiling;
- population stability threshold;
- human-review capacity impact.

A rejected challenger remains auditable rather than silently replacing the production model.

## Monitoring

Production-oriented monitoring should cover:

- input contract failures;
- feature freshness;
- missing values and range violations;
- feature PSI/KS;
- score drift;
- calibration after delayed labels arrive;
- approval / decline / review rates;
- reviewer override rates;
- latency and error rate;
- segment-level degradation.

## Retraining triggers

Example triggers include:

- sustained PSI above a configured threshold;
- material calibration deterioration;
- segment-level performance regression;
- business-cost degradation;
- upstream feature/schema changes;
- scheduled refresh after enough delayed labels mature.

A trigger starts evaluation; it does not automatically guarantee promotion.

## Reproducibility

- deterministic synthetic generation where practical;
- explicit random seeds;
- versioned artifacts;
- SHA-256 artifact integrity checks;
- registry metadata;
- CI for tests/lint/container build;
- explicit model and policy metrics.

## Governance checklist

Before promoting a candidate:

- [ ] data contract passed
- [ ] temporal validation completed
- [ ] no known leakage
- [ ] discrimination gates passed
- [ ] calibration gates passed
- [ ] cost/threshold analysis completed
- [ ] segment stability reviewed
- [ ] drift baseline captured
- [ ] explanation sanity checks completed
- [ ] latency/load test completed
- [ ] review-capacity impact accepted
- [ ] artifact checksum stored
- [ ] rollback target available
- [ ] monitoring dashboards/alerts configured
- [ ] promotion reason recorded

## Limitations

Synthetic data cannot reproduce real portfolio selection effects, policy feedback loops, macroeconomic shocks, reporting delays, reject inference, legal constraints, fairness requirements, or institution-specific risk appetite. Those topics belong in a real model-risk process and are intentionally not disguised as solved by this portfolio repository.
