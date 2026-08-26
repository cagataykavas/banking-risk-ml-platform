# Senior Data Scientist — Global Banking Interview Battle Sheet

This document is an interview-preparation companion to the public banking-risk platform. The answers are intentionally compact enough to rehearse aloud, while the repository provides code to defend the claims.

## 1. End-to-end ML and business framing

### 1. Walk me through an end-to-end banking ML project.
Start from the business decision, not the algorithm. Define the unit of prediction, label horizon, decision latency, cost of false positives/negatives and operational constraints. Build leakage-safe data, create a baseline, compare models, evaluate discrimination and calibration, choose thresholds against business costs/review capacity, register/version the model, deploy batch or online scoring, monitor data/score/performance drift, and feed human outcomes back into evaluation.

### 2. What is the difference between a model metric and a business metric?
A model metric describes statistical behavior: ROC-AUC, PR-AUC, log loss, Brier score, KS. A business metric describes operational/economic outcomes: approval rate, bad rate, fraud loss captured, review rate, conversion, expected loss, analyst hours, latency or revenue. Threshold selection connects the two.

### 3. Why can the highest-AUC model be the wrong model?
Because AUC is ranking quality across all thresholds. It does not guarantee good calibration, acceptable latency, stability, explainability or the best operating point under asymmetric costs. A slightly lower-AUC model can dominate economically at the actual threshold.

### 4. How do you choose a threshold?
Estimate the cost/value of each decision outcome and operational constraints such as maximum manual-review rate. Evaluate candidate thresholds or two-threshold policies and choose a region that balances expected loss, approval rate, bad rate, captured risk and reviewer capacity. Revalidate on future time periods.

### 5. What does expected loss mean?
In a transparent simplified form, `EL = PD × LGD × EAD`: probability of default, loss given default and exposure at default. A production regulatory implementation requires richer definitions and governance, but the decomposition is useful for connecting probability outputs to money.

## 2. Classification metrics

### 6. ROC-AUC vs PR-AUC?
ROC-AUC measures ranking across true-positive and false-positive rates and can look optimistic on very imbalanced data. PR-AUC focuses precision and recall for the positive class and is often more informative for rare fraud/default outcomes.

### 7. What is KS in credit risk?
The maximum separation between cumulative score distributions for positive and negative classes. For a classifier it can be calculated as the maximum `TPR - FPR` over thresholds. It is useful as a separation/ranking diagnostic, not a substitute for calibration or business evaluation.

### 8. What is lift?
The event rate in a selected high-score fraction divided by the overall event rate. Lift@10% tells us how much more concentrated positives are in the riskiest 10% than in the population.

### 9. What is calibration?
If a model assigns probability 0.20 to many observations, roughly 20% of those observations should experience the event. Calibration matters when probabilities drive pricing, expected loss or decision thresholds.

### 10. What is the Brier score?
Mean squared error between predicted probabilities and binary outcomes. Lower is better. It captures probabilistic accuracy and is sensitive to calibration as well as discrimination.

### 11. How do you evaluate calibration?
Reliability/calibration plots, Brier score, expected calibration error and comparison of observed vs predicted event rate within probability bins. Recheck calibration over time and across meaningful segments.

### 12. Precision vs recall in fraud?
Recall measures how much fraud we catch; precision measures what fraction of alerts are actually fraud. Very high recall can overwhelm investigators with false positives, so the decision is constrained by analyst capacity and loss economics.

## 3. Validation and leakage

### 13. Why is a random train/test split dangerous in financial data?
Because observations are time-dependent and random splitting can put future behavior, duplicated customer states or post-outcome information into the training set. Use chronological holdouts or rolling/expanding-window validation when deployment is forward-looking.

### 14. What is target leakage?
Any feature contains information unavailable at the real decision time or derived from the future target outcome. Examples include collection status after default, chargeback resolution after a fraud transaction, or a rolling statistic that accidentally includes the current/future event.

### 15. How do you create leakage-safe rolling features?
Order by entity and event time and use windows ending strictly before the current event. In the Spark example the range ends at `-1` second rather than including the current transaction.

### 16. Why use a temporal gap between train and validation?
To reduce contamination from near-duplicate events, processing latency or labels/features whose final state arrives shortly after the event. The appropriate gap depends on the data-generating process.

### 17. What is out-of-time validation?
Testing on a later, untouched period that represents the deployment future. It is particularly important for banking models because population and macroeconomic behavior drift.

## 4. Drift and monitoring

### 18. What is PSI?
Population Stability Index compares binned distributions between a reference population and a current population. Values around 0.10 are often treated as a warning and 0.25 as substantial drift in many industry contexts, but thresholds should be calibrated to the domain rather than treated as universal laws.

### 19. PSI vs KS for drift?
PSI summarizes distribution changes across bins and is easy to monitor; KS measures maximum empirical CDF separation. Both can signal drift, but neither proves model performance degraded. Performance monitoring requires labels when available.

### 20. What would you monitor in production?
Input schema/quality, missingness, feature distributions, feature freshness, prediction distributions, calibration when labels arrive, performance metrics, decision/override rates, latency, errors, throughput, model version and segment-specific outcomes.

### 21. Data drift vs concept drift?
Data/covariate drift means `P(X)` changes. Concept drift means the relationship `P(Y|X)` changes. The first can be detected without labels; the second ultimately requires outcomes or strong proxy monitoring.

## 5. Modeling

### 22. Why start with logistic regression?
It is fast, interpretable, well understood, easy to regularize and a valuable baseline. A complex model should beat it on the metrics and operating constraints that actually matter.

### 23. L1 vs L2 regularization?
L1 encourages sparse coefficients and can perform feature selection. L2 shrinks coefficients smoothly and is often more stable with correlated predictors. The choice is validated empirically and depends on interpretability/stability goals.

### 24. Trees vs linear models for credit risk?
Tree ensembles capture nonlinear interactions with less feature engineering; linear/logistic models are easier to reason about and can be very stable. The choice depends on performance, calibration, governance, latency, explanation requirements and data scale.

### 25. How do you handle class imbalance?
Use appropriate metrics, stratified/temporal validation, class weights or sampling where justified, and tune thresholds to costs. Do not assume oversampling automatically solves the business problem.

### 26. What is probability calibration after class weighting/resampling?
Training interventions can distort raw probabilities. If probabilities matter, evaluate and potentially calibrate them on an untouched representative validation set using methods such as Platt scaling or isotonic regression.

### 27. How do you explain a black-box model?
Global importance for population-level behavior, local explanations for individual decisions, counterfactual/sensitivity analysis, partial dependence/ICE where appropriate, and explicit validation of explanation stability. Always label the explanation method rather than calling any heuristic SHAP.

## 6. Statistics

### 28. What is bias-variance tradeoff?
High-bias models underfit systematic structure; high-variance models fit training-specific noise. Regularization, model complexity, sample size and ensembling affect the tradeoff.

### 29. What is a confidence interval?
A procedure that, under repeated sampling assumptions, produces intervals containing the true parameter at a specified long-run rate. It is not literally the probability that a fixed parameter lies in this particular frequentist interval.

### 30. Bootstrap use cases?
Estimate uncertainty of metrics/parameters when analytic formulas are inconvenient. Resample at the correct unit—e.g., customer rather than individual transactions when observations within customer are dependent.

### 31. Correlation vs causation?
Predictive association is not causal effect. Confounding, selection and feedback can produce correlations. A risk model can be useful predictively without supporting causal intervention claims.

## 7. Spark and data engineering

### 32. What is lazy evaluation in Spark?
Transformations build a logical plan; Spark executes when an action requires results. The optimizer can then rewrite/optimize the plan before execution.

### 33. Narrow vs wide transformations?
Narrow transformations can compute each output partition from a limited input partition set without full redistribution. Wide transformations such as `groupBy` or many joins require shuffle across executors and are generally more expensive.

### 34. Why is shuffle expensive?
It requires partitioning, serialization, disk/network I/O and coordination across executors. Large/skewed shuffles cause stragglers and memory pressure.

### 35. What is data skew?
Some partition keys contain far more rows than others, causing a few tasks to dominate runtime. Diagnose key frequencies/task metrics and mitigate with broadcast joins, AQE skew handling, salting, better partition keys or preaggregation.

### 36. When use a broadcast join?
When one side is small enough to distribute safely to executors, avoiding shuffle of the large fact table. The tradeoff is executor memory usage.

### 37. `repartition` vs `coalesce`?
`repartition` can increase or decrease partitions and performs shuffle. `coalesce` usually reduces partitions with less movement and is useful when shrinking output partition count, though it can create imbalance.

### 38. Why Parquet?
Columnar storage, compression, predicate/column pruning and strong integration with analytical engines. It is much better than row-oriented text formats for large analytical scans.

### 39. Why avoid Python UDFs when Spark built-ins exist?
Built-ins are visible to Catalyst and can benefit from optimized JVM execution/code generation. Python UDFs introduce serialization boundaries and obscure expressions from the optimizer.

### 40. What would you inspect when a Spark job is slow?
Physical plan, shuffle volume, stage/task durations, skew, spill, partition counts/sizes, join strategy, file sizes, predicate pushdown, cached data, executor memory/GC and unnecessary UDFs/actions.

## 8. GCP / production

### 41. BigQuery vs Cloud SQL?
BigQuery is a serverless analytical warehouse optimized for large scans/aggregations. Cloud SQL is managed relational OLTP. Do not use the warehouse as a transactional application database or the OLTP database as a petabyte analytical engine.

### 42. Vertex AI role in an ML platform?
Managed training/pipelines, metadata/experiments, model registry, endpoints, batch prediction and monitoring integrations. The exact architecture can mix managed and custom components.

### 43. Pub/Sub vs BigQuery?
Pub/Sub is event messaging/stream transport; BigQuery is analytical storage/query. A common pipeline sends events through Pub/Sub/Dataflow and lands curated data into BigQuery.

### 44. Cloud Run vs GKE for model serving?
Cloud Run is operationally simpler for stateless containerized HTTP workloads and scales to zero. GKE offers much more control for complex networking, sidecars, specialized scheduling, GPU-heavy or platform-level Kubernetes requirements.

### 45. How would you deploy a risk model safely?
Immutable artifact/version, registry, quality gates, reproducible preprocessing, staged/canary rollout when possible, health/latency monitoring, decision logging, rollback path, feature/score drift monitoring and delayed outcome evaluation.

### 46. What is model lineage?
Traceability from model version back to code, data snapshot/features, hyperparameters, environment and evaluation results. It makes debugging, governance and rollback possible.

## 9. Behavioral / seniority

### 47. Tell me about a model that did not work.
Use a real failure-analysis story. Explain the hypothesis, what metric/behavior disproved it, what diagnostics you ran, what you changed and what you learned. Avoid pretending every experiment became production.

### 48. How do you disagree with a stakeholder asking for “more accuracy”?
Translate “accuracy” into the actual decision. Show trade-off curves and ask which business cost/constraint matters. Present alternatives using expected loss, review capacity, approval rate or service-level impact.

### 49. How do you communicate uncertainty?
Separate what is measured, assumed and unknown. Use confidence intervals/sensitivity analysis when appropriate, describe data limitations and avoid presenting a probability score as certainty.

### 50. You have slightly under four years; why should we hire you for a senior posting?
Do not fake years. Emphasize breadth of lifecycle ownership and evidence: data engineering, model development, anomaly detection, explainability, deployment-oriented APIs/MLOps, financial-risk portfolio work, Spark/GCP labs and the ability to discuss business trade-offs. Let them decide whether scope compensates for calendar years.

## Repository map for answers

- `src/banking_metrics.py` — ROC-AUC, AP, Brier, KS, lift, ECE and operating-point analysis.
- `src/cost_sensitive_policy.py` — business-cost/review-capacity optimization and expected loss.
- `src/temporal_validation.py` — rolling out-of-time evaluation.
- `src/drift_monitoring.py` — PSI/KS feature and score drift.
- `fintech-data-platform/spark/banking_features.py` — leakage-aware rolling transaction features.
- `fintech-data-platform/spark/performance_lab.py` — broadcast joins, skew/salting, AQE and partitioning.
- `gcp-ml-platform/pipelines/banking_vertex_pipeline.py` — BigQuery/Vertex-oriented training, quality gate, registry and deployment flow.

## One-minute positioning

> I'm an AI/ML engineer with a strong engineering background, and I've been deliberately moving toward end-to-end data-science systems rather than isolated model training. My experience includes anomaly detection, explainability, simulation/RL and LLM systems. For banking specifically, I've built public synthetic reference projects around risk modeling, temporal validation, calibration, expected-loss-aware decision thresholds, human review, Spark feature engineering, data pipelines and GCP/Vertex deployment patterns. I'm slightly below the four-year requirement in full-time experience, so I don't try to disguise that; I applied because the technical scope matches the kind of problems I've already been building toward and I can demonstrate the lifecycle end to end.
