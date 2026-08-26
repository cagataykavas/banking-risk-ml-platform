# Selection Bias, Reject Inference and Feedback Loops

A banking model is often trained on outcomes that are only observed for customers who passed an earlier policy. That means the labeled population may be **selected by the historical decision process** rather than representative of every future applicant.

This document is intentionally an interview/research note, not an implementation claiming to solve reject inference.

## Why the problem appears

Suppose the historical policy declined applications with very high utilization and many late payments. Those declined cases never became booked accounts, so their future default outcomes are not observed.

The training sample therefore contains:

```text
all applicants
     ↓
historical policy
     ↓
approved / booked population
     ↓
observed outcomes
```

A model trained only on the final box learns from a policy-filtered population.

## Consequences

- observed event rates can differ from the full applicant population;
- model performance measured only on booked customers may not transfer to policy expansions;
- changing the policy changes the data that will be observed later;
- comparisons between champion and challenger models can be biased if they operate on different selected populations;
- segment-level coverage matters as much as aggregate AUC.

## Common approaches discussed in banking

### 1. Do nothing, but be explicit

If approval rates are high and rejected applicants are not materially different, a team may decide the bias is tolerable for the use case. The key is documenting the assumption rather than pretending the sample is random.

### 2. Parceling / augmentation

Assign inferred outcomes to rejected cases based on score bands or observed event rates. This is simple but can reinforce assumptions from the existing model and should be sensitivity-tested.

### 3. Reweighting

Estimate the probability that an observation is accepted/observed and use inverse-probability-style weights. This requires a credible model of the historical selection mechanism and can become unstable for regions with very low acceptance probability.

### 4. Semi-supervised / model-based methods

Treat rejected outcomes as latent and iterate between outcome estimation and model fitting. These methods can look sophisticated while still being driven by unverifiable assumptions.

### 5. Controlled exploration

Where legally and operationally appropriate, a policy can deliberately approve a small randomized boundary sample to obtain less-selected labels. This can produce stronger evidence but involves real risk/cost and governance constraints.

## Interview answer structure

If asked *"How would you handle rejected applications with no labels?"*, a strong answer is:

1. identify that the issue is **sample-selection bias**, not ordinary missing-at-random labels;
2. quantify approval coverage and compare feature distributions between approved/rejected populations;
3. understand the historical policy that generated the selection;
4. evaluate whether reweighting or augmentation assumptions are defensible;
5. run sensitivity analyses rather than presenting one inferred-label method as truth;
6. validate on future cohorts after any policy change;
7. monitor feedback loops because the deployed model changes the data it later learns from.

## Related portfolio modules

- `src/temporal_validation.py` — future-cohort validation;
- `src/drift_monitoring.py` — population shift diagnostics;
- `src/segment_stability.py` — coverage/performance by country or segment;
- `src/champion_challenger.py` — guarded model promotion;
- `src/stress_testing.py` — policy robustness under deterioration scenarios.

## Important distinction

Reject inference is **not** the same as class imbalance.

SMOTE, class weights, or under-sampling can alter how observed positive/negative classes are learned. They do not magically recover unknown outcomes for a population excluded by prior policy.
