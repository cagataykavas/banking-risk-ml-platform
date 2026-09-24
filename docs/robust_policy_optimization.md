# Robust two-threshold policy optimization

Optimizing approve/review/reject thresholds against one set of probabilities and one economic
assumption set can select a fragile policy. A small deterioration in probability of default (PD),
loss given default or review effectiveness may reverse the apparent winner.

`src.robust_policy` evaluates every fixed threshold pair across a declared scenario set. For each
scenario it:

1. applies a bounded log-odds shift and PD multiplier;
2. routes the stressed probabilities through the same fixed thresholds;
3. computes ex-ante value for approve, manual-review and reject decisions; and
4. enforces approval and review-capacity constraints.

A policy is feasible only when those operational constraints hold in every scenario. Among feasible
policies, the optimizer maximizes worst-case expected value per application. Deterministic tie
breakers prefer lower maximum scenario regret, higher mean value and then lower thresholds.

```python
from src.robust_policy import default_policy_scenarios, optimize_robust_policy

report = optimize_robust_policy(
    calibrated_pd,
    scenarios=default_policy_scenarios(),
)
if not report.accepted:
    raise RuntimeError(report.reason_codes)

selected = report.selected
print(selected.approve_below, selected.reject_above)
print(selected.worst_case_scenario, selected.worst_case_value_per_application)
```

The report is JSON-ready and retains scenario-level value and routing rates, the worst-case
scenario, maximum regret, and the number of evaluated and feasible policies.

## Trust boundary and limitations

Scenario definitions are governed inputs, not forecasts. The included defaults are synthetic
demonstration assumptions and are not regulatory parameters or recommendations. Production values
need Finance/Risk ownership, independent validation and versioning.

Expected value relies on calibrated PDs and transparent average economics. It does not model credit
limits, exposure at default, discounting, lifetime cash flows, competing risks, customer response,
selection effects, correlated defaults or capital constraints. A PD shift changes routing and
expected losses but does not simulate a full portfolio path. Maximin selection is deliberately
conservative and should be compared with scenario probabilities, regret limits and out-of-time
realized outcomes before deployment.
