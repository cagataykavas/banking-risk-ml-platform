-- Synthetic portfolio example: leakage-aware banking risk feature mart.
-- Written in PostgreSQL-style SQL. Adapt timestamp/date functions for other warehouses.

WITH application_base AS (
    SELECT
        a.application_id,
        a.customer_id,
        a.application_time,
        a.requested_amount,
        a.income,
        a.employment,
        a.housing,
        a.default_label
    FROM applications AS a
    WHERE a.application_time IS NOT NULL
),

transaction_history AS (
    SELECT
        a.application_id,
        COUNT(t.transaction_id) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '30 days'
              AND t.event_time < a.application_time
        ) AS txn_count_30d,
        COALESCE(SUM(t.amount) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '30 days'
              AND t.event_time < a.application_time
        ), 0.0) AS spend_30d,
        COALESCE(AVG(t.amount) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '90 days'
              AND t.event_time < a.application_time
        ), 0.0) AS avg_ticket_90d,
        COALESCE(STDDEV_POP(t.amount) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '90 days'
              AND t.event_time < a.application_time
        ), 0.0) AS ticket_volatility_90d,
        COUNT(*) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '30 days'
              AND t.event_time < a.application_time
              AND t.country <> 'TR'
        ) AS cross_border_txn_30d,
        COUNT(*) FILTER (
            WHERE t.event_time >= a.application_time - INTERVAL '30 days'
              AND t.event_time < a.application_time
              AND EXTRACT(HOUR FROM t.event_time) BETWEEN 0 AND 5
        ) AS night_txn_30d
    FROM application_base AS a
    LEFT JOIN transactions AS t
      ON t.customer_id = a.customer_id
     AND t.event_time < a.application_time
     AND t.event_time >= a.application_time - INTERVAL '90 days'
    GROUP BY a.application_id
),

account_history AS (
    SELECT
        a.application_id,
        COUNT(p.payment_id) FILTER (
            WHERE p.due_date >= a.application_time::date - INTERVAL '365 days'
              AND p.due_date < a.application_time::date
        ) AS payments_due_12m,
        COUNT(p.payment_id) FILTER (
            WHERE p.due_date >= a.application_time::date - INTERVAL '365 days'
              AND p.due_date < a.application_time::date
              AND p.paid_date > p.due_date
        ) AS late_payments_12m,
        MAX(
            CASE
                WHEN p.due_date < a.application_time::date
                 AND p.paid_date > p.due_date
                THEN p.paid_date - p.due_date
                ELSE 0
            END
        ) AS max_days_late_12m
    FROM application_base AS a
    LEFT JOIN payments AS p
      ON p.customer_id = a.customer_id
     AND p.due_date < a.application_time::date
     AND p.due_date >= a.application_time::date - INTERVAL '365 days'
    GROUP BY a.application_id
),

customer_prior_applications AS (
    SELECT
        current.application_id,
        COUNT(prior.application_id) AS prior_applications,
        COUNT(prior.application_id) FILTER (
            WHERE prior.application_time >= current.application_time - INTERVAL '90 days'
        ) AS prior_applications_90d
    FROM application_base AS current
    LEFT JOIN application_base AS prior
      ON prior.customer_id = current.customer_id
     AND prior.application_time < current.application_time
    GROUP BY current.application_id
),

joined AS (
    SELECT
        a.application_id,
        a.customer_id,
        a.application_time,
        a.requested_amount,
        a.income,
        a.employment,
        a.housing,
        a.default_label,
        COALESCE(t.txn_count_30d, 0) AS txn_count_30d,
        COALESCE(t.spend_30d, 0.0) AS spend_30d,
        COALESCE(t.avg_ticket_90d, 0.0) AS avg_ticket_90d,
        COALESCE(t.ticket_volatility_90d, 0.0) AS ticket_volatility_90d,
        COALESCE(t.cross_border_txn_30d, 0) AS cross_border_txn_30d,
        COALESCE(t.night_txn_30d, 0) AS night_txn_30d,
        COALESCE(h.payments_due_12m, 0) AS payments_due_12m,
        COALESCE(h.late_payments_12m, 0) AS late_payments_12m,
        COALESCE(h.max_days_late_12m, 0) AS max_days_late_12m,
        COALESCE(p.prior_applications, 0) AS prior_applications,
        COALESCE(p.prior_applications_90d, 0) AS prior_applications_90d
    FROM application_base AS a
    LEFT JOIN transaction_history AS t USING (application_id)
    LEFT JOIN account_history AS h USING (application_id)
    LEFT JOIN customer_prior_applications AS p USING (application_id)
)

SELECT
    *,
    requested_amount / NULLIF(income, 0.0) AS requested_amount_to_income,
    spend_30d / NULLIF(income, 0.0) AS spend_to_income_30d,
    late_payments_12m::DOUBLE PRECISION / NULLIF(payments_due_12m, 0) AS late_payment_rate_12m,
    CASE
        WHEN prior_applications_90d >= 3 THEN 1
        ELSE 0
    END AS high_application_velocity
FROM joined;

-- Key leakage rule:
-- Every history join is constrained to timestamps strictly BEFORE application_time.
-- default_label is retained only as the training target; it must never enter feature transformations.
