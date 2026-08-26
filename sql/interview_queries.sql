-- SQL interview practice for banking / fintech data science.
-- PostgreSQL-style syntax; adapt date functions for BigQuery/Spark SQL as needed.

-- 1) Top customers by 30-day transaction volume.
WITH recent AS (
    SELECT *
    FROM transactions
    WHERE event_time >= CURRENT_TIMESTAMP - INTERVAL '30 days'
)
SELECT
    customer_id,
    COUNT(*) AS transaction_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS average_amount
FROM recent
GROUP BY customer_id
ORDER BY total_amount DESC
LIMIT 20;


-- 2) Running balance per account using a window function.
SELECT
    account_id,
    event_time,
    transaction_id,
    signed_amount,
    SUM(signed_amount) OVER (
        PARTITION BY account_id
        ORDER BY event_time, transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_balance
FROM account_ledger;


-- 3) Seven-day rolling transaction volume excluding the current transaction.
SELECT
    customer_id,
    transaction_id,
    event_time,
    amount,
    SUM(amount) OVER (
        PARTITION BY customer_id
        ORDER BY event_time
        RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
    ) AS prior_7d_amount
FROM transactions;


-- 4) Find customers whose daily spend is more than 3x their trailing-30-day daily average.
WITH daily AS (
    SELECT
        customer_id,
        event_time::date AS event_date,
        SUM(amount) AS daily_spend
    FROM transactions
    GROUP BY customer_id, event_time::date
),
with_baseline AS (
    SELECT
        *,
        AVG(daily_spend) OVER (
            PARTITION BY customer_id
            ORDER BY event_date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS prior_30d_daily_avg
    FROM daily
)
SELECT *
FROM with_baseline
WHERE prior_30d_daily_avg IS NOT NULL
  AND daily_spend > 3 * prior_30d_daily_avg
ORDER BY daily_spend / NULLIF(prior_30d_daily_avg, 0) DESC;


-- 5) Latest application per customer.
WITH ranked AS (
    SELECT
        a.*,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id
            ORDER BY application_time DESC, application_id DESC
        ) AS rn
    FROM applications AS a
)
SELECT *
FROM ranked
WHERE rn = 1;


-- 6) Customers who transacted but never submitted an application.
SELECT DISTINCT t.customer_id
FROM transactions AS t
LEFT JOIN applications AS a
  ON a.customer_id = t.customer_id
WHERE a.customer_id IS NULL;


-- 7) Cohort default rate by application month.
SELECT
    DATE_TRUNC('month', application_time) AS cohort_month,
    COUNT(*) AS applications,
    AVG(default_label::DOUBLE PRECISION) AS default_rate
FROM applications
WHERE default_label IS NOT NULL
GROUP BY DATE_TRUNC('month', application_time)
ORDER BY cohort_month;


-- 8) Approval rate and observed bad rate by model version.
SELECT
    d.model_version,
    COUNT(*) AS decisions,
    AVG((d.decision_route = 'auto_approve')::INT) AS approval_rate,
    AVG((d.decision_route = 'human_review')::INT) AS review_rate,
    AVG(o.default_label::DOUBLE PRECISION) FILTER (
        WHERE d.decision_route = 'auto_approve'
          AND o.default_label IS NOT NULL
    ) AS bad_rate_approved
FROM decisions AS d
LEFT JOIN outcomes AS o USING (application_id)
GROUP BY d.model_version
ORDER BY d.model_version;


-- 9) Compare human vs model decisions and useful overrides.
SELECT
    COUNT(*) AS reviewed_cases,
    AVG((r.human_decision <> r.model_decision)::INT) AS override_rate,
    AVG((r.human_decision = o.default_label)::INT) AS human_accuracy,
    AVG((r.model_decision = o.default_label)::INT) AS model_accuracy,
    AVG((
        r.human_decision <> r.model_decision
        AND r.human_decision = o.default_label
    )::INT) AS useful_override_rate
FROM reviews AS r
JOIN outcomes AS o USING (application_id)
WHERE o.default_label IS NOT NULL;


-- 10) Detect duplicate transaction IDs.
SELECT
    transaction_id,
    COUNT(*) AS copies
FROM transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1
ORDER BY copies DESC;


-- 11) Customer transaction gaps using LAG.
WITH ordered AS (
    SELECT
        customer_id,
        transaction_id,
        event_time,
        LAG(event_time) OVER (
            PARTITION BY customer_id
            ORDER BY event_time
        ) AS previous_event_time
    FROM transactions
)
SELECT
    *,
    event_time - previous_event_time AS time_since_previous
FROM ordered
WHERE previous_event_time IS NOT NULL;


-- 12) Merchant concentration per customer.
WITH merchant_spend AS (
    SELECT
        customer_id,
        merchant_id,
        SUM(amount) AS merchant_amount
    FROM transactions
    GROUP BY customer_id, merchant_id
),
total_spend AS (
    SELECT
        customer_id,
        SUM(merchant_amount) AS total_amount
    FROM merchant_spend
    GROUP BY customer_id
)
SELECT
    m.customer_id,
    MAX(m.merchant_amount / NULLIF(t.total_amount, 0)) AS largest_merchant_share
FROM merchant_spend AS m
JOIN total_spend AS t USING (customer_id)
GROUP BY m.customer_id;


-- 13) Rank risky applications within each country without collapsing rows.
SELECT
    application_id,
    country,
    default_probability,
    DENSE_RANK() OVER (
        PARTITION BY country
        ORDER BY default_probability DESC
    ) AS country_risk_rank
FROM scored_applications;


-- 14) Data-quality check: missingness rate per important feature.
SELECT
    COUNT(*) AS rows,
    AVG((income IS NULL)::INT) AS income_missing_rate,
    AVG((utilization IS NULL)::INT) AS utilization_missing_rate,
    AVG((employment IS NULL)::INT) AS employment_missing_rate
FROM applications;


-- 15) Why WHERE vs HAVING matters.
-- WHERE filters rows before aggregation; HAVING filters aggregated groups.
SELECT
    customer_id,
    SUM(amount) AS total_amount
FROM transactions
WHERE event_time >= CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY customer_id
HAVING SUM(amount) > 10000;


-- 16) Avoid accidental fan-out from many-to-many joins.
-- Pre-aggregate each many-side to one row per customer/application before joining.
WITH txn_features AS (
    SELECT customer_id, COUNT(*) AS txn_count
    FROM transactions
    GROUP BY customer_id
),
payment_features AS (
    SELECT customer_id, COUNT(*) AS payment_count
    FROM payments
    GROUP BY customer_id
)
SELECT
    c.customer_id,
    COALESCE(t.txn_count, 0) AS txn_count,
    COALESCE(p.payment_count, 0) AS payment_count
FROM customers AS c
LEFT JOIN txn_features AS t USING (customer_id)
LEFT JOIN payment_features AS p USING (customer_id);


-- 17) Percentile threshold for the riskiest 5%.
SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY default_probability) AS p95_score
FROM scored_applications;


-- 18) Applications above the daily 95th percentile risk score.
WITH scored AS (
    SELECT
        *,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY default_probability)
        OVER (PARTITION BY score_time::date) AS daily_p95
    FROM scored_applications
)
SELECT *
FROM scored
WHERE default_probability >= daily_p95;


-- 19) Month-over-month event-rate change.
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', application_time) AS month,
        AVG(default_label::DOUBLE PRECISION) AS event_rate
    FROM applications
    WHERE default_label IS NOT NULL
    GROUP BY DATE_TRUNC('month', application_time)
)
SELECT
    month,
    event_rate,
    LAG(event_rate) OVER (ORDER BY month) AS previous_event_rate,
    event_rate - LAG(event_rate) OVER (ORDER BY month) AS absolute_change
FROM monthly
ORDER BY month;


-- 20) Index discussion prompt, not a universal prescription:
-- For frequent customer-history lookups bounded by event time, an index such as
-- (customer_id, event_time) can support selective range scans.
-- Validate using EXPLAIN / EXPLAIN ANALYZE against actual workload and data distribution.
CREATE INDEX IF NOT EXISTS idx_transactions_customer_time
    ON transactions (customer_id, event_time);
