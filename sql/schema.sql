CREATE TABLE IF NOT EXISTS customers (
  customer_id BIGSERIAL PRIMARY KEY,
  external_id TEXT UNIQUE NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  country_code CHAR(2),
  segment TEXT
);

CREATE TABLE IF NOT EXISTS transactions (
  transaction_id BIGSERIAL PRIMARY KEY,
  customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
  event_time TIMESTAMPTZ NOT NULL,
  amount NUMERIC(14,2) NOT NULL,
  merchant_category TEXT,
  channel TEXT,
  country_code CHAR(2),
  is_chargeback BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS risk_scores (
  score_id BIGSERIAL PRIMARY KEY,
  customer_id BIGINT NOT NULL REFERENCES customers(customer_id),
  scored_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  model_version TEXT NOT NULL,
  probability DOUBLE PRECISION NOT NULL CHECK (probability BETWEEN 0 AND 1),
  decision TEXT NOT NULL,
  features JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_transactions_customer_time ON transactions(customer_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_transactions_event_time ON transactions(event_time DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_customer_time ON risk_scores(customer_id, scored_at DESC);
CREATE INDEX IF NOT EXISTS idx_risk_scores_features_gin ON risk_scores USING GIN(features);

CREATE MATERIALIZED VIEW IF NOT EXISTS customer_30d_features AS
SELECT c.customer_id,
       COUNT(t.transaction_id) FILTER (WHERE t.event_time >= NOW() - INTERVAL '30 days') AS txn_count_30d,
       COALESCE(SUM(t.amount) FILTER (WHERE t.event_time >= NOW() - INTERVAL '30 days'), 0) AS spend_30d,
       COALESCE(AVG(t.amount) FILTER (WHERE t.event_time >= NOW() - INTERVAL '30 days'), 0) AS avg_ticket_30d,
       COUNT(*) FILTER (WHERE t.is_chargeback AND t.event_time >= NOW() - INTERVAL '90 days') AS chargebacks_90d
FROM customers c
LEFT JOIN transactions t ON t.customer_id = c.customer_id
GROUP BY c.customer_id;
