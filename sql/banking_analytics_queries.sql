-- ============================================================
-- Banking Transaction Intelligence Model
-- SQL Query Library — Banking Analytics & Fraud Detection
-- Professional SQL with CTEs, Window Functions, CASE WHEN
-- ============================================================

-- ─────────────────────────────────────────────────────────────
-- 1. MONTHLY TRANSACTION VOLUME & P&L SUMMARY
-- ─────────────────────────────────────────────────────────────

WITH monthly_base AS (
    SELECT
        DATE_TRUNC('month', transaction_date)   AS month_start,
        TO_CHAR(transaction_date, 'YYYY-MM')    AS month_year,
        COUNT(*)                                 AS total_transactions,
        SUM(transaction_amount)                  AS gross_transaction_value,
        SUM(fee_income)                          AS total_fee_income,
        SUM(interchange_income)                  AS total_interchange,
        SUM(processing_cost)                     AS total_processing_cost,
        SUM(fraud_loss)                          AS total_fraud_loss,
        SUM(chargeback_loss)                     AS total_chargeback_loss,
        SUM(refund_loss)                         AS total_refund_loss,
        SUM(net_revenue)                         AS total_net_revenue,
        SUM(net_pnl_impact)                      AS total_net_pnl,
        SUM(fraud_flag)                          AS fraud_count,
        SUM(chargeback_flag)                     AS chargeback_count
    FROM banking_transactions
    WHERE transaction_status <> 'Pending'
    GROUP BY 1, 2
),
monthly_with_variance AS (
    SELECT
        *,
        LAG(total_net_pnl)      OVER (ORDER BY month_start) AS prior_month_pnl,
        LAG(total_fraud_loss)   OVER (ORDER BY month_start) AS prior_month_fraud,
        LAG(total_net_revenue)  OVER (ORDER BY month_start) AS prior_month_revenue,
        ROUND(
            (total_net_pnl - LAG(total_net_pnl) OVER (ORDER BY month_start))
            / NULLIF(ABS(LAG(total_net_pnl) OVER (ORDER BY month_start)), 0) * 100, 2
        ) AS net_pnl_mom_variance_pct,
        ROUND(
            (total_fraud_loss - LAG(total_fraud_loss) OVER (ORDER BY month_start))
            / NULLIF(ABS(LAG(total_fraud_loss) OVER (ORDER BY month_start)), 0) * 100, 2
        ) AS fraud_loss_mom_pct
    FROM monthly_base
)
SELECT
    month_year,
    total_transactions,
    ROUND(gross_transaction_value, 2)    AS gross_tv,
    ROUND(total_net_revenue, 2)          AS net_revenue,
    ROUND(total_net_pnl, 2)             AS net_pnl,
    ROUND(total_fraud_loss, 2)           AS fraud_loss,
    ROUND(total_chargeback_loss, 2)      AS chargeback_loss,
    fraud_count,
    ROUND(fraud_count::NUMERIC / total_transactions * 100, 2) AS fraud_rate_pct,
    net_pnl_mom_variance_pct,
    fraud_loss_mom_pct
FROM monthly_with_variance
ORDER BY month_start;


-- ─────────────────────────────────────────────────────────────
-- 2. SUSPICIOUS TRANSACTION DETECTION — MULTI-RULE LOGIC
-- ─────────────────────────────────────────────────────────────

WITH rule_signals AS (
    SELECT
        t.*,
        -- R01: High value vs historical average
        CASE WHEN t.transaction_amount >= t.historical_average_transaction_amount * 5
             THEN 1 ELSE 0 END                                     AS r01_high_value,
        -- R02: Velocity spike (daily count > 10)
        CASE WHEN COUNT(*) OVER (
                PARTITION BY t.customer_id, t.transaction_date
             ) > 10 THEN 1 ELSE 0 END                             AS r02_velocity,
        -- R03: Failed authentication
        CASE WHEN t.failed_attempt_count >= 3 THEN 1 ELSE 0 END   AS r03_failed_auth,
        -- R04: Multiple logins
        CASE WHEN t.login_attempts >= 4 THEN 1 ELSE 0 END         AS r04_logins,
        -- R05: Off-hours transaction
        CASE WHEN EXTRACT(HOUR FROM t.transaction_time::TIME) < 6
             THEN 1 ELSE 0 END                                     AS r05_off_hours,
        -- R10: Z-score outlier (approximation via segment average)
        CASE WHEN t.transaction_amount > 3.5 *
                STDDEV(t.transaction_amount) OVER (
                    PARTITION BY t.customer_segment
                ) + AVG(t.transaction_amount) OVER (
                    PARTITION BY t.customer_segment
                ) THEN 1 ELSE 0 END                                AS r10_z_score,
        -- R11: Balance inconsistency
        CASE WHEN t.debit_credit_flag = 'Debit'
              AND t.account_balance_after < 0 THEN 1 ELSE 0 END   AS r11_balance,
        -- R15: Duplicate transaction
        CASE WHEN COUNT(*) OVER (
                PARTITION BY t.customer_id, t.transaction_amount,
                             t.merchant_name, t.transaction_date
             ) > 1 THEN 1 ELSE 0 END                              AS r15_duplicate,
        -- R17: Cross-border suspicious
        CASE WHEN t.merchant_category IN (
                'Crypto Exchanges', 'Financial Services / Money Transfer',
                'Gaming & Gambling'
             ) AND t.risk_score > 50 AND t.transaction_amount > 1000
             THEN 1 ELSE 0 END                                     AS r17_cross_border
    FROM banking_transactions t
    WHERE t.transaction_status NOT IN ('Failed', 'Declined')
),
flagged AS (
    SELECT
        *,
        (r01_high_value + r02_velocity + r03_failed_auth + r04_logins +
         r05_off_hours + r10_z_score + r11_balance + r15_duplicate +
         r17_cross_border)                                         AS rules_triggered,
        CASE
            WHEN risk_score >= 80 OR
                 (r01_high_value + r02_velocity + r03_failed_auth +
                  r04_logins + r05_off_hours + r10_z_score + r11_balance +
                  r15_duplicate + r17_cross_border) >= 3
            THEN 'SUSPICIOUS'
            ELSE 'NORMAL'
        END                                                        AS transaction_flag
    FROM rule_signals
)
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    transaction_amount,
    transaction_type,
    channel,
    merchant_category,
    risk_score,
    rules_triggered,
    transaction_flag,
    net_pnl_impact,
    fraud_loss,
    chargeback_loss
FROM flagged
WHERE transaction_flag = 'SUSPICIOUS'
ORDER BY rules_triggered DESC, risk_score DESC
LIMIT 500;


-- ─────────────────────────────────────────────────────────────
-- 3. HIGH-VALUE TRANSACTION OUTLIERS (SEGMENT-RELATIVE)
-- ─────────────────────────────────────────────────────────────

WITH segment_stats AS (
    SELECT
        customer_segment,
        AVG(transaction_amount)    AS seg_mean,
        STDDEV(transaction_amount) AS seg_std,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY transaction_amount)
                                   AS seg_p95
    FROM banking_transactions
    GROUP BY customer_segment
),
outliers AS (
    SELECT
        t.transaction_id,
        t.customer_id,
        t.transaction_date,
        t.transaction_amount,
        t.customer_segment,
        t.channel,
        t.merchant_category,
        t.fraud_flag,
        t.risk_score,
        s.seg_mean,
        s.seg_std,
        s.seg_p95,
        ROUND((t.transaction_amount - s.seg_mean) / NULLIF(s.seg_std, 0), 4) AS z_score,
        ROUND(t.transaction_amount / NULLIF(s.seg_mean, 0), 2)                AS ratio_to_avg
    FROM banking_transactions t
    JOIN segment_stats s ON t.customer_segment = s.customer_segment
    WHERE t.transaction_amount > s.seg_mean + 3.5 * NULLIF(s.seg_std, 0)
)
SELECT *
FROM outliers
ORDER BY z_score DESC
LIMIT 200;


-- ─────────────────────────────────────────────────────────────
-- 4. CHARGEBACK-HEAVY MERCHANTS — RISK RANKING
-- ─────────────────────────────────────────────────────────────

WITH merchant_stats AS (
    SELECT
        merchant_name,
        merchant_category,
        COUNT(*)                         AS total_transactions,
        SUM(chargeback_flag)             AS chargeback_count,
        SUM(chargeback_loss)             AS total_chargeback_loss,
        SUM(fraud_flag)                  AS fraud_count,
        SUM(fraud_loss)                  AS total_fraud_loss,
        SUM(refund_flag)                 AS refund_count,
        SUM(refund_loss)                 AS total_refund_loss,
        SUM(transaction_amount)          AS gross_tv,
        SUM(net_pnl_impact)              AS net_pnl,
        AVG(risk_score)                  AS avg_risk_score
    FROM banking_transactions
    GROUP BY merchant_name, merchant_category
    HAVING COUNT(*) >= 10
),
ranked AS (
    SELECT
        *,
        ROUND(chargeback_count::NUMERIC / total_transactions * 100, 2)  AS chargeback_rate_pct,
        ROUND(fraud_count::NUMERIC / total_transactions * 100, 2)        AS fraud_rate_pct,
        ROUND(total_chargeback_loss / NULLIF(gross_tv, 0) * 100, 4)     AS cb_loss_pct_of_tv,
        RANK() OVER (ORDER BY total_chargeback_loss DESC)                AS cb_loss_rank,
        RANK() OVER (ORDER BY chargeback_count::NUMERIC /
                     total_transactions DESC)                             AS cb_rate_rank,
        NTILE(5) OVER (ORDER BY avg_risk_score DESC)                     AS risk_quintile
    FROM merchant_stats
)
SELECT
    merchant_name,
    merchant_category,
    total_transactions,
    chargeback_count,
    chargeback_rate_pct,
    ROUND(total_chargeback_loss, 2)  AS chargeback_loss,
    fraud_rate_pct,
    ROUND(total_fraud_loss, 2)        AS fraud_loss,
    ROUND(net_pnl, 2)                AS net_pnl,
    ROUND(avg_risk_score, 1)         AS avg_risk_score,
    cb_loss_rank,
    CASE risk_quintile
        WHEN 1 THEN 'CRITICAL'
        WHEN 2 THEN 'HIGH'
        WHEN 3 THEN 'MEDIUM'
        WHEN 4 THEN 'LOW'
        ELSE 'MINIMAL'
    END                              AS merchant_risk_tier
FROM ranked
WHERE chargeback_count > 0
ORDER BY total_chargeback_loss DESC
LIMIT 50;


-- ─────────────────────────────────────────────────────────────
-- 5. REVENUE LEAKAGE DETECTION
-- ─────────────────────────────────────────────────────────────

WITH leakage_base AS (
    SELECT
        channel,
        customer_segment,
        merchant_category,
        DATE_TRUNC('month', transaction_date)  AS month_start,
        SUM(fee_income)                         AS fee_income,
        SUM(interchange_income)                 AS interchange,
        SUM(processing_cost)                    AS processing_cost,
        SUM(refund_loss)                        AS refund_loss,
        SUM(chargeback_loss)                    AS chargeback_loss,
        SUM(fraud_loss)                         AS fraud_loss,
        SUM(net_revenue)                        AS net_revenue,
        SUM(net_pnl_impact)                     AS net_pnl,
        COUNT(*)                                AS tx_count
    FROM banking_transactions
    GROUP BY 1, 2, 3, 4
),
with_leakage AS (
    SELECT
        *,
        (fee_income + interchange)               AS gross_income,
        GREATEST(0, -(net_pnl))                  AS pnl_leakage,
        ROUND(GREATEST(0, -(net_pnl))
              / NULLIF(fee_income + interchange, 0) * 100, 2) AS leakage_pct,
        CASE
            WHEN net_pnl < 0 AND fee_income > 0 THEN 'LEAKING'
            WHEN net_pnl < 0 AND fee_income = 0 THEN 'LOSS-MAKING'
            ELSE 'PROFITABLE'
        END                                      AS profitability_status
    FROM leakage_base
)
SELECT *
FROM with_leakage
WHERE profitability_status IN ('LEAKING', 'LOSS-MAKING')
ORDER BY pnl_leakage DESC
LIMIT 100;


-- ─────────────────────────────────────────────────────────────
-- 6. CUSTOMER-LEVEL RISK SCORE & PROFILING
-- ─────────────────────────────────────────────────────────────

WITH customer_profile AS (
    SELECT
        customer_id,
        customer_segment,
        customer_age_band,
        geography,
        COUNT(*)                                          AS total_transactions,
        SUM(transaction_amount)                           AS total_spend,
        AVG(transaction_amount)                           AS avg_transaction,
        MAX(transaction_amount)                           AS max_transaction,
        SUM(fraud_flag)                                   AS fraud_count,
        SUM(chargeback_flag)                              AS chargeback_count,
        SUM(refund_flag)                                  AS refund_count,
        SUM(reversal_flag)                                AS reversal_count,
        SUM(fraud_loss)                                   AS total_fraud_loss,
        SUM(chargeback_loss)                              AS total_chargeback_loss,
        SUM(net_revenue)                                  AS customer_revenue,
        SUM(net_pnl_impact)                               AS customer_pnl,
        AVG(risk_score)                                   AS avg_risk_score,
        MAX(risk_score)                                   AS max_risk_score,
        SUM(failed_attempt_count)                         AS total_failed_attempts,
        SUM(login_attempts)                               AS total_login_attempts,
        MAX(transaction_date)                             AS last_transaction_date,
        MIN(transaction_date)                             AS first_transaction_date
    FROM banking_transactions
    GROUP BY 1, 2, 3, 4
),
risk_scored AS (
    SELECT
        *,
        ROUND(fraud_count::NUMERIC / total_transactions * 100, 2)        AS customer_fraud_rate,
        ROUND(chargeback_count::NUMERIC / total_transactions * 100, 2)   AS customer_cb_rate,
        ROUND(customer_pnl / NULLIF(total_transactions, 0), 4)           AS pnl_per_transaction,
        -- Composite customer risk score
        LEAST(100, ROUND(
            avg_risk_score * 0.30 +
            LEAST(50, fraud_count * 10) * 0.30 +
            LEAST(30, chargeback_count * 5) * 0.20 +
            LEAST(20, total_failed_attempts * 2) * 0.20
        , 1))                                                             AS composite_risk_score,
        NTILE(10) OVER (ORDER BY avg_risk_score DESC)                    AS risk_decile
    FROM customer_profile
)
SELECT
    customer_id,
    customer_segment,
    customer_age_band,
    geography,
    total_transactions,
    ROUND(total_spend, 2)             AS total_spend,
    ROUND(avg_transaction, 2)         AS avg_transaction,
    fraud_count,
    customer_fraud_rate               AS fraud_rate_pct,
    chargeback_count,
    ROUND(total_fraud_loss, 2)        AS fraud_loss,
    ROUND(customer_revenue, 2)        AS customer_revenue,
    ROUND(customer_pnl, 2)           AS customer_pnl,
    ROUND(avg_risk_score, 1)         AS avg_risk_score,
    composite_risk_score,
    risk_decile,
    CASE
        WHEN composite_risk_score >= 80 THEN 'CRITICAL'
        WHEN composite_risk_score >= 60 THEN 'HIGH'
        WHEN composite_risk_score >= 40 THEN 'MEDIUM'
        WHEN composite_risk_score >= 20 THEN 'LOW'
        ELSE 'MINIMAL'
    END                               AS risk_tier
FROM risk_scored
ORDER BY composite_risk_score DESC
LIMIT 200;


-- ─────────────────────────────────────────────────────────────
-- 7. CHANNEL PROFITABILITY ANALYSIS
-- ─────────────────────────────────────────────────────────────

SELECT
    channel,
    branch_or_digital_flag,
    COUNT(*)                                            AS transactions,
    ROUND(SUM(transaction_amount), 2)                  AS gross_tv,
    ROUND(SUM(fee_income), 2)                          AS fee_income,
    ROUND(SUM(interchange_income), 2)                  AS interchange,
    ROUND(SUM(processing_cost), 2)                     AS processing_cost,
    ROUND(SUM(fraud_loss), 2)                          AS fraud_loss,
    ROUND(SUM(chargeback_loss), 2)                     AS chargeback_loss,
    ROUND(SUM(refund_loss), 2)                         AS refund_loss,
    ROUND(SUM(net_revenue), 2)                         AS net_revenue,
    ROUND(SUM(net_pnl_impact), 2)                      AS net_pnl,
    ROUND(AVG(risk_score), 1)                          AS avg_risk_score,
    SUM(fraud_flag)                                     AS fraud_count,
    ROUND(SUM(fraud_flag)::NUMERIC / COUNT(*) * 100, 2) AS fraud_rate_pct,
    ROUND(SUM(net_pnl_impact) / COUNT(*), 4)           AS pnl_per_transaction,
    ROUND(SUM(processing_cost) / NULLIF(SUM(fee_income + interchange_income), 0) * 100, 2)
                                                        AS cost_to_income_ratio
FROM banking_transactions
GROUP BY channel, branch_or_digital_flag
ORDER BY net_pnl DESC;


-- ─────────────────────────────────────────────────────────────
-- 8. DUPLICATE TRANSACTION DETECTION
-- ─────────────────────────────────────────────────────────────

WITH potential_duplicates AS (
    SELECT
        transaction_id,
        customer_id,
        account_id,
        transaction_date,
        transaction_amount,
        merchant_name,
        transaction_type,
        channel,
        fraud_flag,
        risk_score,
        net_pnl_impact,
        COUNT(*) OVER (
            PARTITION BY customer_id, transaction_amount,
                         merchant_name, transaction_date
        ) AS duplicate_group_size,
        ROW_NUMBER() OVER (
            PARTITION BY customer_id, transaction_amount,
                         merchant_name, transaction_date
            ORDER BY transaction_id
        ) AS row_num
    FROM banking_transactions
)
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    transaction_amount,
    merchant_name,
    transaction_type,
    channel,
    duplicate_group_size,
    row_num,
    CASE WHEN row_num > 1 THEN 'POTENTIAL DUPLICATE' ELSE 'ORIGINAL' END AS duplicate_status,
    risk_score,
    net_pnl_impact
FROM potential_duplicates
WHERE duplicate_group_size > 1
ORDER BY duplicate_group_size DESC, customer_id, transaction_date;


-- ─────────────────────────────────────────────────────────────
-- 9. FAILED TRANSACTION ANALYSIS
-- ─────────────────────────────────────────────────────────────

WITH failed_analysis AS (
    SELECT
        channel,
        customer_segment,
        merchant_category,
        authorization_method,
        transaction_type,
        COUNT(*) FILTER (WHERE transaction_status = 'Failed')   AS failed_count,
        COUNT(*) FILTER (WHERE transaction_status = 'Declined') AS declined_count,
        COUNT(*)                                                  AS total_count,
        ROUND(AVG(failed_attempt_count), 2)                      AS avg_failed_attempts,
        SUM(transaction_amount) FILTER (WHERE transaction_status IN ('Failed', 'Declined'))
                                                                  AS failed_value,
        SUM(processing_cost) FILTER (WHERE transaction_status IN ('Failed', 'Declined'))
                                                                  AS wasted_processing_cost
    FROM banking_transactions
    GROUP BY 1, 2, 3, 4, 5
)
SELECT
    channel,
    customer_segment,
    merchant_category,
    authorization_method,
    total_count,
    failed_count,
    declined_count,
    (failed_count + declined_count)                              AS total_failed_declined,
    ROUND((failed_count + declined_count)::NUMERIC / total_count * 100, 2) AS failure_rate_pct,
    avg_failed_attempts,
    ROUND(failed_value, 2)                                        AS failed_tx_value,
    ROUND(wasted_processing_cost, 2)                              AS wasted_cost
FROM failed_analysis
WHERE (failed_count + declined_count) > 0
ORDER BY total_failed_declined DESC
LIMIT 50;


-- ─────────────────────────────────────────────────────────────
-- 10. REFUND & REVERSAL ANOMALY DETECTION
-- ─────────────────────────────────────────────────────────────

WITH refund_base AS (
    SELECT
        customer_id,
        merchant_name,
        merchant_category,
        DATE_TRUNC('month', transaction_date) AS month_start,
        COUNT(*) FILTER (WHERE refund_flag = 1)    AS refund_count,
        COUNT(*) FILTER (WHERE reversal_flag = 1)  AS reversal_count,
        COUNT(*)                                    AS total_count,
        SUM(refund_loss)                            AS total_refund_loss,
        SUM(transaction_amount) FILTER (WHERE refund_flag = 1) AS refunded_value,
        SUM(transaction_amount)                     AS total_value,
        AVG(transaction_amount) FILTER (WHERE refund_flag = 1) AS avg_refund_amount,
        AVG(historical_average_transaction_amount)  AS avg_historical_amount
    FROM banking_transactions
    GROUP BY 1, 2, 3, 4
),
anomalous AS (
    SELECT
        *,
        ROUND(refund_count::NUMERIC / NULLIF(total_count, 0) * 100, 2)  AS refund_rate_pct,
        ROUND(reversal_count::NUMERIC / NULLIF(total_count, 0) * 100, 2) AS reversal_rate_pct,
        ROUND(avg_refund_amount / NULLIF(avg_historical_amount, 0), 2)  AS refund_vs_avg_ratio,
        CASE
            WHEN refund_count::NUMERIC / NULLIF(total_count, 0) > 0.30
             AND avg_refund_amount > avg_historical_amount * 1.5
            THEN 'HIGH RISK'
            WHEN refund_count::NUMERIC / NULLIF(total_count, 0) > 0.15
            THEN 'ELEVATED'
            ELSE 'NORMAL'
        END AS refund_anomaly_flag
    FROM refund_base
    WHERE refund_count > 0
)
SELECT
    customer_id,
    merchant_name,
    merchant_category,
    TO_CHAR(month_start, 'YYYY-MM')          AS month,
    total_count,
    refund_count,
    refund_rate_pct,
    reversal_count,
    ROUND(total_refund_loss, 2)               AS refund_loss,
    refund_vs_avg_ratio,
    refund_anomaly_flag
FROM anomalous
WHERE refund_anomaly_flag IN ('HIGH RISK', 'ELEVATED')
ORDER BY refund_loss DESC
LIMIT 100;


-- ─────────────────────────────────────────────────────────────
-- 11. EXCEPTION QUEUE — INVESTIGATION PRIORITY QUEUE
-- ─────────────────────────────────────────────────────────────

WITH exception_base AS (
    SELECT
        t.transaction_id,
        t.customer_id,
        t.account_id,
        t.transaction_date,
        t.transaction_amount,
        t.transaction_type,
        t.channel,
        t.merchant_category,
        t.merchant_name,
        t.customer_segment,
        t.geography,
        t.fraud_flag,
        t.fraud_type,
        t.risk_score,
        t.chargeback_flag,
        t.chargeback_loss,
        t.refund_flag,
        t.refund_loss,
        t.reversal_flag,
        t.failed_attempt_count,
        t.login_attempts,
        t.net_pnl_impact,
        t.fraud_loss,
        -- Velocity count
        COUNT(*) OVER (
            PARTITION BY t.customer_id, t.transaction_date
        ) AS daily_tx_count,
        -- Amount ratio
        ROUND(t.transaction_amount / NULLIF(t.historical_average_transaction_amount, 0), 2)
             AS amount_vs_avg,
        -- Composite risk (simplified)
        LEAST(100, ROUND(
            t.risk_score * 0.50 +
            t.failed_attempt_count * 5 +
            t.chargeback_flag * 20 +
            CASE WHEN t.transaction_amount > t.historical_average_transaction_amount * 5
                 THEN 15 ELSE 0 END
        , 1))                  AS computed_risk
    FROM banking_transactions t
    WHERE t.transaction_status NOT IN ('Pending', 'Failed')
),
tiered AS (
    SELECT
        *,
        CASE
            WHEN computed_risk >= 85 THEN 'CRITICAL'
            WHEN computed_risk >= 70 THEN 'HIGH'
            WHEN computed_risk >= 50 THEN 'MEDIUM'
            ELSE 'LOW'
        END AS investigation_priority
    FROM exception_base
    WHERE
        fraud_flag = 1
        OR chargeback_flag = 1
        OR risk_score >= 60
        OR amount_vs_avg >= 5
        OR failed_attempt_count >= 3
        OR daily_tx_count > 10
)
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    transaction_amount,
    merchant_category,
    merchant_name,
    channel,
    customer_segment,
    fraud_flag,
    fraud_type,
    risk_score,
    chargeback_flag,
    failed_attempt_count,
    amount_vs_avg,
    computed_risk,
    investigation_priority,
    ROUND(net_pnl_impact, 2) AS net_pnl_impact,
    ROUND(fraud_loss, 2)     AS fraud_loss,
    ROUND(chargeback_loss, 2) AS chargeback_loss
FROM tiered
ORDER BY computed_risk DESC, fraud_flag DESC
LIMIT 500;


-- ─────────────────────────────────────────────────────────────
-- 12. CUSTOMER SEGMENT PROFITABILITY
-- ─────────────────────────────────────────────────────────────

SELECT
    customer_segment,
    COUNT(DISTINCT customer_id)                                       AS unique_customers,
    COUNT(*)                                                           AS total_transactions,
    ROUND(SUM(transaction_amount), 2)                                 AS gross_tv,
    ROUND(AVG(transaction_amount), 2)                                 AS avg_transaction,
    ROUND(SUM(fee_income), 2)                                         AS fee_income,
    ROUND(SUM(interchange_income), 2)                                 AS interchange,
    ROUND(SUM(processing_cost), 2)                                    AS processing_cost,
    ROUND(SUM(fraud_loss), 2)                                         AS fraud_loss,
    ROUND(SUM(chargeback_loss), 2)                                    AS chargeback_loss,
    ROUND(SUM(net_revenue), 2)                                        AS net_revenue,
    ROUND(SUM(net_pnl_impact), 2)                                     AS net_pnl,
    ROUND(SUM(net_pnl_impact) / COUNT(*), 4)                          AS pnl_per_tx,
    SUM(fraud_flag)                                                    AS fraud_count,
    ROUND(SUM(fraud_flag)::NUMERIC / COUNT(*) * 100, 2)               AS fraud_rate_pct,
    ROUND(AVG(risk_score), 1)                                         AS avg_risk_score,
    ROUND(SUM(processing_cost) /
          NULLIF(SUM(fee_income + interchange_income), 0) * 100, 2)  AS cost_to_income_ratio
FROM banking_transactions
GROUP BY customer_segment
ORDER BY net_pnl DESC;


-- ─────────────────────────────────────────────────────────────
-- 13. DEVICE & GEOGRAPHY ANOMALY DETECTION
-- ─────────────────────────────────────────────────────────────

WITH customer_normal AS (
    SELECT
        customer_id,
        MODE() WITHIN GROUP (ORDER BY device_id)   AS typical_device,
        MODE() WITHIN GROUP (ORDER BY geography)   AS typical_geography,
        COUNT(DISTINCT device_id)                   AS distinct_devices,
        COUNT(DISTINCT ip_location)                 AS distinct_ips,
        COUNT(DISTINCT geography)                   AS distinct_geographies
    FROM banking_transactions
    GROUP BY customer_id
),
anomalous_sessions AS (
    SELECT
        t.transaction_id,
        t.customer_id,
        t.transaction_date,
        t.transaction_amount,
        t.channel,
        t.device_id,
        t.ip_location,
        t.geography,
        t.risk_score,
        t.fraud_flag,
        cn.typical_device,
        cn.typical_geography,
        cn.distinct_devices,
        cn.distinct_geographies,
        CASE WHEN t.device_id <> cn.typical_device     THEN 1 ELSE 0 END AS device_mismatch,
        CASE WHEN t.geography <> cn.typical_geography  THEN 1 ELSE 0 END AS geo_mismatch
    FROM banking_transactions t
    JOIN customer_normal cn ON t.customer_id = cn.customer_id
    WHERE t.transaction_amount > 500
)
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    transaction_amount,
    channel,
    geography,
    typical_geography,
    device_id,
    typical_device,
    device_mismatch,
    geo_mismatch,
    distinct_devices,
    distinct_geographies,
    risk_score,
    fraud_flag,
    (device_mismatch + geo_mismatch) AS anomaly_signals
FROM anomalous_sessions
WHERE device_mismatch = 1 OR geo_mismatch = 1
ORDER BY anomaly_signals DESC, risk_score DESC
LIMIT 200;
