-- churn analysis SQL queries
-- used these to explore the data before building the Python model
-- and for the RFM segmentation that fed into Tableau


-- basic churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(churned) AS churned,
    ROUND(100.0 * SUM(churned) / COUNT(*), 2) AS churn_rate_pct
FROM retail_customers;


-- churn by customer segment
SELECT
    customer_segment,
    COUNT(*) AS customers,
    SUM(churned) AS churned_count,
    ROUND(100.0 * SUM(churned) / COUNT(*), 1) AS churn_rate_pct
FROM retail_customers
GROUP BY customer_segment
ORDER BY churn_rate_pct DESC;


-- RFM scoring
-- scoring recency inversely (fewer days = higher score)
WITH rfm AS (
    SELECT
        customer_id,
        CURRENT_DATE - MAX(order_date)::DATE AS recency_days,
        COUNT(DISTINCT order_id) AS frequency,
        SUM(order_value) AS monetary
    FROM orders
    GROUP BY customer_id
)
SELECT
    customer_id,
    recency_days,
    frequency,
    ROUND(monetary, 2) AS monetary,
    NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
    NTILE(5) OVER (ORDER BY frequency) AS f_score,
    NTILE(5) OVER (ORDER BY monetary) AS m_score
FROM rfm;


-- churn rate by RFM tier
-- built this to validate that the RFM scores actually correlated with churn
WITH rfm AS (
    SELECT
        customer_id,
        NTILE(5) OVER (ORDER BY (CURRENT_DATE - MAX(order_date)::DATE) DESC) AS r_score,
        NTILE(5) OVER (ORDER BY COUNT(DISTINCT order_id)) AS f_score,
        NTILE(5) OVER (ORDER BY SUM(order_value)) AS m_score
    FROM orders
    GROUP BY customer_id
),
tiered AS (
    SELECT
        r.customer_id,
        CASE
            WHEN (r_score + f_score + m_score) >= 12 THEN 'Champions'
            WHEN (r_score + f_score + m_score) >= 9  THEN 'Loyal'
            WHEN (r_score + f_score + m_score) >= 6  THEN 'At Risk'
            ELSE 'Lost'
        END AS rfm_tier
    FROM rfm r
)
SELECT
    t.rfm_tier,
    COUNT(*) AS customers,
    SUM(c.churned) AS churned,
    ROUND(100.0 * SUM(c.churned) / COUNT(*), 1) AS churn_rate_pct
FROM tiered t
JOIN retail_customers c USING (customer_id)
GROUP BY t.rfm_tier
ORDER BY churn_rate_pct DESC;


-- monthly cohort churn
WITH first_orders AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) AS cohort_month
    FROM orders
    GROUP BY customer_id
)
SELECT
    TO_CHAR(f.cohort_month, 'YYYY-MM') AS cohort,
    COUNT(*) AS cohort_size,
    SUM(c.churned) AS churned,
    ROUND(100.0 * SUM(c.churned) / COUNT(*), 1) AS churn_rate_pct
FROM first_orders f
JOIN retail_customers c USING (customer_id)
GROUP BY f.cohort_month
ORDER BY f.cohort_month;


-- high-value customers showing disengagement - for retention outreach
-- 90+ days no purchase, never officially churned, LTV > $500
SELECT
    c.customer_id,
    c.email,
    c.customer_segment,
    CURRENT_DATE - MAX(o.order_date)::DATE AS days_since_purchase,
    COUNT(DISTINCT o.order_id) AS total_orders,
    ROUND(SUM(o.order_value), 2) AS lifetime_value
FROM orders o
JOIN retail_customers c USING (customer_id)
WHERE c.churned = 0
GROUP BY c.customer_id, c.email, c.customer_segment
HAVING
    CURRENT_DATE - MAX(o.order_date)::DATE > 90
    AND SUM(o.order_value) > 500
ORDER BY lifetime_value DESC
LIMIT 500;
