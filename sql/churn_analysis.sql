CREATE DATABASE IF NOT EXISTS churn_project;
USE churn_project;

CREATE TABLE IF NOT EXISTS telco_churn (
    customerID VARCHAR(50) NOT NULL,
    tenure INT,
    Contract VARCHAR(50),
    MonthlyCharges DECIMAL(10, 2),
    TotalCharges DECIMAL(10, 2),
    Churn VARCHAR(10),
    PRIMARY KEY (customerID)
);

-- Query 1: Analyze churn rate by contract type to identify which contract groups have higher churn exposure.
SELECT
    Contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100,
        2
    ) AS churn_rate
FROM telco_churn
GROUP BY Contract
ORDER BY churn_rate DESC;

-- Query 2: Segment customers by tenure group to compare churn risk across customer lifecycle stages.
SELECT
    CASE
        WHEN tenure BETWEEN 0 AND 12 THEN '0-12 months'
        WHEN tenure BETWEEN 13 AND 48 THEN '13-48 months'
        ELSE '49+ months'
    END AS tenure_segment,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100,
        2
    ) AS churn_rate
FROM telco_churn
GROUP BY tenure_segment
ORDER BY churn_rate DESC;

-- Query 3: Use monthly charge quartiles to compare churn behavior across billing value segments.
WITH monthly_charge_segments AS (
    SELECT
        customerID,
        MonthlyCharges,
        Churn,
        NTILE(4) OVER (ORDER BY MonthlyCharges) AS monthly_charge_quartile
    FROM telco_churn
)
SELECT
    monthly_charge_quartile,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*) * 100,
        2
    ) AS churn_rate
FROM monthly_charge_segments
GROUP BY monthly_charge_quartile
ORDER BY monthly_charge_quartile;
