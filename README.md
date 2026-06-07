# Customer Churn Prediction

## Project Overview

This project uses the IBM Telco Customer Churn dataset to predict whether a customer is likely to churn. The goal is to connect a standard machine learning workflow with a practical customer retention use case: identifying churn risk, understanding key drivers, and producing outputs that can support targeted business follow-up.

The project is organized as a reproducible portfolio project with source code, notebook analysis, model outputs, visualizations, and ranked high-risk customers.

## Business Problem

Customer churn is a common challenge for subscription-based and telecom businesses because it affects recurring revenue, customer lifetime value, and retention planning. By estimating churn risk, a business can better prioritize customers who may need proactive support, contract review, onboarding assistance, or targeted retention outreach.

This project focuses on three practical questions:

- Which customers are more likely to churn?
- What customer, contract, billing, or service features are associated with churn?
- How can model outputs be translated into retention-oriented business insights?

## Dataset Description

The dataset contains customer demographics, account information, subscribed services, contract details, billing attributes, and churn status for telecom customers.

Key fields include:

- `customerID`
- `tenure`
- `Contract`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

## Tools Used

- Python
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Workflow

1. Load the Telco customer churn dataset.
2. Clean data fields and prepare the target variable.
3. Explore churn patterns across customer, contract, billing, and service attributes.
4. Engineer model-ready features from categorical and numerical variables.
5. Split the dataset into training and testing sets.
6. Train a Random Forest classification model.
7. Evaluate model performance using classification and ROC-AUC metrics.
8. Analyze feature importance to identify key churn drivers.
9. Rank customers by predicted churn probability.
10. Summarize model findings into business-oriented insights and recommendations.

## Results

The model outputs the following evaluation metrics:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.7686 |
| Precision | 0.5480 |
| Recall | 0.7326 |
| F1 Score | 0.6270 |
| ROC-AUC | 0.8429 |

The results are saved in `outputs/model_metrics.csv`, and the top 100 highest-risk customers are saved in `outputs/high_risk_customers.csv`.

From a business perspective, the model is especially useful as a customer risk-prioritization tool: it helps identify customers who may deserve earlier retention attention, while feature importance and segment-level patterns help explain where churn risk is concentrated.

## Visualizations

![Churn distribution](images/churn_distribution.png)

This chart shows the overall churn balance in the dataset, which helps establish the baseline level of churn risk before modeling.

![Contract churn rate](images/contract_churn_rate.png)

This chart compares churn rates across contract types, showing why contract structure is important for retention planning.

![Feature importance](images/feature_importance.png)

This chart highlights the strongest model signals, helping translate model behavior into business areas worth monitoring.

![ROC curve](images/roc_curve.png)

This chart shows the model's ability to separate churn and non-churn customers across classification thresholds.

## Key Insights

- Month-to-month contract customers show higher churn risk, suggesting that customers without longer-term commitments may require more proactive retention engagement.
- Customers with short tenure are more vulnerable to churn, which makes early onboarding and first-year customer support important business levers.
- Higher monthly charges can increase churn exposure when perceived value is weak, indicating that pricing and value communication may affect retention.
- Contract type, tenure, billing, and service-related features are important churn drivers, so retention strategies should consider both customer lifecycle stage and product/service context.
- Customer risk ranking can support targeted retention outreach by helping teams focus on customers with the highest predicted churn probability.

## Project Structure

```text
customer-churn-prediction/
|-- data/
|   `-- Telco-Customer-Churn.csv
|-- images/
|   |-- churn_distribution.png
|   |-- contract_churn_rate.png
|   |-- roc_curve.png
|   `-- feature_importance.png
|-- outputs/
|   |-- business_insights.txt
|   |-- model_metrics.csv
|   `-- high_risk_customers.csv
|-- notebook/
|   `-- churn_analysis.ipynb
|-- src/
|   `-- churn_prediction.py
|-- requirements.txt
|-- .gitignore
`-- README.md
```

## How to Run

```bash
pip install -r requirements.txt
python src/churn_prediction.py
```
