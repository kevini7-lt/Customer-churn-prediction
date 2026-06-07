from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
RANDOM_STATE = 42

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)


# =========================
# Data Loading
# =========================

# Load dataset
df = pd.read_csv(DATA_PATH)
customer_ids = df["customerID"].copy()


# =========================
# Data Cleaning
# =========================

# Clean target variable
df["Churn"] = df["Churn"].astype(str).str.strip()

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Handle missing values using median
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Remove customer identifier
df = df.drop(columns=["customerID"])


# =========================
# Exploratory Data Analysis
# =========================

# Churn Distribution
plt.figure(figsize=(7, 5))
ax = sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette=["#2E86AB", "#D1495B"],
    legend=False,
)
ax.set_title("Churn Distribution")
ax.set_xlabel("Churn")
ax.set_ylabel("Customers")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "churn_distribution.png", dpi=300)
plt.close()

# Contract Type vs Churn Rate
contract_churn = (
    df.assign(churn_flag=(df["Churn"] == "Yes").astype(int))
    .groupby("Contract", as_index=False)["churn_flag"]
    .mean()
    .sort_values("churn_flag", ascending=False)
)

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=contract_churn,
    x="Contract",
    y="churn_flag",
    hue="Contract",
    palette=["#D1495B", "#F29E4C", "#2E86AB"],
    legend=False,
)
ax.set_title("Contract Type vs Churn Rate")
ax.set_xlabel("Contract Type")
ax.set_ylabel("Churn Rate")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "contract_churn_rate.png", dpi=300)
plt.close()


# =========================
# Feature Engineering
# =========================

# Binary encode target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# One-hot encode categorical variables
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]


# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)


# =========================
# Model Training
# =========================

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=8,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
model.fit(X_train, y_train)


# =========================
# Model Evaluation
# =========================

# Generate predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}

pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nCustomer Churn Prediction - Model Performance")
print("=" * 52)
for metric, value in metrics.items():
    print(f"{metric.replace('_', ' ').title():<10}: {value:.4f}")
print("\nConfusion Matrix")
print(conf_matrix)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="#2E86AB", linewidth=2, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], color="#9A9A9A", linestyle="--", linewidth=1)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "roc_curve.png", dpi=300)
plt.close()


# =========================
# Feature Importance Analysis
# =========================

# Display Top 10 Features
feature_importance = (
    pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    .sort_values("importance", ascending=False)
    .head(10)
)

print("\nTop 10 Feature Importances")
print("=" * 52)
print(feature_importance.to_string(index=False))

plt.figure(figsize=(9, 6))
ax = sns.barplot(
    data=feature_importance,
    x="importance",
    y="feature",
    hue="feature",
    palette="viridis",
    legend=False,
)
ax.set_title("Top 10 Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=300)
plt.close()


# =========================
# Customer Risk Ranking
# =========================

# Score all customers
high_risk_customers = (
    pd.DataFrame(
        {
            "customerID": customer_ids,
            "churn_probability": model.predict_proba(X)[:, 1],
        }
    )
    .sort_values("churn_probability", ascending=False)
    .head(100)
)
high_risk_customers.to_csv(OUTPUT_DIR / "high_risk_customers.csv", index=False)


# =========================
# Business Insights
# =========================

# Analyze customer segments
top_features = ", ".join(feature_importance["feature"].head(5))
overall_churn = df["Churn"].mean()
month_to_month_churn = df.loc[df["Contract"] == "Month-to-month", "Churn"].mean()
short_tenure_churn = df.loc[df["tenure"] <= 12, "Churn"].mean()
long_tenure_churn = df.loc[df["tenure"] > 48, "Churn"].mean()
high_charge_cutoff = df["MonthlyCharges"].quantile(0.75)
high_charge_churn = df.loc[df["MonthlyCharges"] >= high_charge_cutoff, "Churn"].mean()

business_insights = f"""
Model Summary
-------------
Accuracy: {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1 Score: {metrics['f1_score']:.4f}
ROC-AUC: {metrics['roc_auc']:.4f}
Overall Churn Rate: {overall_churn:.2%}

Key Churn Drivers
-----------------
The strongest model signals are: {top_features}.
Contract type, tenure, monthly charges, and service variables are key churn drivers.

Customer Segments At Risk
-------------------------
Month-to-month customers have a churn rate of {month_to_month_churn:.2%}.
Customers with tenure of 12 months or less have a churn rate of {short_tenure_churn:.2%}.
Customers with tenure above 48 months have a churn rate of {long_tenure_churn:.2%}.
Customers in the highest monthly charge quartile have a churn rate of {high_charge_churn:.2%}.

Business Recommendations
------------------------
1. Prioritize retention outreach for customers with the highest churn probability.
2. Promote annual contract upgrades for month-to-month customers.
3. Launch early-tenure onboarding support during the first 12 months.
4. Review pricing and value perception for high monthly charge customers.
5. Use targeted offers for customers with limited service bundles.
6. Monitor churn risk before renewal and billing-cycle milestones.
7. Refresh retention strategy as feature importance patterns change.
""".strip()

with open(OUTPUT_DIR / "business_insights.txt", "w", encoding="utf-8") as file:
    file.write(business_insights)
