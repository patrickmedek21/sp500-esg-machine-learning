import os
os.chdir("C:/Users/patri/Documents/Northeastern/Data Mining & Machine Learning/Project")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# Load data
df = pd.read_excel("ESG_Selected_Features_with_PCA.xlsx")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# Define beating market at 24%
market_return = 0.24
df["BEAT_MARKET"] = (df["RETURN_1YR"] > market_return).astype(int)

print("\nBeat market distribution:")
print(df["BEAT_MARKET"].value_counts())
print(df["BEAT_MARKET"].value_counts(normalize=True))

# Split ESG score into bins
esg_labels = [
    "Very Low (0–20%)",
    "Low (20–40%)",
    "Medium (40–60%)",
    "High (60–80%)",
    "Very High (80–100%)"
]

df["TOTAL_ESG"] = df[["ENV_SCORE", "SOCIAL_SCORE", "GOV_SCORE"]].mean(axis=1)

df["ESG_GROUP_5"] = pd.qcut(
    df["TOTAL_ESG"],
    q=5,
    labels=esg_labels
)

# Define features
control_vars = ["BETA", "MARKET_CAP", "EBITDA"]
esg_vars = ["ENV_SCORE", "SOCIAL_SCORE", "GOV_SCORE"]

df = df.drop(columns=["SECTOR_Real Estate"])
df = pd.get_dummies(df, columns=["SECTOR"], drop_first=True)
sector_vars = [col for col in df.columns if col.startswith("SECTOR_")]

feature_vars = control_vars + esg_vars + sector_vars

X = df[feature_vars].copy()
y = df["BEAT_MARKET"].copy()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Random forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_auc = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc")
cv_bal_acc = cross_val_score(rf, X, y, cv=cv, scoring="balanced_accuracy")

print("\n===== RANDOM FOREST RESULTS =====")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("CV ROC-AUC Mean:", round(cv_auc.mean(), 4))
print("CV ROC-AUC Std:", round(cv_auc.std(), 4))
print("CV Balanced Accuracy Mean:", round(cv_bal_acc.mean(), 4))
print("CV Balanced Accuracy Std:", round(cv_bal_acc.std(), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance plot
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

importance_top = importance.head(min(10, len(importance)))

bar_colors = []
for col in importance_top.index:
    if col in esg_vars:
        bar_colors.append("#2E8B57")
    elif col in control_vars:
        bar_colors.append("#4C78A8")
    else:
        bar_colors.append("#9E9E9E")

plt.figure(figsize=(10, 5.5))
bars = plt.bar(importance_top.index, importance_top.values, color=bar_colors, edgecolor="black", linewidth=0.6)
plt.title("Random Forest Feature Importance for Market Outperformance Classification", fontsize=13, weight="bold")
plt.ylabel("Feature Importance")
plt.xlabel("")
plt.xticks(rotation=55, ha="right")
plt.ylim(0, importance_top.max() * 1.18)

for bar, val in zip(bars, importance_top.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.003,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor="#2E8B57", edgecolor="black", label="ESG Variable"),
    Patch(facecolor="#4C78A8", edgecolor="black", label="Financial Variable"),
    Patch(facecolor="#9E9E9E", edgecolor="black", label="Sector Variable")
]
plt.legend(handles=legend_handles, frameon=False, loc="upper right")

plt.tight_layout()
plt.show()

# Category totals
financial_importance = importance[[c for c in importance.index if c in control_vars]].sum()
esg_importance = importance[[c for c in importance.index if c in esg_vars]].sum()
sector_importance = importance[[c for c in importance.index if c in sector_vars]].sum() if sector_vars else 0

print("\nFinancial Importance:", round(financial_importance, 4))
print("ESG Importance:", round(esg_importance, 4))
print("Sector Importance:", round(sector_importance, 4))

# Test results table
test_results = X_test.copy()
test_results["y_true"] = y_test
test_results["y_pred"] = y_pred
test_results["y_prob"] = y_prob
test_results["TOTAL_ESG"] = df.loc[X_test.index, "TOTAL_ESG"]
test_results["ESG_GROUP_5"] = df.loc[X_test.index, "ESG_GROUP_5"]
test_results["BETA"] = df.loc[X_test.index, "BETA"]

# Split BETA into bins
beta_bins = pd.qcut(test_results["BETA"], q=3)
beta_labels = [f"{interval.left:.1f}–{interval.right:.1f}" for interval in beta_bins.cat.categories]

test_results["BETA_GROUP_3"] = pd.qcut(
    test_results["BETA"],
    q=3,
    labels=beta_labels
)

# Predicted probability of beating market by ESG bin, stratified by beta range
plot_df = (
    test_results.groupby(["BETA_GROUP_3", "ESG_GROUP_5"], observed=False)["y_prob"]
    .mean()
    .reset_index()
)

plot_df["ESG_GROUP_5"] = pd.Categorical(
    plot_df["ESG_GROUP_5"],
    categories=esg_labels,
    ordered=True
)

plot_df["BETA_GROUP_3"] = pd.Categorical(
    plot_df["BETA_GROUP_3"],
    categories=beta_labels,
    ordered=True
)

g = sns.FacetGrid(
    plot_df,
    col="BETA_GROUP_3",
    col_order=beta_labels,
    sharey=True,
    height=4,
    aspect=1.15
)

colors = ["#3BA272", "#F2B134", "#E76F51"]

for ax, beta_group, color in zip(g.axes.flat, beta_labels, colors):
    subset = plot_df[plot_df["BETA_GROUP_3"] == beta_group]
    
    sns.lineplot(
        data=subset,
        x="ESG_GROUP_5",
        y="y_prob",
        marker="o",
        linewidth=2.5,
        markersize=8,
        ax=ax,
        color=color
    )
    
    ax.set_title(f"Beta Range: {beta_group}", color=color, fontsize=12, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, plot_df["y_prob"].max() * 1.15)
    ax.tick_params(axis="x", rotation=55)
    
    x_vals = range(len(subset))
    y_vals = subset["y_prob"].values
    
    if len(y_vals) > 0:
        ax.text(x_vals[0], y_vals[0] + 0.04, f"{y_vals[0]:.2f}", fontsize=9, ha="center")
        ax.text(x_vals[-1], y_vals[-1] + 0.04, f"{y_vals[-1]:.2f}", fontsize=9, ha="center")

g.set_axis_labels("", "Predicted Probability of Beating Market")
g.fig.subplots_adjust(top=0.76, bottom=0.20)
g.fig.suptitle(
    "Model-Predicted Market Outperformance by ESG Level, Stratified by Beta",
    fontsize=14,
    weight="bold"
)
g.fig.text(0.5, -0.2, "ESG Level (Percentile Bins)", ha="center", fontsize=12)

plt.show()

print("\n===== AVG PREDICTED PROBABILITY BY ESG GROUP AND BETA RANGE =====")
print(
    plot_df.pivot(
        index="ESG_GROUP_5",
        columns="BETA_GROUP_3",
        values="y_prob"
    )
)