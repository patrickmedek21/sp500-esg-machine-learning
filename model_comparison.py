import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, ConfusionMatrixDisplay,
                             mean_squared_error, r2_score)


df = pd.read_csv('/Users/shivanshi/Downloads/Merged_Data(in).csv')
print(df.shape)
print(df.columns.tolist())
print(df.head())
print(df.describe())
print(df.info())

print(df.isnull().sum())
df = df.dropna()
print("Shape after dropping missing:", df.shape)

sp500_return = 0.24
df['BEAT_MARKET'] = (df['RETURN_1YR'] > sp500_return).astype(int)
print(df['BEAT_MARKET'].value_counts())
print("% that beat market:", round(df['BEAT_MARKET'].mean()*100, 2), "%")

features = ['ENV_SCORE', 'SOCIAL_SCORE', 'GOV_SCORE',
            'MARKET_CAP', 'EBITDA', 'BETA', 'REVENUE_GROWTH']
 
X = df[features]
y = df['BEAT_MARKET']
 
print("Features:\n", X.head())
print("\nTarget distribution:\n", y.value_counts())

corr_data = df[features + ['RETURN_1YR', 'BEAT_MARKET']]
correlation_matrix = corr_data.corr()
print(correlation_matrix)
 
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=0.3,
                                                      random_state=42)
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

scaler = StandardScaler()
scaler.fit(X_train)
 
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
 
y_pred_lr = lr.predict(X_test_scaled)

y_pred_lr_binary = (y_pred_lr >= 0.5).astype(int)
 
# %%
print("=== LINEAR REGRESSION ===")
print("R-squared:", round(r2_score(y_test, y_pred_lr), 4))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred_lr)), 4))
print("Accuracy (with 0.5 threshold):", round(accuracy_score(y_test, y_pred_lr_binary)*100, 2), "%")
print(classification_report(y_test, y_pred_lr_binary))

cm_lr = confusion_matrix(y_test, y_pred_lr_binary)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                  display_labels=['Did Not Beat', 'Beat Market'])
disp_lr.plot(cmap='Blues')
plt.title('Linear Regression - Confusion Matrix')
plt.show()

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
 
y_pred_log = log_reg.predict(X_test_scaled)
 
print("=== LOGISTIC REGRESSION ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_log)*100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log,
                                   display_labels=['Did Not Beat', 'Beat Market'])
disp_log.plot(cmap='Blues')
plt.title('Logistic Regression - Confusion Matrix')
plt.show()

coef_df = pd.DataFrame({'Feature': features,
                         'Coefficient': log_reg.coef_[0]})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print("\nLogistic Regression Coefficients:")
print(coef_df)
 
dt = DecisionTreeClassifier(random_state=42, max_depth=5)
dt.fit(X_train_scaled, y_train)
 
y_pred_dt = dt.predict(X_test_scaled)
 
print("=== DECISION TREE ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt)*100, 2), "%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                                  display_labels=['Did Not Beat', 'Beat Market'])
disp_dt.plot(cmap='Blues')
plt.title('Decision Tree - Confusion Matrix')
plt.show()

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=features, class_names=['Did Not Beat', 'Beat Market'],
          filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.show()
 
importance_df = pd.DataFrame({'Feature': features,
                               'Importance': dt.feature_importances_})
importance_df = importance_df.sort_values('Importance', ascending=False)
print("\nFeature Importance (Decision Tree):")
print(importance_df)
 
plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Decision Tree - Feature Importance')
plt.tight_layout()
plt.show()


acc_lr = accuracy_score(y_test, y_pred_lr_binary) * 100
acc_log = accuracy_score(y_test, y_pred_log) * 100
acc_dt = accuracy_score(y_test, y_pred_dt) * 100
 
print("=" * 50)
print("MODEL ACCURACY COMPARISON")
print("=" * 50)
print(f"Linear Regression:    {acc_lr:.2f}%")
print(f"Logistic Regression:  {acc_log:.2f}%")
print(f"Decision Tree:        {acc_dt:.2f}%")
print("=" * 50)


models = ['Linear Regression', 'Logistic Regression', 'Decision Tree']
accuracies = [acc_lr, acc_log, acc_dt]
 
plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.2f}%', ha='center', fontsize=11)
plt.tight_layout()
plt.show()

best_model_name = models[np.argmax(accuracies)]
print(f"\nBest Model: {best_model_name}")
 
if best_model_name == 'Linear Regression':
    best_cm = cm_lr
    best_pred = y_pred_lr_binary
elif best_model_name == 'Logistic Regression':
    best_cm = cm_log
    best_pred = y_pred_log
else:
    best_cm = cm_dt
    best_pred = y_pred_dt
 
print("\nConfusion Matrix (Best Model):")
print(best_cm)
print("\nTN (True Negative):", best_cm[0][0])
print("FP (False Positive):", best_cm[0][1])
print("FN (False Negative):", best_cm[1][0])
print("TP (True Positive):", best_cm[1][1])
 
print("\nDetailed Classification Report:")
print(classification_report(y_test, best_pred,
                            target_names=['Did Not Beat', 'Beat Market']))

plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Did Not Beat', 'Beat Market'],
            yticklabels=['Did Not Beat', 'Beat Market'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Best Model ({best_model_name}) - Confusion Matrix Heatmap')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()




import os
os.makedirs('ESG_Plots', exist_ok=True)

# 1 - Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
corr_data = df[features + ['RETURN_1YR', 'BEAT_MARKET']]
sns.heatmap(corr_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('ESG_Plots/1_Correlation_Matrix_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 2 - Linear Regression Confusion Matrix
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                                  display_labels=['Did Not Beat', 'Beat Market'])
disp_lr.plot(cmap='Blues')
plt.title('Linear Regression - Confusion Matrix')
plt.savefig('ESG_Plots/2_Linear_Regression_CM.png', dpi=300, bbox_inches='tight')
plt.close()

# 3 - Logistic Regression Confusion Matrix
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log,
                                   display_labels=['Did Not Beat', 'Beat Market'])
disp_log.plot(cmap='Blues')
plt.title('Logistic Regression - Confusion Matrix')
plt.savefig('ESG_Plots/3_Logistic_Regression_CM.png', dpi=300, bbox_inches='tight')
plt.close()

# 4 - Decision Tree Confusion Matrix
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt,
                                  display_labels=['Did Not Beat', 'Beat Market'])
disp_dt.plot(cmap='Blues')
plt.title('Decision Tree - Confusion Matrix')
plt.savefig('ESG_Plots/4_Decision_Tree_CM.png', dpi=300, bbox_inches='tight')
plt.close()

# 5 - Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=features, class_names=['Did Not Beat', 'Beat Market'],
          filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.savefig('ESG_Plots/5_Decision_Tree_Visual.png', dpi=300, bbox_inches='tight')
plt.close()

# 6 - Feature Importance
plt.figure(figsize=(8, 5))
importance_df = pd.DataFrame({'Feature': features, 'Importance': dt.feature_importances_})
importance_df = importance_df.sort_values('Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title('Decision Tree - Feature Importance')
plt.tight_layout()
plt.savefig('ESG_Plots/6_Feature_Importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 7 - Model Accuracy Comparison
plt.figure(figsize=(8, 5))
models = ['Linear Regression', 'Logistic Regression', 'Decision Tree']
accuracies = [acc_lr, acc_log, acc_dt]
bars = plt.bar(models, accuracies, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 100)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.2f}%', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('ESG_Plots/7_Model_Accuracy_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 8 - Best Model Confusion Matrix Heatmap
best_model_name = models[np.argmax(accuracies)]
if best_model_name == 'Linear Regression':
    best_cm = cm_lr
elif best_model_name == 'Logistic Regression':
    best_cm = cm_log
else:
    best_cm = cm_dt

plt.figure(figsize=(6, 5))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Did Not Beat', 'Beat Market'],
            yticklabels=['Did Not Beat', 'Beat Market'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Best Model ({best_model_name}) - Confusion Matrix Heatmap')
plt.tight_layout()
plt.savefig('ESG_Plots/8_Best_Model_CM_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 9 - Feature Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('ESG_Plots/9_Feature_Correlation_Heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("All 9 plots saved to 'ESG_Plots' folder!")
print("Location:", os.path.abspath('ESG_Plots'))