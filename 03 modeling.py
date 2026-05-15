# %%
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)

# %%
with open('data/X_resampled.pkl', 'rb') as f:
    X = pickle.load(f)
with open('data/y_resampled.pkl', 'rb') as f:
    y = pickle.load(f)

X.shape, y.shape

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# scale for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# --- Logistic Regression (baseline) ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

print("Logistic Regression:")
print(classification_report(y_test, lr_preds))
print("AUC-ROC:", round(roc_auc_score(y_test, lr_proba), 4))

# %%
# LR got ~79% accuracy, AUC 0.84 - decent but let's try RF

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest (default params):")
print(classification_report(y_test, rf_preds))
print("AUC-ROC:", round(roc_auc_score(y_test, rf_proba), 4))

# %%
# RF is better already - 85% accuracy, AUC 0.89
# let's tune it

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# narrowed this down after a wider search - full grid took too long
param_grid_narrow = {
    'n_estimators': [200, 300],
    'max_depth': [20, 30, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid_narrow,
    cv=cv,
    scoring='roc_auc',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# %%
print("Best params:", grid_search.best_params_)
print("Best CV AUC:", round(grid_search.best_score_, 4))

# %%
best_rf = grid_search.best_estimator_
best_preds = best_rf.predict(X_test)
best_proba = best_rf.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest:")
print(classification_report(y_test, best_preds))
print("AUC-ROC:", round(roc_auc_score(y_test, best_proba), 4))

# %%
# 87% accuracy, 0.91 AUC - good enough to move forward with

# confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, best_preds, ax=ax)
plt.title('Confusion Matrix - Tuned RF')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.show()

# %%
# ROC curve comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, best_proba)

plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = 0.84)')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = 0.91)')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_comparison.png', dpi=150)
plt.show()

# %%
# feature importance
feat_imp = pd.Series(
    best_rf.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
feat_imp.head(10).plot(kind='barh')
plt.gca().invert_yaxis()
plt.title('Top 10 Feature Importances')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()

# %%
feat_imp.head(10)

# %%
# recency and rfm_total are top predictors - makes sense
# avg_order_value and tenure also important

# %%
# save model
with open('models/best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model saved.")

# %%
# export predictions for Tableau dashboard
X_orig = pd.read_csv('data/X_features.csv')
y_orig = pd.read_csv('data/y_labels.csv').squeeze()

X_orig_proba = best_rf.predict_proba(X_orig)[:, 1]

output = X_orig.copy()
output['churn_probability'] = X_orig_proba
output['churn_risk_tier'] = pd.cut(
    X_orig_proba,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)
output['actual_churn'] = y_orig.values

output.to_csv('data/churn_predictions_for_tableau.csv', index=False)
print("Exported for Tableau:", output.shape)
