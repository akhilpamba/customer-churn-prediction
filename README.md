# Customer Churn Prediction

Retail churn prediction project. Dataset has ~50k customer records with purchase history, demographics, and a churn label.

## What I did

Started with some EDA to understand the data, then built features (RFM indicators mostly), handled class imbalance with SMOTE, and trained logistic regression + random forest models. Random forest ended up being significantly better.

Final results:
- Accuracy: 87%
- AUC-ROC: 0.91
- Used k-fold CV and GridSearchCV for tuning

## Files

- 01_eda_and_cleaning.py
- 02_feature_engineering.py
- 03_modeling.py
- churn_sql_queries.sql

## Notes

Class imbalance was around 78/22 so SMOTE helped noticeably.
