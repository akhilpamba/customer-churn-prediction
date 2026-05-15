# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# %%
df = pd.read_csv('data/customers_cleaned.csv')
df.shape

# %%
# RFM scoring
# already have recency_days, order_count, total_spend from EDA

# quintile-based scoring (1-5)
df['r_score'] = pd.qcut(df['recency_days'], q=5, labels=[5, 4, 3, 2, 1])  # lower recency = better
df['f_score'] = pd.qcut(df['order_count'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
df['m_score'] = pd.qcut(df['total_spend'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])

df[['r_score', 'f_score', 'm_score']].head()

# %%
df['r_score'] = df['r_score'].astype(int)
df['f_score'] = df['f_score'].astype(int)
df['m_score'] = df['m_score'].astype(int)

df['rfm_total'] = df['r_score'] + df['f_score'] + df['m_score']
df['rfm_total'].describe()

# %%
# avg order value
df['avg_order_value'] = np.where(
    df['order_count'] > 0,
    df['total_spend'] / df['order_count'],
    0
)

# %%
# days since join
df['joined_date'] = pd.to_datetime(df['joined_date'])
df['days_since_join'] = (pd.Timestamp.today() - df['joined_date']).dt.days

# %%
# purchase frequency rate (orders per month of tenure)
df['purchase_rate'] = np.where(
    df['tenure_months'] > 0,
    df['order_count'] / df['tenure_months'],
    0
)

# %%
# encode customer_segment
le = LabelEncoder()
df['segment_encoded'] = le.fit_transform(df['customer_segment'])
print(dict(zip(le.classes_, le.transform(le.classes_))))

# %%
# encode gender if present
if 'gender' in df.columns:
    df['gender_encoded'] = le.fit_transform(df['gender'].fillna('Unknown'))

# %%
# final feature set
feature_cols = [
    'recency_days', 'order_count', 'total_spend',
    'r_score', 'f_score', 'm_score', 'rfm_total',
    'avg_order_value', 'tenure_months', 'days_since_join',
    'purchase_rate', 'segment_encoded', 'age'
]

# drop any with nulls just to be safe
df[feature_cols].isnull().sum()

# %%
# couple of age nulls - median fill
df['age'] = df['age'].fillna(df['age'].median())

# %%
X = df[feature_cols]
y = df['churned']

print("Class distribution before SMOTE:")
print(y.value_counts(normalize=True).round(3))

# %%
# SMOTE to handle 78/22 imbalance
# tried without it first - model was predicting majority class almost always
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True).round(3))

# %%
# save for modeling
import pickle

with open('data/X_resampled.pkl', 'wb') as f:
    pickle.dump(X_resampled, f)

with open('data/y_resampled.pkl', 'wb') as f:
    pickle.dump(y_resampled, f)

# also save original (unbalanced) for final eval
X.to_csv('data/X_features.csv', index=False)
y.to_csv('data/y_labels.csv', index=False)

print("done")
