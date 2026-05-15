# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('retail_customers.csv')
df.head()

# %%
df.shape

# %%
# quick look at nulls
df.isnull().sum()

# %%
# tenure and total_spend have some nulls - check if they're random or systematic
df[df['tenure_months'].isnull()].head(10)

# %%
# looks like new customers (joined_date within last 30 days) - makes sense they'd have no tenure calc
# will fill with 0 for now
df['tenure_months'] = df['tenure_months'].fillna(0)

# %%
# total_spend nulls - these are customers with 0 orders, fill with 0
df['total_spend'].isnull().sum()  # 312 rows
df['total_spend'] = df['total_spend'].fillna(0)

# %%
df.dtypes

# %%
# joined_date is object, need to convert
df['joined_date'] = pd.to_datetime(df['joined_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

# %%
df.describe()

# %%
# churn distribution
df['churned'].value_counts()

# %%
# 78/22 split - decent imbalance, will need to address this
df['churned'].value_counts(normalize=True).round(3)

# %%
# age distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.tight_layout()
plt.savefig('plots/age_dist.png', dpi=150)
plt.show()

# %%
# spend by churn status
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='churned', y='total_spend')
plt.title('Total Spend by Churn Status')
plt.tight_layout()
plt.show()

# churned customers clearly spend less - expected but good to confirm

# %%
# tenure vs churn
plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x='churned', y='tenure_months')
plt.title('Tenure by Churn Status')
plt.tight_layout()
plt.show()

# %%
# check for outliers in total_spend
df['total_spend'].quantile([0.95, 0.99, 0.999])

# %%
# top 0.1% has spend > 15k - probably b2b accounts or data errors
# capping at 99.9th percentile
spend_cap = df['total_spend'].quantile(0.999)
df['total_spend'] = df['total_spend'].clip(upper=spend_cap)

# %%
# recency - days since last purchase
df['recency_days'] = (pd.Timestamp.today() - df['last_purchase_date']).dt.days
df['recency_days'].describe()

# %%
# customers with no purchase date (new signups) - set recency to max + 1
max_recency = df['recency_days'].max()
df['recency_days'] = df['recency_days'].fillna(max_recency + 1)

# %%
# correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=150)
plt.show()

# %%
# churn rate by customer segment
df.groupby('customer_segment')['churned'].mean().sort_values(ascending=False)

# %%
# interesting - 'occasional' segment has 40%+ churn, 'loyal' segment < 5%
# segment will definitely be useful as a feature

# %%
# save cleaned data
df.to_csv('data/customers_cleaned.csv', index=False)
print("saved:", df.shape)
