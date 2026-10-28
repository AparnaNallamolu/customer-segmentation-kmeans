import pandas as pd
from sklearn.cluster import KMeans

# Sample data: 10 customers (hand-made for practice)
df = pd.DataFrame({
    "customer_id":  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "annual_spend": [12, 15, 14, 11, 13, 48, 52, 50, 47, 55],   # in thousands of rupees
    "visits_month": [2, 3, 2, 1, 3, 9, 10, 8, 9, 11],
})

# Look at the data
print(df.head())
print(df.describe())

# Features: customer_id is only a label, so it is not used
X = df[["annual_spend", "visits_month"]]

# Fit k-means with k = 2
model = KMeans(n_clusters=2, random_state=0, n_init=10)
model.fit(X)

# Results
df["cluster"] = model.labels_
print("Centres:\n", model.cluster_centers_)
print("Cost (inertia):", model.inertia_)

# Describe each cluster
print(df.groupby("cluster")[["annual_spend", "visits_month"]].mean())

# Interpretation:
# Low-spend group (about 13K, about 2 visits a month): occasional shoppers -> discount offers
# High-spend group (about 50K, about 9 visits a month): regular shoppers -> new-collection alerts
