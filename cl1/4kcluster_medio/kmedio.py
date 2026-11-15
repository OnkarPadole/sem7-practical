#python3 -m pip install pandas numpy matplotlib scikit-learn scikit-learn-extra

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids

# Load dataset
df = pd.read_csv("Creditcard_data.csv")

# Select features (remove target column)
X = df.drop(columns=["Class"], errors="ignore")

# Standardize data
X = StandardScaler().fit_transform(X)

# Take sample for faster execution
X = X[:5000]

# Silhouette Method
scores = []

for k in range(2, 7):
    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)

    score = silhouette_score(X, labels)
    scores.append(score)

    print("Clusters:", k, "Silhouette Score:", score)

# Plot graph
plt.plot(range(2, 7), scores, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method")
plt.show()

# Best number of clusters
best_k = range(2, 7)[scores.index(max(scores))]
print("Optimal Clusters:", best_k)

# Final K-Medoids model
model = KMedoids(n_clusters=best_k, random_state=42)
labels = model.fit_predict(X)

print("Cluster Labels:", labels[:10])
