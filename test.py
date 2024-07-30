import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import model


data = pd.read_csv("testData.csv", header=None, sep='\t')


def toFloat (col) -> float:
    col = col.replace(',', '.')
    return float(col)


data.columns = ["x1", "x2"]
data["x1"] = data["x1"].apply(toFloat)
data["x2"] = data["x2"].apply(toFloat)

model = model.my_KMC()

optimal_k = model.elbow(data, plot=True)
print(f"Optimal number of clusters for custom model: {optimal_k}")

model.k = optimal_k
centroids_custom, assignments_custom = model.fit(data)

kmeans_sklearn = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_sklearn.fit(data)
assignments_sklearn = kmeans_sklearn.labels_
centroids_sklearn = kmeans_sklearn.cluster_centers_

silhouette_my_model = model.silhouette_score(data)
print(f"Silhouette score for custom model: {silhouette_my_model}")

silhouette_sklearn = silhouette_score(data, assignments_sklearn)
print(f"Silhouette score for sklearn model: {silhouette_sklearn}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(data["x1"], data["x2"], c=assignments_custom, cmap="viridis", marker='o', edgecolor='k', s=50, alpha=0.7)
plt.scatter(centroids_custom[:, 0], centroids_custom[:, 1], c="red", marker='X', s=200, label="Centroids")
plt.title("My K-means Clustering")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(data["x1"], data["x2"], c=assignments_sklearn, cmap="viridis", marker='o', edgecolor='k', s=50, alpha=0.7)
plt.scatter(centroids_sklearn[:, 0], centroids_sklearn[:, 1], c="red", marker='X', s=200, label="Centroids")
plt.title("sklearn K-means Clustering")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.show()
