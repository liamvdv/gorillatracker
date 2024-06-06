import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from gorillatracker.utils.embedding_generator import read_embeddings_from_disk

# Read embeddings from disk
df = read_embeddings_from_disk("example.pkl")

# Extract embeddings and labels
embeddings = np.vstack(df["embedding"].values)
labels = df["label"].values

# Define the range of k values to test
k_values = list(range(2000, 31000, 1000))

# Initialize lists to store the evaluation metrics
silhouette_scores = []
davies_bouldin_scores = []

# Evaluate K-Means with different k values using k-means++ initialization
for k in k_values:
    print(f"Running K-Means++ with k={k}")
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)

    silhouette_avg = silhouette_score(embeddings, cluster_labels)
    davies_bouldin_avg = davies_bouldin_score(embeddings, cluster_labels)

    silhouette_scores.append(silhouette_avg)
    davies_bouldin_scores.append(davies_bouldin_avg)

# Plot the evaluation metrics
fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Silhouette Score", color=color)
ax1.plot(k_values, silhouette_scores, "o-", color=color)
ax1.tick_params(axis="y", labelcolor=color)

ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("Davies-Bouldin Score", color=color)
ax2.plot(k_values, davies_bouldin_scores, "s-", color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.title("Evaluation of K-Means Clustering with k-means++ Initialization")
plt.savefig("kmeans_evaluation_plot.png")

# Find the optimal number of clusters
optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
optimal_k_davies_bouldin = k_values[np.argmin(davies_bouldin_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")
print(f"Optimal number of clusters based on Davies-Bouldin score: {optimal_k_davies_bouldin}")

# Save the results to a file
results = {
    "k_values": k_values,
    "silhouette_scores": silhouette_scores,
    "davies_bouldin_scores": davies_bouldin_scores,
    "optimal_k_silhouette": optimal_k_silhouette,
    "optimal_k_davies_bouldin": optimal_k_davies_bouldin,
}

results_df = pd.DataFrame(results)
results_df.to_csv("kmeans_evaluation_results.csv", index=False)
