from typing import Optional
import pandas as pd
import numpy as np
import faiss
from sklearn.cluster import KMeans

import dill
import os
import matplotlib.pyplot as plt
import time
import sys

from sklearn.metrics import silhouette_samples
from scipy.spatial.distance import pdist, squareform

RUNPREFIX = "with-sklearn"

BASE_PATH = "/workspaces/gorillatracker/emirhan/gorillatracker"
BASE_DIR = "emirhan_checkpoints"
CHECKPOINT_DIR = f"{BASE_PATH}/{BASE_DIR}/{RUNPREFIX}"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_centroids(file_path, cache=f"{CHECKPOINT_DIR}/{sys.argv[1]}/centroids.pkl"):
    print("Loading and preprocessing data...")
    if os.path.exists(cache):
        print("Loading centroids from checkpoint...")
        with open(cache, "rb") as f:
            return pd.read_pickle(f)

    with open(file_path, "rb") as dill_file:
        df = dill.load(dill_file)

    df["id"] = df["id"].apply(lambda x: int(x.item()))
    df["embedding"] = df["embedding"].apply(np.array)
    df["label"] = df["label"].apply(lambda x: x.item())

    print(
        f"Loaded {df['id'].count()} samples, {df['id'].nunique()} unique IDs, "
        f"{df['label_string'].nunique()} unique label strings, {df['label'].nunique()} unique labels"
    )

    print("Calculating centroids...")
    centroid_df = df.groupby("label")["embedding"].apply(lambda x: list(pd.DataFrame(x.tolist()).mean())).reset_index()
    centroid_df.columns = ["tracklet_id", "embedding"]

    if not os.path.exists(os.path.dirname(cache)):
        os.makedirs(os.path.dirname(cache))
    centroid_df.to_pickle(cache)
    print(f"Centroids saved to {cache}")

    return centroid_df


# def faiss_kmeans_clustering(embeddings, k):
#     d = embeddings.shape[1]
#     kmeans = faiss.Kmeans(d, k, niter=300, verbose=True, nredo=1)
#     print("Using CPU for FAISS clustering")
#     kmeans.train(embeddings)
#     _, labels = kmeans.index.search(embeddings, 1)
#     return labels.flatten()


def sklearn_kmeans_clustering(embeddings, k):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(embeddings)
    return kmeans.labels_


def calculate_squareform_distances(embeddings):
    cache_file = f"{CHECKPOINT_DIR}/{sys.argv[1]}/squareform_distances.npy"
    if os.path.exists(cache_file):
        print("Loading cached squareform distances...")
        return np.load(cache_file)

    print("Calculating squareform distances...")
    start = time.time()
    distances = squareform(pdist(embeddings))
    if not os.path.exists(os.path.dirname(cache_file)):
        os.makedirs(os.path.dirname(cache_file))
    np.save(cache_file, distances)
    print(f"Squareform distance calculation time: {time.time() - start:.2f} seconds")
    return distances


def cached_silhouette_score(labels, distances):
    start = time.time()
    silhouette_vals = silhouette_samples(distances, labels, metric="precomputed")
    score = np.mean(silhouette_vals)
    print(f"Silhouette score calculation time: {time.time() - start:.2f} seconds")
    return score


evaluate_k_history = {}


def evaluate_k(embeddings, distances, k):
    if k <= 1:
        return float("-inf")  # Invalid k, return worst possible score
    if k in evaluate_k_history:
        return evaluate_k_history[k]
    cluster_labels = sklearn_kmeans_clustering(embeddings, k)
    score = cached_silhouette_score(cluster_labels, distances)
    evaluate_k_history[k] = score
    return score


# Warning, only works if we expect the silhouette score to follow a unimodal distribution (curve)
# The silhouette score has a unimodal distribution, for natural datasets (it doesn't when the data is equidistant)
# For grid organized points, the silhouette score will be continously sinking and not exhibit a unimodal dist, https://chatgpt.com/c/66ef02de-a460-8000-8b72-5903dceeccc1
# For random points, the silhouette score will jump around and not exhibit a unimodal dist, https://chatgpt.com/c/66ef02de-a460-8000-8b72-5903dceeccc1
def ternary_search(embeddings, distances, left, right, epsilon=50, exact=False):
    while right - left > epsilon:
        left_third = left + (right - left) // 3
        right_third = left + 2 * (right - left) // 3
        print(f"Left: {left}, Right: {right}")
        left_score = evaluate_k(embeddings, distances, left_third)
        right_score = evaluate_k(embeddings, distances, right_third)
        print(f"1/3 ({left_third}) score: {left_score}, 2/3 ({right_third}) score: {right_score}")
        if left_score < right_score:
            left = left_third
        else:
            right = right_third

    # Final evaluation to find the best k in the remaining range
    if exact:
        best_k, best_score = max(
            ((k, evaluate_k(embeddings, distances, k)) for k in range(left, right + 1)), key=lambda x: x[1]
        )
    else:
        best_k, best_score = right, evaluate_k(
            embeddings, distances, right
        )  # just choose one bound; no more computation

    return best_k, best_score


def visualize_silhouette_score(results, title, path: Optional[str] = None):
    plt.figure(figsize=(10, 6))
    plt.plot(*zip(*sorted(results.items())), marker="o")
    plt.title(f"Silhouette Score vs Number of Clusters - {title}")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    if path:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {path}")
    else:
        plt.show()

    plt.close()


def main():
    # Load and preprocess data
    print(sys.argv[1])
    centroid_df = get_centroids(f"{BASE_PATH}/{sys.argv[1]}")

    embeddings = np.array(centroid_df["embedding"].tolist()).astype("float32")
    distances = calculate_squareform_distances(embeddings)

    # Define search range
    min_k = 2
    max_k = centroid_df.shape[0]  # Set an upper limit to avoid overfitting

    print("Performing ternary search for optimal k...")
    optimal_k, best_score = ternary_search(embeddings, distances, min_k, max_k)

    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Best silhouette score: {best_score}")

    # Perform final clustering with optimal k
    final_labels = sklearn_kmeans_clustering(embeddings, optimal_k)

    final_df = centroid_df.copy()
    final_df["label"] = final_labels

    # Save results
    if not os.path.exists(f"{CHECKPOINT_DIR}/{sys.argv[1]}"):
        os.makedirs(f"{CHECKPOINT_DIR}/{sys.argv[1]}")
    final_df.to_pickle(f"{CHECKPOINT_DIR}/{sys.argv[1]}/final_clustered_df_ternary.pkl")
    print("Final results saved.")

    # Visualize results
    # Note: This visualization will only show the points that were evaluated during the ternary search
    visualize_silhouette_score(
        evaluate_k_history, "Ternary Search Clustering", path=f"{CHECKPOINT_DIR}/{sys.argv[1]}/ternary_search_silhouette_plot.png"
    )


if __name__ == "__main__":
    main()
