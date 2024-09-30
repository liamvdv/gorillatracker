import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
from gorillatracker.classification.clustering import calculate_metrics, get_cache_key


class CosineSimilarityKMeans:
    def __init__(self, n_clusters, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)

    def fit_predict(self, X):
        X_normalized = normalize(X)
        cosine_sim = cosine_similarity(X_normalized)
        distance_matrix = 1 - cosine_sim
        return self.kmeans.fit_predict(distance_matrix)


def sweep_clustering_algorithms_cosine(df, configs, cache_dir=None):
    results = []

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for dataset, model, algorithm, param_combinations in tqdm(configs, desc="Processing configurations"):
        print("Processing", dataset, model, algorithm)
        subset = df[(df["model"] == model) & (df["dataset"] == dataset)]
        subset = subset.reset_index(drop=True)

        if subset.empty:
            raise ValueError(f"No data found for {dataset} - {model}")

        embeddings = np.stack(subset["embedding"].to_numpy())
        true_labels = subset["label"].to_numpy()

        for params in param_combinations:
            cache_key = get_cache_key(dataset, model, algorithm, params)
            cache_file = os.path.join(cache_dir or "", f"{cache_key}.pkl")
            if cache_dir and os.path.exists(cache_file):
                print(f"Loading cached result for {dataset}, {model}, {algorithm}, {params}")
                metric = pd.read_pickle(cache_file)
            else:
                print("Processing", dataset, model, algorithm, params)
                if algorithm == "KMeans":
                    clusterer = CosineSimilarityKMeans(random_state=42, **params)
                    labels = clusterer.fit_predict(embeddings)
                elif algorithm == "AgglomerativeClustering":
                    clusterer = AgglomerativeClustering(metric="cosine", linkage="average", **params)
                    labels = clusterer.fit_predict(embeddings)
                elif algorithm == "HDBSCAN":
                    distance_matrix = 1 - cosine_similarity(normalize(embeddings))
                    clusterer = HDBSCAN(metric="precomputed", **params)
                    labels = clusterer.fit_predict(distance_matrix)
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")

                metric = calculate_metrics(embeddings, labels, true_labels, metric="cosine")
                metric.update(
                    {
                        "dataset": dataset,
                        "model": model,
                        "algorithm": algorithm,
                        "algorithm_params": params,
                        "n_clusters": len(np.unique(labels[labels != -1])),  # Excluding noise points
                        "n_true_clusters": len(np.unique(true_labels)),
                    }
                )
                if cache_dir:
                    pd.to_pickle(metric, cache_file)

            results.append(metric)

    return pd.DataFrame(results)
