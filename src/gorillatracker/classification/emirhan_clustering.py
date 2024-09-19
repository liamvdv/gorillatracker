# %%
from gorillatracker.classification.metrics import analyse_embedding_space
import pandas as pd
import numpy as np

base = "/workspaces/gorillatracker"
ids = np.load(f"{base}/vit_ids.npy")
embeddings = np.load(f"{base}/vit_embeddings.npy")
labels = np.load(f"{base}/vit_labels.npy")

# Create a DataFrame
df = pd.DataFrame({"id": list(ids), "embedding": list(embeddings), "label": labels})

# %%
print(df.head())
print(df["id"].count())
print(df["id"].nunique())
print(df["label"].nunique())

# %%
# Group by label and calculate the mean of the embeddings
centroid_df = df.groupby("label")["embedding"].apply(lambda x: list(pd.DataFrame(x.tolist()).mean())).reset_index()
centroid_df.columns = ["tracklet_id", "embedding"]

# %%
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


def generate_results_df(df, k_values):

    embeddings = np.array(centroid_df["embedding"].tolist())

    # Initialize lists to store results
    results = []

    # Perform grid search
    for k in tqdm(k_values, desc="Grid Search Progress"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Update the 'label' column with cluster labels
        df["label"] = cluster_labels

        # Analyze the embedding space
        metrics = analyse_embedding_space(df)

        # Store results
        results.append({**metrics, "k": k})
    results_df = pd.DataFrame(results)
    return results_df


count = centroid_df.shape[0]
results_df = generate_results_df(centroid_df, range(1, count, 10000))

# Print the results
print(results_df.head())

# %%
results_df.to_pickle("emirhan_metrics_df.pkl")

# %%
from gorillatracker.classification.clustering import visualize_alg_metrics
from gorillatracker.classification.metrics import formatted_names

dataset = "ssl_embeddings_tracklet_centroids"
model = "A"
algorithm = "KMeans"

results_df["dataset"] = dataset
results_df["model"] = model
results_df["algorithm"] = algorithm
results_df["algorithm_params"] = results_df["k"].apply(lambda x: {"n_clusters": x})
results_df["n_true_clusters"] = 0

visualize_alg_metrics(results_df, dataset, model, algorithm, formatted_names)

# %%
k = results_df["k"].iloc[results_df["silhouette_coefficient"].idxmax()]
sweep = range(max(1, k - 1000), min(k + 1000, count), 100)
results_df = generate_results_df(centroid_df, sweep)
dataset = "ssl_embeddings_tracklet_centroids"
model = "A"
algorithm = "KMeans"

results_df["dataset"] = dataset
results_df["model"] = model
results_df["algorithm"] = algorithm
results_df["algorithm_params"] = results_df["k"].apply(lambda x: {"n_clusters": x})
results_df["n_true_clusters"] = 0

visualize_alg_metrics(results_df, dataset, model, algorithm, formatted_names)
results_df.to_pickle("emirhan_metrics_df_finegrained.pkl")

# %%
k = results_df["k"].iloc[results_df["silhouette_coefficient"].idxmax()]
embeddings = np.array(centroid_df["embedding"].tolist())
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

final_df = centroid_df.copy()
final_df["label"] = cluster_labels
final_df.to_pickle("emirhan_clustered_df.pkl")
print(final_df.head())

# %%
import os

os.rename("emirhan_clustered_df.pkl", f"{base}/gorillatracker/complete_19_09_2024_clustered_df.pkl")
