# %%
from gorillatracker.classification.metrics import analyse_embedding_space
import pandas as pd
import dill

print("Running")
base = "/workspaces/gorillatracker/emirhan"

# fp = f"{base}/gorillatracker/notebooks/ssl_embeddings.pkl"
fp = f"{base}/gorillatracker/ssl_embeddings_new.pkl"

with open(fp, "rb") as dill_file:
    df = dill.load(dill_file)

# %%
import numpy as np

df["id"] = df["id"].apply(lambda x: int(x.item()))
df["embedding"] = df["embedding"].apply(np.array) # [t.item() for t in x]
df["label"] = df["label"].apply(lambda x: x.item()) # np.array([t.item() for t in x])

print(df.head())
print(df["id"].count())
print(df["id"].nunique())
print(df["label_string"].nunique())
print(df["label"].nunique())

# %%
# Group by label and calculate the mean of the embeddings
centroid_df = df.groupby("label")["embedding"].apply(lambda x: list(pd.DataFrame(x.tolist()).mean())).reset_index()
centroid_df.columns = ["tracklet_id", "embedding"]
print(centroid_df.head())

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
step = count // 16
results_df = generate_results_df(centroid_df, range(1, count, step))

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
k_values = range(min(1, k - step), k + step, min(1, step // 10))
results_df = generate_results_df(centroid_df, k_values)
dataset = "ssl_embeddings_tracklet_centroids"
model = "A"
algorithm = "KMeans"

results_df["dataset"] = dataset
results_df["model"] = model
results_df["algorithm"] = algorithm
results_df["algorithm_params"] = results_df["k"].apply(lambda x: {"n_clusters": x})
results_df["n_true_clusters"] = 0

visualize_alg_metrics(results_df, dataset, model, algorithm, formatted_names)
results_df.to_pickle("emirhan_metrics_df_finegrained71k.pkl")

# %%
k = results_df["k"].iloc[results_df["silhouette_coefficient"].idxmax()]
embeddings = np.array(centroid_df["embedding"].tolist())
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

final_df = centroid_df.copy()
final_df["label"] = cluster_labels
final_df.to_pickle("emirhan_clustered_df71k.pkl")
print(final_df.head())

# %%
import shutil


# os.rename("emirhan_clustered_df.pkl", f"{base}/gorillatracker/complete_18_09_2024_clustered_df.pkl")
shutil.copy(
    "emirhan_clustered_df71k.pkl",
    "/workspaces/gorillatracker/emirhan/gorillatracker/complete_19_09_2024_clustered_df71k.pkl",
)
