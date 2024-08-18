import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score


def compute_centroids(df):
    """
    Compute centroids for each label group using the mean of all embeddings.

    Args:
    df (pd.DataFrame): DataFrame containing 'label', 'label_string', and 'embedding' columns.

    Returns:
    pd.DataFrame: DataFrame containing centroids for each label group.
    """
    centroids = df.groupby(["label"])["embedding"].apply(lambda x: np.mean(np.vstack(x), axis=0))
    centroid_df = pd.DataFrame({"centroid": centroids.values})
    centroid_df[["label"]] = pd.DataFrame(centroids.index.tolist(), index=centroid_df.index)
    return centroid_df


def compute_iqr_centroids(df):
    """
    Compute centroids for each label group using embeddings within the IQR.

    Args:
    df (pd.DataFrame): DataFrame containing 'label', 'label_string', and 'embedding' columns.

    Returns:
    pd.DataFrame: DataFrame containing IQR-based centroids for each label group.
    """

    def iqr_mean(group):
        embeddings = np.vstack(group)
        q1 = np.percentile(embeddings, 25, axis=0)
        q3 = np.percentile(embeddings, 75, axis=0)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        mask = np.all((embeddings >= lower_bound) & (embeddings <= upper_bound), axis=1)
        return np.mean(embeddings[mask], axis=0)

    iqr_centroids = df.groupby(["label"])["embedding"].apply(iqr_mean)
    iqr_centroid_df = pd.DataFrame({"centroid": iqr_centroids.values})
    iqr_centroid_df[["label"]] = pd.DataFrame(iqr_centroids.index.tolist(), index=iqr_centroid_df.index)
    return iqr_centroid_df


def analyse_embedding_space(df) -> dict:
    centroid_df = compute_centroids(df)

    for label in centroid_df["label"]:
        centroid = centroid_df[centroid_df["label"] == label]["centroid"].values[0]
        embeddings = df[df["label"] == label]["embedding"].tolist()
        distances = cdist(embeddings, [centroid])
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        avg_distance = np.mean(distances)
        centroid_df.loc[centroid_df["label"] == label, "min_distance"] = min_distance
        centroid_df.loc[centroid_df["label"] == label, "max_distance"] = max_distance
        centroid_df.loc[centroid_df["label"] == label, "avg_distance"] = avg_distance

    all_embeddings = df["embedding"].tolist()
    all_dist = cdist(all_embeddings, all_embeddings)

    # Create a mask to exclude the diagonal (self-distances)
    mask = ~np.eye(all_dist.shape[0], dtype=bool)

    all_centroid_dist = cdist(centroid_df["centroid"].tolist(), centroid_df["centroid"].tolist())
    centroid_mask = ~np.eye(all_centroid_dist.shape[0], dtype=bool)

    # Convert embeddings to a numpy array for sklearn functions
    embeddings_array = np.array(all_embeddings)
    labels_array = df["label"].values

    # Calinski-Harabasz Index
    ch_index = calinski_harabasz_score(embeddings_array, labels_array)

    # Within-cluster Sum of Squares (WCSS)
    wcss = sum(centroid_df["avg_distance"] ** 2 * centroid_df["label"].map(df["label"].value_counts()))

    # Silhouette Coefficient
    silhouette_avg = silhouette_score(embeddings_array, labels_array)

    # Davies-Bouldin Index
    db_index = davies_bouldin_score(embeddings_array, labels_array)

    # Dunn Index
    min_inter_cluster_distance = np.min(all_centroid_dist[centroid_mask])
    max_intra_cluster_distance = centroid_df["max_distance"].max()
    dunn_index = min_inter_cluster_distance / max_intra_cluster_distance

    metrics = {
        "global_max_dist": np.max(all_dist[mask]),
        "global_min_dist": np.min(all_dist[mask]),
        "global_avg_dist": np.mean(all_dist[mask]),
        "global_std_dist": np.std(all_dist[mask]),
        "intra_min_dist": centroid_df["min_distance"].min(),
        "intra_max_dist": centroid_df["max_distance"].max(),
        "intra_avg_dist": centroid_df["avg_distance"].mean(),
        "intra_std_dist": centroid_df["avg_distance"].std(),
        "inter_min_dist": np.min(all_centroid_dist[centroid_mask]),
        "inter_max_dist": np.max(all_centroid_dist[centroid_mask]),
        "inter_avg_dist": np.mean(all_centroid_dist[centroid_mask]),
        "inter_std_dist": np.std(all_centroid_dist[centroid_mask]),
        "calinski_harabasz_index": ch_index,
        "wcss": wcss,
        "silhouette_coefficient": silhouette_avg,
        "davies_bouldin_index": db_index,
        "dunn_index": dunn_index,
    }
    return metrics


formatted_names = {
    "global_max_dist": "Pairwise Max Distance",
    "global_min_dist": "Pairwise Min Distance",
    "global_avg_dist": "Pairwise Avg Distance",
    "global_std_dist": "Pairwise Std Dev of Distances",
    "intra_min_dist": "Within-Cluster Min Distance",
    "intra_max_dist": "Within-Cluster Max Distance",
    "intra_avg_dist": "Within-Cluster Avg Distance",
    "intra_std_dist": "Within-Cluster Std Dev of Distances",
    "inter_min_dist": "Between-Cluster Min Distance",
    "inter_max_dist": "Between-Cluster Max Distance",
    "inter_avg_dist": "Between-Cluster Avg Distance",
    "inter_std_dist": "Between-Cluster Std Dev of Distances",
    "calinski_harabasz_index": "Calinski-Harabasz Index",
    "wcss": "Within-Cluster Sum of Squares",
    "silhouette_coefficient": "Silhouette Coefficient",
    "davies_bouldin_index": "Davies-Bouldin Index",
    "dunn_index": "Dunn Index",
}


def format_metrics(metrics):
    global formatted_names
    print("Cluster Evaluation Metrics:")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"{formatted_names[key]}: {value:.4f}")
