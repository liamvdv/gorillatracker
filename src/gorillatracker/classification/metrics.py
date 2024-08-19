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
    """Looks more difficult than is. It just needs to handle unclassified points (label -1) and only 0/1 cluster"""
    all_embeddings = np.array(df["embedding"].tolist())
    labels = df["label"].values

    # Separate clustered and unclustered points
    clustered_mask = labels >= 0
    clustered_embeddings = all_embeddings[clustered_mask]
    clustered_labels = labels[clustered_mask]

    # Calculate the proportion of unclustered points
    total_points = len(labels)
    unclustered_points = np.sum(labels == -1)
    unclustered_ratio = unclustered_points / total_points if total_points > 0 else None

    all_dist = cdist(all_embeddings, all_embeddings)
    mask = ~np.eye(all_dist.shape[0], dtype=bool)

    metrics = {
        "unclustered_ratio": unclustered_ratio,
        "global_max_dist": np.max(all_dist[mask]) if len(all_embeddings) > 1 else None,
        "global_min_dist": np.min(all_dist[mask]) if len(all_embeddings) > 1 else None,
        "global_avg_dist": np.mean(all_dist[mask]) if len(all_embeddings) > 1 else None,
        "global_std_dist": np.std(all_dist[mask]) if len(all_embeddings) > 1 else None,
        "intra_min_dist": None,
        "intra_max_dist": None,
        "intra_avg_dist": None,
        "intra_std_dist": None,
        "inter_min_dist": None,
        "inter_max_dist": None,
        "inter_avg_dist": None,
        "inter_std_dist": None,
        "calinski_harabasz_index": None,
        "wcss": None,
        "silhouette_coefficient": None,
        "davies_bouldin_index": None,
        "dunn_index": None,
    }

    # If there are no clustered points or only one point, return metrics with default None values
    if len(clustered_embeddings) <= 1:
        return metrics

    # Compute intra-cluster metrics
    unique_labels = np.unique(clustered_labels)
    intra_distances = []
    centroids = []

    for label in unique_labels:
        cluster_points = clustered_embeddings[clustered_labels == label]
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
        distances = cdist(cluster_points, [centroid]).flatten()
        intra_distances.extend(distances)

    intra_distances = np.array(intra_distances)

    metrics.update(
        {
            "intra_min_dist": np.min(intra_distances),
            "intra_max_dist": np.max(intra_distances),
            "intra_avg_dist": np.mean(intra_distances),
            "intra_std_dist": np.std(intra_distances),
            "wcss": np.sum(intra_distances**2),  # Within-cluster Sum of Squares
        }
    )

    # Compute metrics that require at least two clusters
    if len(unique_labels) > 1:
        metrics.update(
            {
                "calinski_harabasz_index": calinski_harabasz_score(clustered_embeddings, clustered_labels),
                "silhouette_coefficient": silhouette_score(clustered_embeddings, clustered_labels),
                "davies_bouldin_index": davies_bouldin_score(clustered_embeddings, clustered_labels),
            }
        )

        # Compute inter-cluster metrics
        centroids = np.array(centroids)
        inter_distances = cdist(centroids, centroids)
        inter_mask = ~np.eye(inter_distances.shape[0], dtype=bool)

        metrics.update(
            {
                "inter_min_dist": np.min(inter_distances[inter_mask]),
                "inter_max_dist": np.max(inter_distances[inter_mask]),
                "inter_avg_dist": np.mean(inter_distances[inter_mask]),
                "inter_std_dist": np.std(inter_distances[inter_mask]),
                "dunn_index": (
                    np.min(inter_distances[inter_mask]) / np.max(intra_distances)
                    if np.max(intra_distances) > 0
                    else None
                ),
            }
        )

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
