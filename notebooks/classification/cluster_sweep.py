# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import adjusted_rand_score, confusion_matrix, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from gorillatracker.classification.metrics import analyse_embedding_space
from scipy.optimize import linear_sum_assignment


def calculate_metrics(embeddings: np.ndarray, labels: np.ndarray, true_labels: np.ndarray) -> dict:
    """wraps analyse_embedding_space and adds class-weighted F1 score and precision"""
    assert len(labels) == len(true_labels) == len(embeddings)
    df = pd.DataFrame({"embedding": embeddings.tolist(), "label": labels.tolist()})
    metrics = analyse_embedding_space(df)

    # "label matching problem" in clustering evaluation
    matched_labels = match_labels(true_labels, labels)

    # Compute class-weighted F1 score
    f1 = f1_score(true_labels, matched_labels, average="weighted")

    # Compute class-weighted precision
    precision = precision_score(true_labels, matched_labels, average="weighted")

    # NOTE(liamvdv): not over matched labels, can handle arbitrary cluster labels
    # https://en.wikipedia.org/wiki/Rand_index#/media/File:Example_for_Adjusted_Rand_index.svg
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html
    # The adjusted Rand index is thus ensured to have a value close to 0.0 for random labeling independently of the number of clusters and samples and exactly 1.0 when the clusterings are identical (up to a permutation). The adjusted Rand index is bounded below by -0.5 for especially discordant clusterings.
    ars = adjusted_rand_score(true_labels, labels)

    metrics.update({"weighted_f1_score": f1, "weighted_precision": precision, "adjusted_rand_score": ars})
    return metrics


def match_labels(true_labels, predicted_labels):
    """
    Match predicted cluster labels to true labels using the Hungarian algorithm.

    NOTE(liamvdv): Necessary because cluster labels are arbitrary and may not match the true labels but represent the same clusters.
    """
    assert len(true_labels) == len(predicted_labels)

    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Use the Hungarian algorithm to find the best matching
    row_ind, col_ind = linear_sum_assignment(-cm)

    label_mapping = {pred: true for pred, true in zip(col_ind, row_ind)}

    matched_labels = np.array([label_mapping.get(label, label) for label in predicted_labels])

    return matched_labels


# %%
import matplotlib.pyplot as plt
import math
import pandas as pd


def visualize_alg_metrics(result_df, dataset, model, algorithm, formatted_names):
    """
    Create a grid of charts where every metric is shown for the specified dataset, model, and algorithm.
    The x-axis values are determined based on the algorithm used.
    Includes a separate figure showing the n_true_clusters value.

    Parameters:
    result_df (pd.DataFrame): DataFrame containing metrics for all runs
    dataset (str): The dataset to filter by
    model (str): The model to filter by
    algorithm (str): The algorithm to filter by
    formatted_names (dict): Dictionary mapping metric names to formatted display names

    Returns:
    None (displays the plot)
    """
    # Filter the DataFrame
    filtered_df = result_df[
        (result_df["dataset"] == dataset) & (result_df["model"] == model) & (result_df["algorithm"] == algorithm)
    ]

    if filtered_df.empty:
        print(f"No data found for dataset={dataset}, model={model}, algorithm={algorithm}")
        return

    # Get the list of metrics (excluding certain columns)
    exclude_cols = ["dataset", "model", "algorithm", "algorithm_params", "n_clusters", "n_true_clusters"]
    metrics = [col for col in filtered_df.columns if col not in exclude_cols and not col.startswith("global_")]

    # Determine x-axis values based on the algorithm
    if algorithm in ["AgglomerativeClustering", "KMeans"]:
        filtered_df["x_value"] = filtered_df["algorithm_params"].apply(lambda x: x["n_clusters"])
        x_label = "Number of Clusters"
    elif algorithm == "HDBSCAN":
        filtered_df["x_value"] = filtered_df["algorithm_params"].apply(lambda x: x["min_cluster_size"])
        x_label = "Min Cluster Size"
    elif algorithm == "DBSCAN":
        filtered_df["x_value"] = filtered_df["algorithm_params"].apply(lambda x: f"{x['eps']:.2f}, {x['min_samples']}")
        x_label = "Eps, Min Samples"
    else:
        print(f"Unsupported algorithm: {algorithm}")
        return

    # Calculate the grid dimensions
    n_metrics = len(metrics) + 1  # +1 for n_true_clusters
    n_cols = 5  # You can adjust this for a different layout
    n_rows = math.ceil(n_metrics / n_cols)

    # Create the plot with extra space at the top
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows + 1))  # Added 1 to the height

    # Flatten the axs array for easier indexing
    axs = axs.flatten()

    # Plot each metric
    for i, metric in enumerate(metrics):
        x = filtered_df["x_value"]
        y = filtered_df[metric]

        axs[i].plot(x, y, marker="o")
        axs[i].set_title(formatted_names.get(metric, metric))
        axs[i].set_xlabel(x_label)
        axs[i].set_ylabel("Value")
        axs[i].grid(True)

        # Rotate x-axis labels for better readability, especially for DBSCAN
        if algorithm == "DBSCAN":
            axs[i].set_xticklabels(x, rotation=45, ha="right")

    # Add n_true_clusters as a separate figure
    n_true_clusters = filtered_df["n_true_clusters"].iloc[0]  # Assuming it's the same for all rows
    axs[i + 1].text(
        0.5,
        0.5,
        f"True Number of Clusters: {n_true_clusters}",
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=14,
    )
    axs[i + 1].axis("off")  # Hide axes for this text-only figure

    # Remove any unused subplots
    for j in range(i + 2, len(axs)):
        fig.delaxes(axs[j])

    # Add suptitle with padding
    plt.suptitle(f"Metrics for {dataset} - {model} - {algorithm}", fontsize=16, y=0.98)

    # Adjust the layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Adjust top padding

    plt.show()


def with_min_label_count(df: pd.DataFrame, min: int) -> pd.DataFrame:
    """
    Create a copy of the DataFrame, keeping only rows where the label appears
    at least 'min' times in the original DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must have a 'label' column.
    min (int): Minimum number of occurrences for a label to be included.

    Returns:
    pd.DataFrame: A new DataFrame with filtered rows.

    Raises:
    ValueError: If 'label' column is not present in the DataFrame.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column")

    # Count label occurrences
    label_counts = df["label"].value_counts()

    # Get labels that appear at least 'min' times
    valid_labels = label_counts[label_counts >= min].index

    # Create a new DataFrame with only the valid labels
    filtered_df = df[df["label"].isin(valid_labels)].copy()

    return filtered_df


def upto_max_label_count(df: pd.DataFrame, max: int, seed: int = 42) -> pd.DataFrame:
    """
    Create a copy of the DataFrame, randomly sampling rows for labels that exceed
    the specified maximum count, while keeping all rows for labels that don't exceed the maximum.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must have a 'label' column.
    max (int): Maximum number of occurrences for each label.
    seed (int): Random seed for reproducibility.

    Returns:
    pd.DataFrame: A new DataFrame with sampled rows.

    Raises:
    ValueError: If 'label' column is not present in the DataFrame.
    """
    if "label" not in df.columns:
        raise ValueError("DataFrame must have a 'label' column")

    # Create a local random number generator
    rng = np.random.default_rng(seed)

    # Group the DataFrame by label
    grouped = df.groupby("label")

    sampled_dfs = []

    for label, group in grouped:
        if len(group) > max:
            # Randomly sample 'max' rows from the group
            sampled_group = group.sample(n=max, random_state=rng)
        else:
            # Keep all rows if the count doesn't exceed max
            sampled_group = group

        sampled_dfs.append(sampled_group)

    # Concatenate all sampled groups
    if len(sampled_dfs) == 0:
        return pd.DataFrame(columns=df.columns)
    else:
        return pd.concat(sampled_dfs).reset_index(drop=True)


# %%
MERGED_DF = pd.read_pickle("merged.pkl")
# %% [markdown]
# # Sweep


# %%
def extend_merge_df(df, name: str, min=None, max=None):
    lo = with_min_label_count(df, min) if isinstance(min, int) else df
    hi = upto_max_label_count(lo, max) if isinstance(max, int) else lo
    hi["dataset"] = name
    return hi


df = MERGED_DF
spac_min3_max10 = [
    extend_merge_df(df[(df["model"] == model) & (df["dataset"] == "SPAC")], "SPAC+min3+max10", min=3, max=10)
    for model in df["model"].unique().tolist()
]
spac_min3 = [
    extend_merge_df(df[(df["model"] == model) & (df["dataset"] == "SPAC")], "SPAC+min3", min=3)
    for model in df["model"].unique().tolist()
]
bristol_min25max25 = [
    # 25 because every individual has 25 samples
    extend_merge_df(df[(df["model"] == model) & (df["dataset"] == "Bristol")], "Bristol+min25+max25", min=25, max=25)
    for model in df["model"].unique().tolist()
]
dfs = [df, *spac_min3_max10, *spac_min3, *bristol_min25max25]
nonempty_dfs = [df for df in dfs if len(df) > 0]
EXT_MERGED_DF = pd.concat(nonempty_dfs, ignore_index=True)

# %%
import itertools
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering, HDBSCAN
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import warnings

# NOTE(liamvdv): configure the speed level here
speed_level = "fast"


speed_levels = [None, "detailed", "fast", "veryfast"]


def speed(*args):
    """pass upto 3 arguments; the later ones will should indicate more resource usage"""
    global speed_level, speed_levels
    speed = speed_levels.index(speed_level)
    assert speed != -1, f"Invalid speed level: '{speed_level}'. Must be one of {speed_levels}"
    return args[:speed][-1]


def param_grid(params):
    param_combinations = [dict(zip(params.keys(), values)) for values in itertools.product(*params.values())]
    return param_combinations


def sweep_clustering_algorithms(df, configs):
    results = []

    for dataset, model, algorithm, param_combinations in tqdm(configs, desc="Processing configurations"):
        print("Processing", dataset, model, algorithm)
        # Filter the dataframe
        subset = df[(df["model"] == model) & (df["dataset"] == dataset)]
        subset = subset.reset_index(drop=True)

        if subset.empty:
            raise ValueError(f"No data found for {dataset} - {model}")

        embeddings = np.stack(subset["embedding"].to_numpy())
        true_labels = subset["label"].to_numpy()

        # Scale the embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)

        for params in param_combinations:
            print("Processing", dataset, model, algorithm, params)
            if algorithm == "KMeans":
                clusterer = KMeans(random_state=42, **params)
            elif algorithm == "AgglomerativeClustering":
                clusterer = AgglomerativeClustering(**params)
            elif algorithm == "HDBSCAN":
                clusterer = HDBSCAN(**params)
            elif algorithm == "DBSCAN":
                clusterer = DBSCAN(**params)
            elif algorithm == "GaussianMixture":
                clusterer = GaussianMixture(random_state=42, **params)
            elif algorithm == "SpectralClustering":
                clusterer = SpectralClustering(random_state=42, **params)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")

            labels = clusterer.fit_predict(scaled_embeddings)

            metric = calculate_metrics(scaled_embeddings, labels, true_labels)
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
            results.append(metric)

    return pd.DataFrame(results)


configs = []

# Add configurations for correctness assuring synthetic datasets/models
synthetic = [
    ("Synthetic 200c 10n", "Synthetic", "KMeans", param_grid({"n_clusters": range(2, 401, 20)})),
    ("Synthetic 200c 10n", "Synthetic", "AgglomerativeClustering", param_grid({"n_clusters": range(2, 401, 20)})),
    (
        "Synthetic 200c 10n",
        "Synthetic",
        "HDBSCAN",
        param_grid({"min_cluster_size": range(2, 21), "min_samples": [2]}),
    ),
    (
        "Synthetic 200c 10n",
        "Synthetic",
        "DBSCAN",
        param_grid({"eps": np.arange(0.1, 2.1, 0.2), "min_samples": [2, 4, 8]}),
    ),
    ("Synthetic 20c 20n", "Synthetic", "KMeans", param_grid({"n_clusters": range(2, 44, 2)})),
    ("Synthetic 20c 20n", "Synthetic", "AgglomerativeClustering", param_grid({"n_clusters": range(2, 44, 2)})),
    (
        "Synthetic 20c 20n",
        "Synthetic",
        "HDBSCAN",
        param_grid({"min_cluster_size": range(2, 10), "min_samples": [2]}),
    ),
    (
        "Synthetic 20c 20n",
        "Synthetic",
        "DBSCAN",
        param_grid({"eps": np.arange(0.1, 2.1, 0.2), "min_samples": [2, 4, 8]}),
    ),
    # ("Synthetic 200c 10n", "Synthetic", "GaussianMixture", flatten_grid({"n_components": range(2, 401, 20), "covariance_type": ["full", "tied", "diag", "spherical"]})),
    # ("Synthetic 200c 10n", "Synthetic", "SpectralClustering" ,flatten_grid( {"n_clusters": range(2, 401, 20), "affinity": ["rbf", "nearest_neighbors"]}))
]
configs.extend(synthetic)

# Add SPAC dataset configurations
models = ["ViT-Finetuned", "ViT-Pretrained", "EfN-Pretrained"]  # TODO(liamvdv): + ["EfN-Finetuned"]
spac = [
    config
    for model in models
    for ds in ["SPAC", "SPAC+min3", "SPAC+min3+max10"]
    for config in [
        (ds, model, "KMeans", param_grid({"n_clusters": range(2, 181, speed(1, 5, 20))})),
        (ds, model, "AgglomerativeClustering", param_grid({"n_clusters": range(2, 181, speed(1, 5, 20))})),
        (ds, model, "HDBSCAN", param_grid({"min_cluster_size": [2]})),
        # ("SPAC", model, "DBSCAN", param_grid({"eps": [0.1, 0.5, 1.0], "min_samples": [2, 4, 8]})),
        # ("SPAC", "ViT-Finetuned", "GaussianMixture", flatten_grid({"n_components": range(2, 181, 5), "covariance_type": ["full", "tied", "diag", "spherical"]})),
        # ("SPAC", "ViT-Finetuned", "SpectralClustering", flatten_grid({"n_clusters": range(2, 181, 5), "affinity": ["rbf", "nearest_neighbors"]})),
    ]
]
configs.extend(spac)

# Add Bristol dataset configurations
bristol = [
    config
    for model in models
    for ds in ["Bristol", "Bristol+min25+max25"]
    for config in [
        (ds, model, "KMeans", param_grid({"n_clusters": range(2, 50, speed(1, 5))})),
        (ds, model, "AgglomerativeClustering", param_grid({"n_clusters": range(2, 50, speed(1, 5))})),
        (ds, model, "HDBSCAN", param_grid({"min_cluster_size": [2]})),
        # ("Bristol", model, "DBSCAN", param_grid({"eps": [0.1, 0.5, 1.0], "min_samples": [2, 4, 8]})),
        # ("Bristol", model, "GaussianMixture", flatten_grid({"n_components": range(2, 181, 5), "covariance_type": ["full", "tied", "diag", "spherical"]})),
        # ("Bristol", model, "SpectralClustering", flatten_grid({"n_clusters": range(2, 181, 5), "affinity": ["rbf", "nearest_neighbors"]})),
    ]
]
configs.extend(bristol)


print("Number of configurations:", len(configs))
print("Number of algorithm configurations:", sum(len(params) for _, _, _, params in configs))

results_df = sweep_clustering_algorithms(EXT_MERGED_DF, configs)
results_df.to_pickle("results.pkl")
