import os
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple, Dict
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import math
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from gorillatracker.classification.metrics import analyse_embedding_space


def generate_folds(df, k=5, seed=42):
    """
    Generate label-disjunct folds for each (dataset, model) combination using a local random number generator.

    Args:
    df (pd.DataFrame): Input dataframe with columns 'dataset', 'model', 'label'
    k (int): Number of folds
    seed (int): Random seed for reproducibility

    Returns:
    pd.DataFrame: Copy of input dataframe with an additional 'fold' column
    """
    # Create a copy of the input dataframe
    df_with_folds = df.copy()
    df_with_folds["fold"] = -1  # Initialize fold column

    # Create a local random number generator
    rng = np.random.default_rng(seed)

    # Group by dataset and model
    for (dataset, model), group in df.groupby(["dataset", "model"]):
        # Get unique labels for this group
        unique_labels = group["label"].unique()

        # Shuffle labels using the local RNG
        rng.shuffle(unique_labels)

        # Split labels into k groups
        label_folds = np.array_split(unique_labels, k)

        # Create a mapping from label to fold
        label_to_fold = {}
        for fold, labels in enumerate(label_folds):
            for label in labels:
                label_to_fold[label] = fold

        # Assign folds to rows
        mask = (df_with_folds["dataset"] == dataset) & (df_with_folds["model"] == model)
        df_with_folds.loc[mask, "fold"] = df_with_folds.loc[mask, "label"].map(label_to_fold)

    # Ensure all rows have been assigned a fold
    assert (df_with_folds["fold"] != -1).all(), "Some rows were not assigned a fold"

    return df_with_folds


def split(
    df: pd.DataFrame, k: int = 5, construction_method: str = "equal_classes", seed: int = 42
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split the dataframe into k train-test combinations based on the 'fold' column.
    Then applies the specified construction method to create new classes.

    Args:
    df (pd.DataFrame): Input dataframe with 'fold', 'dataset', 'model', and 'label' columns
    k (int): Number of folds (default 5)
    construction_method (str): Method to construct new classes ('equal_classes' or 'equal_images')
    seed (int): Random seed for reproducibility

    Returns:
    List[Tuple[pd.DataFrame, pd.DataFrame]]: List of (train, test) dataframe pairs
    """
    assert "fold" in df.columns, "Dataframe must have a 'fold' column"
    assert "dataset" in df.columns and "model" in df.columns, "Dataframe must have 'dataset' and 'model' columns"
    assert "label" in df.columns, "Dataframe must have a 'label' column"
    assert (
        len(df["dataset"].unique()) == 1 and len(df["model"].unique()) == 1
    ), "Dataframe should contain only one dataset and model combination"
    MIN_SAMPLES_REMAINING = 2  # for knn5 vote to win (2,1,1,1 nn scenario)
    rng = np.random.default_rng(seed)

    splits = []
    for i in range(k):
        test_df = df[df["fold"] == i].copy()
        train_df = df[df["fold"] != i].copy()
        test_df["is_new"] = True
        train_df["is_new"] = False

        if construction_method == "equal_classes":
            # Create a mask of non-sampleable images
            non_sampleable_mask = pd.Series(False, index=train_df.index)
            for label in train_df["label"].unique():
                label_indices = train_df[train_df["label"] == label].index
                if len(label_indices) <= MIN_SAMPLES_REMAINING:
                    non_sampleable_mask[label_indices] = True
                else:
                    keep_indices = rng.choice(label_indices, size=MIN_SAMPLES_REMAINING, replace=False)
                    non_sampleable_mask[keep_indices] = True

            # Calculate the number of samples to move
            n_samples_to_move = len(test_df)

            # Sample from the sampleable images
            sampleable_df = train_df[~non_sampleable_mask]
            if len(sampleable_df) < n_samples_to_move:
                n_samples_to_move = len(sampleable_df)

            samples_to_move = sampleable_df.sample(n=n_samples_to_move, random_state=rng)

            # Move the samples
            train_df = train_df.drop(samples_to_move.index)
            test_df = pd.concat([test_df, samples_to_move], ignore_index=True)

        elif construction_method == "equal_images":
            n_images_to_move = len(test_df)
            sampleable_df = train_df.groupby("label").filter(lambda x: len(x) > MIN_SAMPLES_REMAINING)

            if len(sampleable_df) < n_images_to_move:
                n_images_to_move = len(sampleable_df)

            samples_to_move = sampleable_df.sample(n=n_images_to_move, random_state=rng)

            train_df = train_df.drop(samples_to_move.index)
            test_df = pd.concat([test_df, samples_to_move], ignore_index=True)

        else:
            raise ValueError(f"Unknown construction method: {construction_method}")

        splits.append((train_df, test_df))

    return splits


def compute_metrics(y_true, y_pred, unique_labels):
    """
    Compute accuracy and macro F1-score for new vs. known classification and multiclass classification among known classes.
    New Class has label -1. The other classes are multiclass.

    Args:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    unique_labels (array-like): List of known class labels

    Returns:
    dict: Dictionary containing computed metrics


    T|F
    1|1 - Correct
    1|2 - Incorrect known
    1|-1 - Incorrect new & new vs. known
    -1|1 - Incorrect known & new vs. known
    -1|-1 - Correct new

    multiclass -> always correct if left equal to right side (no matter if new or known class, it's just classification over n+1)
    """
    # Convert to numpy arrays for easier manipulation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Create binary labels for new vs. known classification
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    binary_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    binary_f1 = f1_score(y_true_binary, y_pred_binary)

    # Compute metrics for multiclass classification (only for known classes)
    mask_true = y_true != -1
    mask_pred = y_pred != -1
    mask = mask_true & mask_pred

    # NOTE(liamvdv): sometimes -1 is not allowed as a label. The behaviour of sklearn does
    # not appear to be consistent on this.
    multiclass = np.unique(np.concatenate([unique_labels, [-1], y_true, y_pred]))
    le = LabelEncoder()
    le.fit(multiclass)
    y_true_encoded = le.transform(y_true)
    y_pred_encoded = le.transform(y_pred)

    # TODO(liamvdv): reconsider if this is the behaviour we want when no match is found. i.d.R. we want NOT to pick this as max, thus assign low precision.
    only_known_accuracy = (
        accuracy_score(y_true_encoded[mask], y_pred_encoded[mask]) if y_true_encoded[mask].size > 0 else 0
    )

    # we will not add the '-1' new class to the labels; this f1 is only about the known classes
    # we want to exclude the 'new' class from the f1 calculation via labels= (note we are NOT using the mask here)
    # this will give us the f1 score over the known classes only [zero_division=1 is used to allow labels to be missing]
    only_known_f1 = (
        f1_score(y_true_encoded, y_pred_encoded, labels=le.transform(unique_labels), average="macro", zero_division=1)
        if not np.all(y_pred == -1)
        else 0
    )

    multiclass_accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    multiclass_f1 = f1_score(y_true_encoded, y_pred_encoded, average="macro")
    multiclass_f1_weighted = f1_score(y_true_encoded, y_pred_encoded, average="weighted")

    # Create a dictionary to store the metrics
    metrics = {
        # how often was new class predicted correctly
        # threshold graph: should start at 1 and go down to 0
        "new_vs_all_accuracy": binary_accuracy,
        # threshold graph: should start at 1 and go down to 0
        "new_vs_all_f1": binary_f1,
        # t=-1 or p=-1 will be excluded
        "only_known_accuracy": only_known_accuracy,
        "only_known_f1": only_known_f1,
        # normal multiclass over n+1 classes
        # threshold graph: should start at
        "multiclass_accuracy": multiclass_accuracy,
        "multiclass_f1": multiclass_f1,
        "multiclass_f1_weighted": multiclass_f1_weighted,
    }

    return metrics


class EmbeddingCentroidCalculator:
    def __init__(self, dataset):
        self.dataset = dataset

    def _filter_outliers_iqr(self, data, q1=25, q3=75, iqr_factor=1.5):
        Q1 = np.percentile(data, q1, axis=0)
        Q3 = np.percentile(data, q3, axis=0)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        mask = np.all((data >= lower_bound) & (data <= upper_bound), axis=1)
        return data[mask]

    def _calculate_centroid(self, group, use_iqr=False):
        embeddings = np.stack(group["embedding"].values)

        if use_iqr:
            embeddings = self._filter_outliers_iqr(embeddings)
        centroid = np.mean(embeddings, axis=0) if embeddings.size > 0 else np.zeros(group["embedding"].iloc[0].shape)
        return pd.Series({"label": group.name, "embedding": centroid})

    def calculate_centroids(self, use_iqr=False):
        return (
            self.dataset.groupby("label").apply(lambda x: self._calculate_centroid(x, use_iqr)).reset_index(drop=True)
        )


class ThresholdedNearestCentroid:
    def __init__(self, threshold, n_neighbors=1):
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.centroids = None
        self.labels = None

    def fit(self, centroids_df):
        self.centroids = np.vstack(centroids_df["embedding"].values)
        self.labels = centroids_df["label"].values
        self.nbrs.fit(self.centroids)
        return self

    def predict(self, query_embeddings):
        distances, indices = self.nbrs.kneighbors(query_embeddings)

        nearest_distances = distances[:, 0]
        predictions = self.labels[indices[:, 0]]

        predictions[nearest_distances > self.threshold] = -1

        return predictions


def knn_openset_recognition(
    dataset: pd.DataFrame,
    queryset: pd.DataFrame,
    thresholds: List[float],
    method: str = "knn1",
    snapshot: Optional[List[float]] = None,
) -> Dict[float, Dict[str, float]]:
    """
    Perform KNN + threshold grid search for open-set recognition with centroid caching.

    Args:
    dataset (pd.DataFrame): Training dataset with 'label' and 'embedding' columns
    queryset (pd.DataFrame): Query dataset with 'label' and 'embedding' columns
    thresholds (List[float]): List of thresholds to search
    method (str): Method to use for classification ('knn1', 'knn5', 'knn1centroid', 'knn1centroid_iqr')
    snapshot (List[float]): List of thresholds to store results for.

    Returns:
    Dict[float, Dict[str, float]]: Results for each threshold
    """
    # Prepare the data
    X_train = np.stack(dataset["embedding"].values)
    y_train = dataset["label"].values
    X_query = np.stack(queryset["embedding"].values)
    y_query = queryset["label"].values

    # DO NOT USE STANDARD SCALER: In represenation learning, we do not scale the embeddings (their distance is important)
    # Instead, we use the raw embeddings. If you were to scale them, we would also need to scale the threshold.
    # scaler = StandardScaler()

    # Fit the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_train)

    # Find the nearest neighbors for queryset
    distances, indices = nbrs.kneighbors(X_query)

    centroider = EmbeddingCentroidCalculator(dataset)

    knn1centroids = centroider.calculate_centroids(use_iqr=False)
    knn1centroids_iqr = centroider.calculate_centroids(use_iqr=True)

    results = {}
    for t in thresholds:
        if method == "knn1":
            predictions = knn1(dataset, indices, distances, t)
        elif method == "knn5":
            predictions = knnK(dataset, indices, distances, t)
        elif method == "knn5distance":
            predictions = knn_distance(dataset, indices, distances, t)
        elif method in ["knn1centroid", "knn1centroid_iqr"]:
            predictions = knn1centroid_generic(
                X_query, t, knn1centroids if method == "knn1centroid" else knn1centroids_iqr
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        results[t] = compute_metrics(y_query, predictions, dataset["label"].unique())

        if snapshot and any(s for s in snapshot if math.isclose(t, s, abs_tol=1e-6)):
            results[t]["y_true"] = y_query
            results[t]["y_pred"] = predictions

    return results


def knn1centroid_generic(query_embeddings, threshold, centroids):
    predictor = ThresholdedNearestCentroid(threshold=threshold, n_neighbors=1)
    predictor.fit(centroids)
    return predictor.predict(query_embeddings)


def knn1(dataset, indices, distances, threshold):
    return knnK(dataset, indices[:, :1], distances[:, :1], threshold)


def knnK(dataset, indices, distances, threshold):
    """note: k is implicitly set to indices.shape[1]"""
    predictions = []
    for idx, dist in zip(indices, distances):
        valid = dataset.iloc[idx][dist <= threshold]
        if not valid.empty:
            prediction = valid["label"].mode()[0]
        else:
            prediction = -1
        predictions.append(prediction)
    return np.array(predictions)


def knn_distance(dataset, indices, distances, threshold):
    """
    note: k is implicitly set to indices.shape[1] (5)
    also note that class imbalance is represented in the summed distance, i.e. far away classes with high density will have a high vote.
    """
    predictions = np.empty(len(indices), dtype=object)
    labels = dataset["label"].values

    for i, (idx, dist) in enumerate(zip(indices, distances)):
        weights = 1 / (1 + dist)
        unique_labels, label_indices = np.unique(labels[idx], return_inverse=True)

        class_votes = np.bincount(label_indices, weights=weights)
        predicted_label_index = np.argmax(class_votes)
        predicted_label = unique_labels[predicted_label_index]

        avg_distance = (
            np.sum(dist[label_indices == predicted_label_index] * weights[label_indices == predicted_label_index])
            / class_votes[predicted_label_index]
        )

        predictions[i] = predicted_label if avg_distance <= threshold else -1

    assert len(predictions) == len(indices)
    return predictions


def make_pickleable(d):
    if isinstance(d, (dict, defaultdict)):
        return {k: make_pickleable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_pickleable(v) for v in d]
    else:
        return d


def run_knn_openset_recognition_cv(
    thresholds: List[float],
    df: pd.DataFrame,
    k_fold: int = 5,
    construction_method: str = "equal_classes",
    seed: int = 42,
    method: str = "knn1",
    snapshots: List[float] = None,
) -> Dict[float, Dict[str, List[float]]]:
    """
    Run KNN open-set recognition with cross-validation.

    Args:
    df (pd.DataFrame): Input dataframe
    k_fold (int): Number of folds for cross-validation
    n, min_samples, max_samples, seed: Parameters for the split function
    thresholds (List[float]): List of thresholds to search
    knn_k (int): Number of neighbors for KNN
    method (str): KNN method to use

    Returns:
    Dict[float, Dict[str, List[float]]]: Cross-validation results for each threshold
    """
    new_label = -1
    assert new_label not in df["label"].unique(), "New label already exists in the dataframe"
    df_foldable = generate_folds(df, k=k_fold, seed=seed)
    splits = split(df_foldable, k=k_fold, construction_method=construction_method, seed=seed)

    cv_results = defaultdict(lambda: defaultdict(list))
    for train_df, test_df in splits:
        classes_total = test_df["label"].nunique()
        classes_new = test_df[test_df["is_new"]]["label"].nunique()
        images_total = test_df.count()
        images_new = test_df[test_df["is_new"]].count()
        # set all test_df labels to new_label if is_new column set
        test_df.loc[test_df["is_new"], "label"] = new_label
        test_df.loc[test_df["is_new"], "label_string"] = "new"
        fold_results = knn_openset_recognition(train_df, test_df, thresholds, method=method, snapshot=snapshots)
        for t, metrics in fold_results.items():
            for metric, value in metrics.items():
                cv_results[t][metric].append(value)

                cv_results[t]["count_queryset_images_new"].append(images_new)
                cv_results[t]["count_queryset_classes_new"].append(classes_new)
                cv_results[t]["count_queryset_images_total"].append(images_total)
                cv_results[t]["count_queryset_classes_total"].append(classes_total)

    return make_pickleable(cv_results)


def visualize_metrics(cv_results: Dict[float, Dict[str, List[float]]], thresholds: List[float]):
    """
    Visualize metrics from cross-validation results, handling None values,
    and combining count_* metrics into a single subplot.

    Args:
    cv_results (Dict[float, Dict[str, List[float]]]): Cross-validation results
    thresholds (List[float]): List of thresholds used

    Returns:
    None (displays the plot)
    """
    metrics = list(cv_results[thresholds[0]].keys())
    count_metrics = [m for m in metrics if m.startswith("count_")]
    other_metrics = [m for m in metrics if not m.startswith("count_")]

    num_metrics = len(other_metrics) + 1  # +1 for the combined count metrics
    num_cols = 3
    num_rows = math.ceil(num_metrics / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle("Metrics across Different Thresholds", fontsize=16)

    # Flatten axes array for easier indexing
    axes = axes.flatten() if num_rows > 1 else [axes]

    # Plot other metrics
    for i, metric in enumerate(other_metrics):
        ax = axes[i]

        valid_thresholds = []
        mean_values = []
        std_values = []

        for t in thresholds:
            values = [v for v in cv_results[t][metric] if v is not None]
            if values:
                valid_thresholds.append(t)
                mean_values.append(np.mean(values))
                std_values.append(np.std(values))

        if valid_thresholds:
            ax.plot(valid_thresholds, mean_values, marker="o")
            ax.fill_between(
                valid_thresholds,
                [m - s for m, s in zip(mean_values, std_values)],
                [m + s for m, s in zip(mean_values, std_values)],
                alpha=0.2,
            )

        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, linestyle="--", alpha=0.7)

    # Plot combined count metrics
    ax = axes[len(other_metrics)]
    ax.set_title("Count Metrics")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")

    for metric in count_metrics:
        mean_values = []
        std_values = []

        for t in thresholds:
            values = [v for v in cv_results[t][metric] if v is not None]
            if values:
                mean_values.append(np.mean(values))
                std_values.append(np.std(values))
            else:
                mean_values.append(np.nan)
                std_values.append(np.nan)

        ax.plot(thresholds, mean_values, marker="o", label=metric)
        ax.fill_between(
            thresholds,
            [m - s for m, s in zip(mean_values, std_values)],
            [m + s for m, s in zip(mean_values, std_values)],
            alpha=0.2,
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Remove any unused subplots
    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def get_optimal_threshold(
    cv_results: Dict[float, Dict[str, List[float]]], metric: str = "multiclass_f1", stability_weight: float = 0.2
) -> float:
    """
    Find the optimal threshold based on the cross-validation results.

    Args:
    cv_results (Dict[float, Dict[str, List[float]]]): Cross-validation results
    metric (str): Metric to optimize. Options: 'multiclass_f1', 'new_vs_all_f1', 'only_known_f1'
    stability_weight (float): Weight given to the stability of the metric across folds (0 to 1)

    Returns:
    float: Optimal threshold
    """
    thresholds = list(cv_results.keys())
    scores = []

    for threshold in thresholds:
        metric_values = [v for v in cv_results[threshold][metric] if v is not None]
        if not metric_values:
            scores.append(0.0)
            continue
        mean_score = np.mean(metric_values)
        std_score = np.std(metric_values)

        # Combine mean and stability (inverse of std) into a single score
        combined_score = (1 - stability_weight) * mean_score + stability_weight * (1 / (1 + std_score))
        scores.append(combined_score)

    optimal_index = np.argmax(scores)
    optimal_threshold = thresholds[optimal_index]

    print(f"Optimal threshold for {metric}: {optimal_threshold}")
    mean = np.mean(cv_results[optimal_threshold][metric]) if any(cv_results[optimal_threshold][metric]) else 0
    std = np.std(cv_results[optimal_threshold][metric]) if any(cv_results[optimal_threshold][metric]) else 0
    print(f"Mean {metric} at optimal threshold: {mean:.4f}")
    print(f"Std {metric} at optimal threshold: {std:.4f}")

    return optimal_threshold


warnings.filterwarnings("error", category=RuntimeWarning, message="invalid value encountered in scalar divide")


def thresholds_selector(df, n_measures=50):
    analysis = analyse_embedding_space(df)
    max_distance = analysis["global_max_dist"]
    min_distance = analysis["global_min_dist"]
    thresholds = np.concatenate([[0], np.linspace(min_distance, max_distance + 1, n_measures)])
    return thresholds


def sweep_configs(dataframe, configs, resolution=50, cache_dir=None):
    """
    Make the sweep faster but also less accurate by choosing lower grid search resolution.
    Now includes caching at the config level, with resolution as part of the cache key.
    """
    print(f"Running {len(configs)} configurations at resolution {resolution}...")
    results = {}

    for config in tqdm(configs):
        dataset, model, labelling_method, construction_method = config

        # Create a unique cache file name for this configuration, including resolution
        cache_file = f"{dataset}_{model}_{labelling_method}_{construction_method}_res{resolution}.pkl"

        if cache_dir and os.path.exists(os.path.join(cache_dir, cache_file)):
            # Load cached results if they exist
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                results[config] = pickle.load(f)
            print(f"Loaded cached results for {config} at resolution {resolution}")
            continue

        print(
            f"Started sweep for dataset '{dataset}', model '{model}', "
            f"labelling method '{labelling_method}', and selector method '{construction_method}'"
        )

        # Filter the dataframe for the current dataset and model
        df = dataframe[(dataframe["dataset"] == dataset) & (dataframe["model"] == model)]

        if df.empty:
            print(f"No data found for dataset '{dataset}' and model '{model}'. Skipping...")
            continue

        # Generate thresholds
        thresholds = thresholds_selector(df, n_measures=resolution)

        # Run cross-validation
        cv_results = run_knn_openset_recognition_cv(
            thresholds=thresholds,
            df=df,
            construction_method=construction_method,
            method=labelling_method,
            k_fold=5,
            seed=42,
        )

        # Get optimal threshold
        optimal_threshold = get_optimal_threshold(cv_results, metric="multiclass_f1")

        # Store results
        results[config] = {
            "cv_results": cv_results,
            "optimal_threshold": optimal_threshold,
            "thresholds": thresholds,
            "resolution": resolution,  # Include resolution in the results
        }

        # Cache the results if a cache directory is provided
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(results[config], f)

        print(
            f"Completed sweep for dataset '{dataset}', model '{model}', "
            f"labelling method '{labelling_method}', and selector method '{construction_method}'"
        )

    return results


def batch_visualize_metrics(results: Dict[Tuple, Dict], configs: List[Tuple]):
    """
    Visualize metrics from cross-validation results for multiple configurations,
    with lighter standard deviation shading.

    Args:
    results (Dict[Tuple, Dict]): Results dictionary from sweep_configs
    configs (List[Tuple]): List of configurations to visualize

    Returns:
    None (displays the plots)
    """
    colors = plt.cm.rainbow(np.linspace(0, 1, len(configs)))

    _one_result = results[configs[0]]["cv_results"]
    metric_names = list(_one_result[list(_one_result.keys())[0]].keys())
    all_metrics = [m for m in metric_names if not m.startswith("count_")]

    num_metrics = len(all_metrics) + 1
    num_cols = 1
    num_rows = math.ceil(num_metrics / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 6 * num_rows))
    fig.suptitle("Metrics across Different Thresholds and Configurations", fontsize=16)

    axes = axes.flatten() if num_rows > 1 else [axes]

    for i, metric in enumerate(all_metrics):
        ax = axes[i]

        for config, color in zip(configs, colors):
            dataset, model, labelling_method, construction_method = config
            cv_results = results[config]["cv_results"]
            thresholds = results[config]["thresholds"]

            valid_thresholds = []
            mean_values = []
            std_values = []

            for t in thresholds:
                values = [v for v in cv_results[t][metric] if v is not None]
                if values:
                    valid_thresholds.append(t)
                    mean_values.append(np.mean(values))
                    std_values.append(np.std(values))

            if valid_thresholds:
                ax.plot(
                    valid_thresholds,
                    mean_values,
                    color=color,
                    label=f"{dataset}-{model}-{labelling_method}-{construction_method}",
                )
                ax.fill_between(
                    valid_thresholds,
                    [m - s for m, s in zip(mean_values, std_values)],
                    [m + s for m, s in zip(mean_values, std_values)],
                    alpha=0.05,  # Reduced alpha for lighter shading
                    color=color,
                )

        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

    # Plot combined count metrics
    ax = axes[len(all_metrics)]
    ax.set_title("Count Metrics")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")

    count_metrics = [m for m in metric_names if m.startswith("count_")]

    for config, color in zip(configs, colors):
        dataset, model, labelling_method, construction_method = config
        cv_results = results[config]["cv_results"]
        thresholds = results[config]["thresholds"]

        for metric in count_metrics:
            mean_values = []
            std_values = []

            for t in thresholds:
                values = [v for v in cv_results[t][metric] if v is not None]
                if values:
                    mean_values.append(np.mean(values))
                    std_values.append(np.std(values))
                else:
                    mean_values.append(np.nan)
                    std_values.append(np.nan)

            ax.plot(
                thresholds,
                mean_values,
                color=color,
                linestyle="--",
                label=f"{dataset}-{model}-{labelling_method}-{construction_method}-{metric}",
            )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    ax.grid(True, linestyle="--", alpha=0.7)

    for i in range(num_metrics, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


labelling_methods = ["knn1", "knn5", "knn5distance", "knn1centroid", "knn1centroid_iqr"]
construction_methods = ["equal_classes", "equal_images"]
from gorillatracker.classification.clustering import EXT_MERGED_DF

df = EXT_MERGED_DF
configs = [
    (dataset, model, labelling_method, construction_method)
    for (dataset, model), _ in df.groupby(["dataset", "model"])
    for labelling_method in labelling_methods
    for construction_method in construction_methods
]
