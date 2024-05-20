from gorillatracker.utils.embedding_generator import generate_embeddings_from_run, read_embeddings_from_disk
from gorillatracker.scripts.visualize_embeddings import EmbeddingProjector
import numpy as np
import pandas as pd
from gorillatracker.clustering.folds import DistinctClassFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    average_precision_score,
    pairwise_distances,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    RocCurveDisplay,
)
import tqdm
import matplotlib.pyplot as plt
from gorillatracker.metrics import knn
from collections import defaultdict



# Multi-Label Model Evaluation
# https://www.kaggle.com/code/kmkarakaya/multi-label-model-evaluation


def norm_label_distribution(df, label_column, seed=42):
    at_least = 2
    df = df.groupby(label_column).filter(lambda x: len(x) >= at_least).reset_index(drop=True)
    distribution = df[label_column].value_counts()
    is_biased = (distribution.max() - distribution.min()) / len(df[label_column]) > 0.03
    if is_biased:

        def downsampler(x):
            down_lo = distribution.min()
            down_hi = distribution.min() * 2
            n = len(x)
            sample_n = max(down_lo, min(n, down_hi))
            return x.sample(sample_n, random_state=seed)

        df = df.groupby(label_column).apply(downsampler).reset_index(drop=True)
        distribution = df[label_column].value_counts()
        assert (distribution.max() - distribution.min()) / len(
            df[label_column]
        ) <= 0.03, f"Labels are not equally distributed {distribution}, pass normalize_label_distribution=True to normalize. Note that this will remove some samples."
        print(distribution)
    return df


"""
Use sth. other than grid search:
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
"""

def knn1(dataset: pd.DataFrame, queryset: pd.DataFrame, thresholds: list[float]):
    """
    returns a dictionary of metric results for each threshold
    """
    ground_truth_matrix, classification_matrix, distances, unique_labels = knn_with_confidence_score(dataset, queryset, [1.0])
    
    new_index = unique_labels.tolist().index("new")
    non_new_indices = [i for i, label in enumerate(unique_labels) if label != "new"]
    
    results = defaultdict(lambda: defaultdict(list))
    for threshold in tqdm.tqdm(thresholds, "Grid Search Thresholds"):
        prediction_matrix = classification_matrix
        for slug, Y_true, Y_pred in [
            ("all", ground_truth_matrix, prediction_matrix),
            ("new", ground_truth_matrix[:, new_index], prediction_matrix[:, new_index]),
            ("non_new", ground_truth_matrix[:, non_new_indices], prediction_matrix[:, non_new_indices]),
        ]:
            results[threshold][f"{slug}/f1"].append(
                f1_score(Y_true, Y_pred, average="binary")
            )
            results[threshold][f"{slug}/roc_auc_score"].append(roc_auc_score(Y_true, Y_pred, labels=unique_labels))
    return results


def geometric_sequence(k: int):
    return [1/2**i for i in range(1, k+1)]
    
    
def knn_weighted_by_geometric_sequence(dataset: pd.DataFrame, queryset: pd.DataFrame, thresholds: list[float], k: int = 5):
    ground_truth_matrix, classification_matrix, distances, unique_labels = knn_with_confidence_score(dataset, queryset, geometric_sequence(k))

    new_index = unique_labels.tolist().index("new")
    non_new_indices = [i for i, label in enumerate(unique_labels) if label != "new"]
    
    results = defaultdict(lambda: defaultdict(list))
    for threshold in tqdm.tqdm(thresholds, "Grid Search Thresholds"):
        prediction_matrix = [
            "new" if distances[i][[i]] > threshold else predicted_label
            for i, predicted_label in enumerate(classification_matrix)
        ]
        for slug, Y_true, Y_pred in [
            ("all", ground_truth_matrix, prediction_matrix),
            ("new", ground_truth_matrix[:, new_index], prediction_matrix[:, new_index]),
            ("non_new", ground_truth_matrix[:, non_new_indices], prediction_matrix[:, non_new_indices]),
        ]:
            results[threshold][f"{slug}/f1"].append(
                f1_score(Y_true, Y_pred, average="binary" if slug == "new" else "weighted")
            )
            results[threshold][f"{slug}/roc_auc_score"].append(roc_auc_score(Y_true, Y_pred, labels=unique_labels))

def knn_with_confidence_score(dataset: pd.DataFrame, queryset: pd.DataFrame, weights: list[float]):
    assert len(weights) > 0
    k = len(weights)
    dataset = np.stack(dataset["embedding"].values)
    query = np.stack(queryset["embedding"].values)
    
    # Binarize the labels
    actual_labels = queryset["label"].values
    combined_labels = np.concatenate([actual_labels, ["new"]])
    unique_labels = np.unique(combined_labels)
    # NOTE(liamvdv): [n_samples, n_classes]
    ground_truth_matrix = label_binarize(actual_labels, classes=unique_labels)
    
    distances = pairwise_distances(query, dataset)
    k_closest_indices = np.argsort(distances, axis=1)[:, :k]
    predicted_labels = dataset.iloc[k_closest_indices]["label"].values
    binary_classmatch_matrix = label_binarize(predicted_labels, classes=unique_labels)
    classification_matrix = binary_classmatch_matrix * weights
    
    return ground_truth_matrix, classification_matrix, distances, unique_labels
    

def find_threshold(
    df: pd.DataFrame,
    label_column: str,
    grid_start: float,
    grid_end: float,
    grid_num: float,
    unique_percentage=0.2,
    seed=42,
    normalize_label_distribution: bool = False,
):
    """You are responsible for normalizing the label distribution."""
    if normalize_label_distribution:
        df = norm_label_distribution(df, label_column, seed)

    # Thresholds for grid search
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    # For storing results
    results = defaultdict(lambda: defaultdict(list))
    kf = DistinctClassFold(n_buckets=5, shuffle=True, random_state=seed)

    # TODO(liamvdv): handle no sufficient match (singletons cannot have match)
    new_label = "new"
    for dataset_df, query_df in kf.split(
        df,
        label_column=label_column,
        label_mask=new_label,
        unique_percentage=unique_percentage,
        normalize_label_distribution=normalize_label_distribution,
    ):
        # Compute pairwise distances between query and training embeddings
        query = np.stack(query_df["embedding"].values)
        dataset = np.stack(dataset_df["embedding"].values)
        distances = pairwise_distances(query, dataset)
        # print("DIST", np.max(distances), np.min(distances), np.mean(distances))
        closest_indices = np.argmin(distances, axis=1)
        perc_of_new_label = query_df[label_column].value_counts().get(new_label, 0) / len(query_df[label_column])

        # Now you can get the predicted label based on the closest train embedding
        predicted_labels = dataset_df.iloc[closest_indices][label_column].values
        actual_labels = query_df[label_column].values

        # Binarize the labels for mAP calculation
        combined_labels = np.concatenate([actual_labels, [new_label]])
        unique_labels = np.unique(combined_labels)
        # NOTE(liamvdv): creates a [len(actual_labels), len(unique_labels)] classification matrix
        #                Values are 0 or 1, single 1 rest 0 per row.
        # [n_samples, n_classes]
        ground_truth_matrix = label_binarize(actual_labels, classes=unique_labels)

        for threshold in tqdm.tqdm(thresholds, "Grid Search Thresholds"):
            thresholded_labels = [
                new_label if distances[i][closest_indices[i]] > threshold else predicted_label
                for i, predicted_label in enumerate(predicted_labels)
            ]
            # Binarize the labels for mAP calculation
            prediction_matrix = label_binarize(thresholded_labels, classes=unique_labels)

            # Calculate mAP for the 'new' label
            assert new_label in unique_labels
            new_index = unique_labels.tolist().index(new_label)

            # [:,<idx>] only selects a given column, here the new_label column
            # We'll look at this as a 'new' or not new problem

            # Calculate mAP for all other non-new labels
            non_new_indices = [i for i, label in enumerate(unique_labels) if label != new_label]

            # not_new_row_mask = np.where(ground_truth_matrix[:, new_index] != 1)[0]
            # assert ground_truth_matrix[:,non_new_indices].sum(axis=1).min() == 1, "There must be exactly 1 classification per row."

            for i, label in enumerate(unique_labels):
                assert np.any(
                    ground_truth_matrix[:, i] == 1
                ), f"No instance of label {i} '{label}' is not present in the ground truth matrix."

            # metrics = dict(threshold=threshold, percent_of_new_label=perc_of_new_label)
            results[threshold]["percent_of_new_label"].append(perc_of_new_label)
            for slug, Y_true, Y_pred in [
                ("all", ground_truth_matrix, prediction_matrix),
                ("new", ground_truth_matrix[:, new_index], prediction_matrix[:, new_index]),
                ("non_new", ground_truth_matrix[:, non_new_indices], prediction_matrix[:, non_new_indices]),
            ]:
                results[threshold][f"{slug}/f1"].append(
                    f1_score(Y_true, Y_pred, average="binary" if slug == "new" else "weighted")
                )
                results[threshold][f"{slug}/roc_auc_score"].append(roc_auc_score(Y_true, Y_pred, labels=unique_labels))

    fold_aggregated = {
        threshold: {name: np.mean(values) for name, values in metrics.items()} for threshold, metrics in results.items()
    }
    return fold_aggregated


def select_best_threshold(folds_aggregated: dict, optimize="new/f1"):
    thres, val = max([(k, v[optimize]) for k, v in folds_aggregated.items()], key=lambda x: x[1])
    return thres


def test_find_thresholds_is_deterministic():
    # generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/cc6tiy3f/workspace", "example-embeddings.pkl")
    df = read_embeddings_from_disk("example-embeddings.pkl")
    # print(df.head())
    # Convert embeddings to numpy arrays if they are not already
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    results = []
    seed = 777
    for run in range(3):
        unique_percentage = 0.2
        map_per_threshold = find_threshold(
            df=df,
            label_column="label_string",
            grid_start=7.0,
            grid_end=20.0,
            grid_num=20,
            unique_percentage=unique_percentage,
            seed=seed,
            normalize_label_distribution=True,
        )
        result = select_best_threshold(map_per_threshold)
        results.append(result)

    assert all(), "Results are not the same for different runs (not working deterministically)"

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from scipy.spatial.distance import cdist


def k_fold_threshold_search(
    df: pd.DataFrame,
    label_column: str,
    grid_start: float,
    grid_end: float,
    grid_num: float,
    unique_percentage=0.2,
    seed=42,
    normalize_label_distribution: bool = False,
):
    """You are responsible for normalizing the label distribution."""
    if normalize_label_distribution:
        df = norm_label_distribution(df, label_column, seed)

    # Thresholds for grid search
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    # For storing results
    results = defaultdict(lambda: defaultdict(list))
    kf = DistinctClassFold(n_buckets=5, shuffle=True, random_state=seed)

    # TODO(liamvdv): handle no sufficient match (singletons cannot have match)
    new_label = "new"
    for dataset_df, query_df in kf.split(
        df,
        label_column=label_column,
        label_mask=new_label,
        unique_percentage=unique_percentage,
        normalize_label_distribution=normalize_label_distribution,
    ):
        for threshold, metrics in threshold_grid_search(dataset_df, query_df, thresholds, k=5).items():
            for metric, value in metrics.items():
                results[threshold][metric].append(value)
    
    aggregated = {
        threshold: {name: np.mean(values) for name, values in metrics.items()} for threshold, metrics in results.items()
    }
    return aggregated


def threshold_grid_search(dataset, queryset, thresholds, k=5):
    # Step 2: Compute pairwise distances
    distances = cdist(queryset['embedding'].tolist(), dataset['embedding'].tolist(), metric='euclidean')
    
    # Step 3: Determine the closest k neighbors and compute confidence scores
    weights = np.array([1 / 2**i for i in range(1, k + 1)])  # Geometric weights
    k_indices = np.argsort(distances, axis=1)[:, :k]  # Indices of k closest neighbors
    closest_labels = np.take_along_axis(dataset['label'].values, k_indices, axis=0)
    
    # Confidence scores calculation
    confidence_scores = np.sum(weights * (closest_labels == queryset['label'].values[:, None]), axis=1)

    # Step 4: Threshold grid search and metrics computation
    results = {}

    for threshold in thresholds:
        predictions = confidence_scores >= threshold
        true_labels = queryset['label'] == 1  # Assuming '1' indicates relevance
        
        # Calculate precision, recall, f1, and average precision (for mAP)
        precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
        recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
        average_precision = average_precision_score(true_labels, predictions)

        results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'average_precision': average_precision
        }

    # Step 5: Output the results
    return results


if __name__ == "__main__":
    # generate_embeddings_from_run("https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/cc6tiy3f/workspace", "example-embeddings.pkl")
    df = read_embeddings_from_disk("example-embeddings.pkl")
    # print(df.head())
    # Convert embeddings to numpy arrays if they are not already
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))

    # for i in range(10):
    #     seed = np.random.randint(0, 1000)
    #     unique_percentage = 0.2
    #     map_per_threshold = find_threshold(
    #         df=df,
    #         label_column="label_string",
    #         grid_start=7.0,
    #         grid_end=20.0,
    #         grid_num=20,
    #         unique_percentage=unique_percentage,
    #         normalize_label_distribution=True,
    #         seed=None
    #     )
    #     plot_map_vs_thresholds(
    #         map_per_threshold, "Threshold vs mAP", f"mAP_vs_Threshold_random_seed_{i}.png"
    #     )

    #     best_threshold, best_overall_map, new_map, non_new_map, perc_of_new_label = select_best_threshold(map_per_threshold)
    #     print(
    #         f"Best threshold: {best_threshold} with overall mAP: {best_overall_map}, new mAP: {new_map}, non-new mAP: {non_new_map}, perc of new label in query set: {perc_of_new_label}"
    #     )
