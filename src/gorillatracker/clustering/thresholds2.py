from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from collections import defaultdict
from gorillatracker.clustering.folds import DistinctClassFold
from scipy.spatial import distance
from sklearn.preprocessing import label_binarize


# https://chat.openai.com/c/b843ee7f-3c92-4771-8ee5-ca76ac4e4e70
"""
# Work

Every function takes 
    (dataset, queryset, thresholds) as input and returns a dictionary of metric results for each threshold. We return the values all/metric_name, new/metric_name, non_new/metric_name.
The dataset and queryset are both pandas DataFrame's with the int column "label" and the numpy array column "embedding".
The new label is "new" and is used to indicate that the query point does not have a match in the dataset.
The thresholds are a list of floats that are used to determine the confidence score of the prediction.
Some functions are more complex and require additional parameters, initialize them with default values.

1. knn1
    - f1 score
2. knn1_centroid
    - f1 score
3. knn1_centroid_iqr (interquartile range)
    - f1 score
4. knn1_centroid_max_distance
    enforces that the points are within a certain distance of the centroid. Centroid does not adjust after pruning.
    - f1 score
5. knn5
    - choose the most common label, if there is a tie, choose any
    - f1 score
6. knnk_weighted_by_geometric_sequence(k)
    - mAP score
7. knnk_weighted_by_distance(k)
    - mAP score


Every function can be wrapped with with_centroid_queryset.
    In the real world we can use multiple images of the same tracking to obtain multiple embeddings of the same individual.
    This allows us to use a centroid over the embeddings of a individual.
    
    def with_centroid_queryset(dataset, queryset, function, *args, **kwargs):
        centroid_qs = queryset.groupby('label')['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0)).reset_index()
        centroid_qs.columns = ['label', 'embedding']
        return function(dataset, centroid_qs, *args, **kwargs)

"""

_le = dict()
_ld = []


def encode(labels: list[str]) -> list[int]:
    global _le
    ints = []
    for label in labels:
        if label not in _le:
            _le[label] = len(_le)
            _ld.append(label)
        ints.append(_le[label])
    return ints


def decode(ints: list[int]) -> list[str]:
    global _le
    return [_ld[i] for i in ints]


new_label_str = "new"
new_label = encode([new_label_str])[0]


def knn1(dataset, queryset, thresholds):
    """
    Find the nearest neighbor and assign that label or 'new' based on threshold.
    """
    # Fit the NearestNeighbors model using dataset embeddings
    nbrs = NearestNeighbors(n_neighbors=1).fit(np.vstack(dataset["embedding"]))
    # Find the nearest neighbors for queryset embeddings
    distances, indices = nbrs.kneighbors(np.vstack(queryset["embedding"]))

    results = {}
    for t in thresholds:
        # Decide label based on threshold: if distance <= t, use dataset label, otherwise 'new'
        predictions = dataset.iloc[indices.flatten()]["label"].where(distances.flatten() <= t, new_label).values

        # Compute F1 score for all labels combined
        f1_all = f1_score(queryset["label"], predictions, average="weighted")

        # Binary classification: new vs non_new
        binary_true = (queryset["label"] != new_label).astype(int)  # 1 for existing labels, 0 for 'new'
        binary_pred = (predictions != new_label).astype(int)  # 1 for existing labels, 0 for 'new'
        f1_non_new = f1_score(binary_true, binary_pred, pos_label=0)  # F1 for 'new' (pos_label=0)
        f1_existing = f1_score(binary_true, binary_pred, pos_label=1)  # F1 for existing/non-new labels

        results[t] = {
            "all/f1": f1_all,
            "new/f1": f1_non_new,  # reusing calculation for binary class 'new'
            "non_new/f1": f1_existing,
        }
    return results


def knn1_centroid(dataset, queryset, thresholds):
    """
    Choose closest class centroid and assign that label or 'new' based on threshold.
    """
    centroid_dataset = transform_to_centroid(dataset)
    return knn1(centroid_dataset, queryset, thresholds)


def transform_to_centroid(dataset):
    df = dataset.groupby("label")["embedding"].apply(calculate_centroid).reset_index()
    df.columns = ["label", "embedding"]
    return df


def filter_outliers_iqr(dataset):
    filtered_datasets = []
    for class_name, group in dataset.groupby("label"):
        centroid = group["embedding"].mean(axis=0)
        distances = group["embedding"].apply(lambda x: distance.euclidean(x, centroid))
        Q1 = np.percentile(distances, 25, axis=0)
        Q3 = np.percentile(distances, 75, axis=0)
        IQR = Q3 - Q1
        in_range = (distances >= Q1 - 1.5 * IQR) & (distances <= Q3 + 1.5 * IQR)
        filtered_embeddings = group[in_range]

        if filtered_embeddings.empty:
            # TODO(liamvdv): ?
            filtered_embeddings = group  # fallback if all are outliers

        filtered_datasets.append(filtered_embeddings)

    return pd.concat(filtered_datasets)


def knn1_centroid_iqr(dataset, queryset, thresholds):
    """
    Do knn1_centroid after filtering out outliers using IQR, producing a more robust centroid.
    """
    dataset_filtered = filter_outliers_iqr(dataset)
    return knn1_centroid(dataset_filtered, queryset, thresholds)


def downsample_class(df, label_column: str, seed: int = 42):
    """
    Randomly downsample each class in the dataset to have at most 10 instances.
    """
    # Group the DataFrame by the label column and apply the downsampling
    downsampled = df.groupby(label_column).apply(lambda x: x.sample(min(len(x), 10), random_state=seed))

    # Reset the index because groupby + apply creates a multi-index
    downsampled.reset_index(drop=True, inplace=True)

    return downsampled


def knn5(dataset, queryset, thresholds):
    """
    Look at the first 5 neighbors and vote (the most common label, random tie) or 'new' based on threshold.
    We ignore neighbors that are further away than the threshold.
    """
    nbrs = NearestNeighbors(n_neighbors=5).fit(np.vstack(dataset["embedding"]))
    distances, indices = nbrs.kneighbors(np.vstack(queryset["embedding"]))

    results = {}
    for t in thresholds:
        predictions = []
        for idx, dist in zip(indices, distances):
            valid = dataset.iloc[idx][dist <= t]
            if not valid.empty:
                prediction = valid["label"].mode()[0]  # Most common or arbitrary tie
            else:
                prediction = new_label
            predictions.append(prediction)
        f1_all = f1_score(queryset["label"], predictions, average="weighted")

        # Binary classification: new vs non_new
        binary_true = (queryset["label"] != new_label).astype(int)  # 1 for existing labels, 0 for 'new'
        binary_pred = (predictions != new_label).astype(int)  # 1 for existing labels, 0 for 'new'
        f1_non_new = f1_score(binary_true, binary_pred, pos_label=0)  # F1 for 'new' (pos_label=0)
        f1_existing = f1_score(binary_true, binary_pred, pos_label=1)  # F1 for existing/non-new labels

        results[t] = {
            "all/f1": f1_all,
            "new/f1": f1_non_new,  # reusing calculation for binary class 'new'
            "non_new/f1": f1_existing,
        }
    return results


def knnk_weighted_by_distance(dataset, queryset, thresholds, k=5):
    """
    Look at the first k neighbors and weight them by a in proximity order by
    geometric sequence. The sum of weights are then the confidence of the prediction.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(np.vstack(dataset["embedding"]))
    distances, indices = nbrs.kneighbors(np.vstack(queryset["embedding"]))

    results = {}
    for t in thresholds:
        predictions = []
        for idx, dist in zip(indices, distances):
            # idx=[int]*k, dist=[float]*k
            closest_dist = dist[0]
            if closest_dist > t:
                predictions.append(new_label)
            else:
                # [weight]*k
                weights = 1 / (1 + dist)  # Inverse distance weighting
                # [label]*k
                labels = dataset.iloc[idx]["label"]  # translate indices to labels

                # Create a DataFrame from labels and weights, then group by labels
                df = pd.DataFrame({"label": labels, "weight": weights})

                # Sum the weights for each label and find the label with the highest sum
                weighted_votes = df.groupby("label")["weight"].sum()

                predictions.append(weighted_votes)
        print(queryset["label"], predictions)
        # expand the predictions, other labels get predication value 0

        results[t] = compute_map_scores(
            queryset["label"], predictions, unique_labels=dataset["label"].values + [new_label]
        )
    return results


def geometric_sequence(k: int):
    # alt: 0.5 ** np.arange(1, k + 1)
    return [1 / 2**i for i in range(1, k + 1)]


def knnk_weighted_by_geometric_sequence(dataset, queryset, thresholds, k=5):
    """
    Look at the first k neighbors and weight them by the inverse of the distance.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(np.vstack(dataset["embedding"]))
    distances, indices = nbrs.kneighbors(np.vstack(queryset["embedding"]))

    geometric_sequence = 0.5 ** np.arange(1, k + 1)  # [0.5, 0.25, 0.125, ...]
    results = {}
    for t in thresholds:
        predictions = []
        for idx, dist in zip(indices, distances):
            if dist[0] > t:
                predictions.append(new_label)
            else:
                # {0,1} * k
                labels = np.ndarray(dataset.iloc[idx]["label"].values) == new_label
                raise NotImplementedError("This is not correct")
                # confidence = np.sum(geometric_sequence * labels)
                # compute the confidence of the prediction
                # [0.5, 0.25, 0.125, ...] * [0, 1, 0, 1, 1] = [0, 0.25, 0, 0.125, 0.125]
                confidence = np.sum(geometric_sequence * labels)

        results[t] = compute_map_scores(queryset["label"], predictions)
    return results


def calculate_centroid(embeddings):
    return np.mean(np.vstack(embeddings), axis=0)


def with_centroid_queryset(dataset, queryset, function, *args, **kwargs):
    centroid_qs = queryset.groupby("label")["embedding"].apply(calculate_centroid).reset_index()
    centroid_qs.columns = ["label", "embedding"]
    return function(dataset, centroid_qs, *args, **kwargs)


def compute_f1_scores(true_labels, pred_labels):
    return {
        "all/f1": f1_score(true_labels, pred_labels, average="weighted"),
        "new/f1": f1_score(true_labels, pred_labels, pos_label="new", average="binary"),
        "non_new/f1": f1_score(true_labels, pred_labels, pos_label=1, average="binary"),
    }


def compute_map_scores(true_labels, pred_labels, unique_labels):
    """
    Only call with confidence scores.
    """
    print(unique_labels)
    # Assuming a utility function for calculating mAP exists or using an appropriate proxy
    true_matrix = label_binarize(true_labels, classes=unique_labels)
    pred_matrix = label_binarize(pred_labels, classes=unique_labels)

    # Calculate mAP for the 'new' label
    assert new_label in unique_labels
    new_index = unique_labels.index(new_label)
    # Calculate mAP for all other 'non-new' labels
    non_new_indices = [i for i, label in enumerate(unique_labels) if label != new_label]

    # not_new_row_mask = np.where(ground_truth_matrix[:, new_index] != 1)[0]
    # assert ground_truth_matrix[:,non_new_indices].sum(axis=1).min() == 1, "There must be exactly 1 classification per row."

    for i, label in enumerate(unique_labels):
        assert np.any(
            true_matrix[:, i] == 1
        ), f"No instance of label {i} '{label}' is not present in the ground truth matrix."

    results = {}
    for slug, Y_true, Y_pred in [
        ("all", true_matrix, pred_matrix),
        ("new", true_matrix[:, new_index], pred_matrix[:, new_index]),
        ("non_new", true_matrix[:, non_new_indices], pred_matrix[:, non_new_indices]),
    ]:
        results[f"{slug}/mAP"].append(
            average_precision_score(Y_true, Y_pred, average="binary" if slug == "new" else "weighted")
        )
        results[f"{slug}/roc_auc_score"].append(roc_auc_score(Y_true, Y_pred, labels=unique_labels))
    return results
    # binary_pred =  pred_labels[]

    # binary_true = (true_labels != new_label).astype(int)  # 1 for existing labels, 0 for 'new'
    # binary_pred = (pred_labels != new_label).astype(int)  # 1 for existing labels, 0 for 'new'

    # # Compute mAP for all labels combined
    # all_map = average_precision_score(true_matrix, pred_matrix, average="weighted")

    # # Compute mAP for 'new' vs non_new
    # new_map = average_precision_score(binary_true, binary_pred, pos_label=0)

    # # Compute mAP for existing/non-new labels
    # non_new_map = average_precision_score(binary_true, binary_pred, pos_label=1)
    # return {
    #     "all/mAP": average_precision_score(true_labels, pred_labels, average="weighted"),
    #     "new/mAP": average_precision_score(true_labels, pred_labels),
    #     "non_new/mAP": average_precision_score(true_labels, pred_labels),
    # }


def k_fold_threshold_search(
    df: pd.DataFrame,
    label_column: str,
    grid_start: float,
    grid_end: float,
    grid_num: float,
    unique_percentage=0.2,
    seed=42,
    function=knn1,
    *function_args,
    **function_kwargs,
):
    """You are responsible for normalizing the label distribution."""
    global le
    # Thresholds for grid search
    thresholds = np.linspace(grid_start, grid_end, grid_num)

    # For storing results
    results = defaultdict(lambda: defaultdict(list))

    assert pd.api.types.is_string_dtype(df[label_column]), "df[label_column] must be string type"
    label_int_column = "label"
    assert label_column != label_int_column, "label_column cannot be 'label', used for integer encoding"

    unique_labels_str = df[label_column].unique().tolist()
    assert (
        new_label_str not in unique_labels_str
    ), f"'{new_label_str}' already exists in column '{label_column}' '{unique_labels_str}'"

    df[label_int_column] = encode(df[label_column])
    unique_labels_int = df[label_int_column].unique().tolist()
    assert (
        new_label not in df[label_int_column].unique()
    ), f"'{new_label}' already exists in column '{label_int_column}' '{unique_labels_int}'"

    kf = DistinctClassFold(n_buckets=5, shuffle=True, random_state=seed)
    new_perc_folds = []
    # TODO(liamvdv): handle no sufficient match (singletons cannot have match)
    for dataset_df, query_df in kf.split(
        df,
        label_column=label_int_column,
        label_mask=new_label,
        unique_percentage=unique_percentage,
    ):
        for threshold, metrics in function(dataset_df, query_df, thresholds, *function_args, **function_kwargs).items():
            for metric, value in metrics.items():
                results[threshold][metric].append(value)
        new_perc_folds.append(query_df["label"].value_counts(normalize=True)[new_label])
    aggregated = {
        threshold: {name: np.mean(values) for name, values in metrics.items()} for threshold, metrics in results.items()
    }
    return aggregated, new_perc_folds
