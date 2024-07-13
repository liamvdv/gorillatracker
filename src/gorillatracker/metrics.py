from collections import Counter, defaultdict
from itertools import islice
from typing import Any, Dict, List, Literal, Optional

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import PIL
import seaborn as sns
import sklearn
import torch
import torchmetrics as tm
import wandb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from scipy.stats import chi2
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from torch.utils.data import DataLoader as Dataloader
from torchmetrics.functional import pairwise_euclidean_distance
from torchvision.transforms import ToPILImage

import gorillatracker.type_helper as gtypes
from gorillatracker.data.contrastive_sampler import get_individual, get_individual_video_id
from gorillatracker.data.nlet import NletDataModule
from gorillatracker.utils.labelencoder import LinearSequenceEncoder

# TODO: What is the wandb run type?
Runner = Any


def load_embeddings_from_wandb(embedding_name: str, run: Runner) -> pd.DataFrame:
    """Load embeddings from wandb Artifact."""
    # Data is a pandas Dataframe with columns: label, embedding_0, embedding_1, ... loaded from wandb from the
    artifact = run.use_artifact(embedding_name, type="embeddings")
    data_table = artifact.get("embeddings_table_epoch_10")  # TODO
    data = pd.DataFrame(data=data_table.data, columns=data_table.columns)
    return data


def tensor_to_image(tensor: torch.Tensor) -> PIL.Image.Image:
    return ToPILImage()(tensor.cpu()).convert("RGB")


def get_n_samples_from_dataloader(dataloader: Dataloader[gtypes.Nlet], n_samples: int = 1) -> list[gtypes.Nlet]:
    samples: list[gtypes.Nlet] = []
    for batch in dataloader:
        ids, images, labels = batch
        row_batch = zip(zip(*ids), zip(*images), zip(*labels))
        take_max_n = n_samples - len(samples)
        samples.extend(list(islice(row_batch, take_max_n)))
        if len(samples) == n_samples:
            break
    return samples


def log_train_images_to_wandb(
    run: Runner, trainer: L.Trainer, train_dataloader: Dataloader[gtypes.Nlet], n_samples: int = 1
) -> None:
    """
    Log nlet images from the train dataloader to wandb.
    Visual sanity check to see if the dataloader works as expected.
    """
    # get first n_samples triplets from the train dataloader
    samples = get_n_samples_from_dataloader(train_dataloader, n_samples=n_samples)
    for i, sample in enumerate(samples):
        # a row (nlet) can either be (ap, p, n) OR (ap, p, n, an)
        row_meaning = ("positive_anchor", "positive", "negative", "negative_anchor")
        _, row_images, row_labels = sample
        img_label_meaning = zip(row_images, row_labels, row_meaning)
        artifacts = [
            wandb.Image(tensor_to_image(img), caption=f"{meaning} label={label}")
            for img, label, meaning in img_label_meaning
        ]
        run.log({f"epoch_{trainer.current_epoch}_nlet_{1+i}": artifacts})


@torch.enable_grad()  # type: ignore
def log_grad_cam_images_to_wandb(run: Runner, trainer: L.Trainer, train_dataloader: Dataloader[gtypes.Nlet]) -> None:
    # NOTE(liamvdv): inverse grad cam support to model since we might not be using
    #                a model which grad cam does not support.
    # NOTE(liamvdv): Transform models may have different interpretations.
    assert trainer.model is not None, "Must only call log_grad_cam_images... after model was initialized."

    if not hasattr(trainer.model, "get_grad_cam_layer"):
        return
    target_layer = trainer.model.get_grad_cam_layer()
    get_reshape_transform = getattr(trainer.model, "get_grad_cam_reshape_transform", lambda: None)
    cam = GradCAM(model=trainer.model, target_layers=[target_layer], reshape_transform=get_reshape_transform())

    samples = get_n_samples_from_dataloader(train_dataloader, n_samples=1)
    wandb_images: List[wandb.Image] = []
    for sample in samples:
        # a row (nlet) can either be (ap, p, n) OR (ap, p, n, an)
        _, row_images, row_labels = sample
        anchor, *rest = row_images
        grayscale_cam = cam(input_tensor=anchor.unsqueeze(0), targets=None)

        # Overlay heatmap on original image
        heatmap = grayscale_cam[0, :]
        image = np.array(ToPILImage()(anchor)).astype(np.float32) / 255.0  # NOTE(liamvdv): needs be normalized
        image_with_heatmap = show_cam_on_image(image, heatmap, use_rgb=True)
        wandb_images.append(wandb.Image(image_with_heatmap, caption=f"label={row_labels[0]}"))
    run.log({"Grad-CAM": wandb_images})


def get_partition_from_dataframe(
    data: pd.DataFrame, partition: Literal["val", "train", "test"] = "val"
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, list[gtypes.Id], torch.Tensor]:
    partition_df = data.where(data["partition"] == partition).dropna()
    partition_labels = torch.tensor(partition_df["label"].tolist()).long()
    partition_embeddings = np.stack(partition_df["embedding"].apply(np.array)).astype(np.float32)
    partition_embeddings = torch.tensor(partition_embeddings)
    partition_ids = partition_df["id"].tolist()
    partition_encoded_labels = torch.tensor(partition_df["encoded_label"].tolist()).long()

    return partition_df, partition_labels, partition_embeddings, partition_ids, partition_encoded_labels


def evaluate_embeddings(
    data: pd.DataFrame,  # columns: label, embedding, id, partition
    embedding_name: str,
    metrics: Dict[str, Any],
    kfold_k: Optional[int] = None,
    dataloader_name: str = "Unknown",
) -> Dict[str, Any]:

    assert (
        all([column in data.columns for column in ["label", "embedding", "id", "partition", "dataset"]])
        and len(data.columns) == 5
    ), "Dataframe must have columns: label, embedding, id, partition. More are not allowed!"

    # NOTE(rob2u): necessary for sanity checking dataloader and val only (problem when not range 0:n-1)
    le = LinearSequenceEncoder()
    data["label"] = data["label"].astype(int)
    data["encoded_label"] = le.encode_list(data["label"].tolist())

    results = {metric_name: metric(data) for metric_name, metric in metrics.items()}

    kfold_str_prefix = f"fold-{kfold_k}/" if kfold_k is not None else ""
    results_parsed = {}
    for metric_name, result in results.items():
        if isinstance(result, dict):
            for key, value in result.items():
                results_parsed.update(
                    {f"{dataloader_name}/{kfold_str_prefix}{embedding_name}/{metric_name}/{key}": value}
                )
        else:
            results_parsed.update({f"{dataloader_name}/{kfold_str_prefix}{embedding_name}/{metric_name}/": result})

    return results_parsed


def _get_crossvideo_masks(
    labels: torch.Tensor, ids: list[gtypes.Id], min_samples: int = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    distance_mask = torch.zeros((len(labels), len(labels)))
    classification_mask = torch.zeros(len(labels))

    vids_per_id: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(lambda: 0)
    )  # NOTE: individual_id -> (individual_video_id -> num_images))
    idx_per_vid: defaultdict[str, list[int]] = defaultdict(list)
    for i, id in enumerate(ids):
        individual_video_id = get_individual_video_id(id)
        vids_per_id[get_individual(id)][individual_video_id] += 1
        idx_per_vid[individual_video_id].append(i)

    for i, id in enumerate(ids):
        individual_video_id = get_individual_video_id(id)

        distance_mask_ = [True] * len(ids)
        for idx in idx_per_vid[individual_video_id]:
            distance_mask_[idx] = False
        distance_mask[i] = torch.tensor(distance_mask_)  # 1 if not same video, 0 if same video

        if (
            sum(vids_per_id[get_individual(id)].values()) - vids_per_id[get_individual(id)][individual_video_id]
            >= min_samples
        ):
            classification_mask[i] = True

    return distance_mask.to(torch.bool), classification_mask.to(torch.bool)


def knn(
    data: pd.DataFrame,
    average: Literal["micro", "macro", "weighted", "none"] = "weighted",
    k: int = 5,
    use_train_embeddings: bool = False,
    use_crossvideo_positives: bool = False,
) -> Dict[str, Any]:
    """
    Algorithmic Description:
    1. Calculate the distance matrix between all embeddings (len(embeddings) x len(embeddings))
       Set the diagonal of the distance matrix to a large value so that the distance to itself is ignored
    2. For each embedding find the k closest [smallest distances] embeddings (len(embeddings) x k)
       First find the indexes, the map to the labels (numbers).
    3. Create classification matrix where every embedding has a row with the probability for each class in it's top k surroundings (len(embeddings) x num_classes)
    4. Select only the validation part of the classification matrix (len(val_embeddings) x num_classes)
    5. Calculate the accuracy, accuracy_top5, auroc and f1 score: Either choose highest probability as class as matched class or check if any of the top 5 classes matches.
    """

    # convert embeddings and labels to tensors
    _, _, val_embeddings, val_ids, val_labels = get_partition_from_dataframe(data, partition="val")
    train_labels, train_embeddings = torch.Tensor([]), torch.Tensor([])
    if use_train_embeddings:
        _, _, train_embeddings, _, train_labels = get_partition_from_dataframe(data, partition="train")

    combined_embeddings = torch.cat([train_embeddings, val_embeddings], dim=0)
    combined_labels = torch.cat([train_labels, val_labels], dim=0)

    num_classes: int = int(torch.max(combined_labels).item() + 1)
    assert num_classes == len(np.unique(combined_labels))
    if num_classes < k:
        k = num_classes

    distance_matrix = pairwise_euclidean_distance(combined_embeddings)
    distance_matrix.fill_diagonal_(float("inf"))

    distance_mask: torch.Tensor
    classification_mask: torch.Tensor
    if use_crossvideo_positives:
        distance_mask, classification_mask = _get_crossvideo_masks(val_labels, val_ids)
        if use_train_embeddings:  # add train embeddings to the distance mask (shapes would not match otherwise)
            train_distance_mask = torch.ones((len(train_labels), len(train_labels) + len(val_labels)))
            distance_mask = torch.cat([torch.ones((len(val_labels), len(train_labels))), distance_mask], dim=1)
            distance_mask = torch.cat([train_distance_mask, distance_mask], dim=0)
            distance_mask = distance_mask.to(torch.bool)
        distance_matrix[~distance_mask] = float("inf")

    _, closest_indices = torch.topk(
        distance_matrix,
        k,
        largest=False,
        sorted=True,
    )
    assert closest_indices.shape == (len(combined_embeddings), k)

    closest_labels = combined_labels[closest_indices]
    assert closest_labels.shape == closest_indices.shape

    classification_matrix = torch.zeros((len(combined_embeddings), num_classes))
    for i in range(num_classes):
        classification_matrix[:, i] = torch.sum(closest_labels == i, dim=1) / k
    assert classification_matrix.shape == (len(combined_embeddings), num_classes)

    # Select only the validation part of the classification matrix
    val_classification_matrix = classification_matrix[-len(val_embeddings) :]

    if use_crossvideo_positives:  # remove all with only one individual_video_id
        val_classification_matrix = val_classification_matrix[classification_mask]
        val_labels = val_labels[classification_mask]

    # assert val_classification_matrix.shape == (len(val_embeddings), num_classes)
    accuracy = tm.functional.accuracy(
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert accuracy is not None
    accuracy_top5 = tm.functional.accuracy(
        val_classification_matrix,
        val_labels,
        task="multiclass",
        num_classes=num_classes,
        top_k=5 if num_classes >= 5 else num_classes,
    )
    assert accuracy_top5 is not None
    auroc = tm.functional.auroc(val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes)
    assert auroc is not None
    f1 = tm.functional.f1_score(
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert f1 is not None
    precision = tm.functional.precision(
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert precision is not None

    return {
        "accuracy": accuracy.item(),
        "accuracy_top5": accuracy_top5.item(),
        "auroc": auroc.item(),
        "f1": f1.item(),
        "precision": precision.item(),
    }


def knn_ssl(
    data: pd.DataFrame,
    dm: NletDataModule,
    average: Literal["micro", "macro", "weighted", "none"] = "weighted",
    k: int = 5,
) -> Dict[str, Any]:
    # TODO: add true label
    _, labels, embeddings, _, _ = get_partition_from_dataframe(data, partition="val")

    negatives = {}
    true_labels = []
    pred_labels = []
    pred_labels_top5 = []

    en = LinearSequenceEncoder()
    labels = torch.tensor(en.encode_list(labels.tolist()))
    current_val_index = 0
    for label in labels.unique():  # type: ignore
        decoded_label = en.decode(label.item())
        image = dm.val[current_val_index].contrastive_sampler.find_any_image(decoded_label)
        negative_labels = dm.val[current_val_index].contrastive_sampler.negative_classes(image)
        negatives[label.item()] = en.encode_list(negative_labels)

    for label in labels.unique():  # type: ignore
        subset_labels = negatives[label.item()] + [label.item()]
        if len(subset_labels) < 2:
            continue
        subset_mask = torch.isin(labels, torch.tensor(subset_labels))
        subset_embeddings = embeddings[subset_mask]
        subset_label_values = labels[subset_mask]
        knn = NearestNeighbors(n_neighbors=max(5, k) + 1, algorithm="auto").fit(subset_embeddings.numpy())
        current_label_mask = subset_label_values == label.item()
        current_label_embeddings = subset_embeddings[current_label_mask]
        _, indices = knn.kneighbors(current_label_embeddings.numpy())
        indices = indices[:, 1:]
        for idx_list in indices:
            neighbor_labels = subset_label_values[idx_list]
            most_common = torch.mode(neighbor_labels[:k]).values.item()
            true_labels.append(label.item())
            pred_labels.append(most_common)
            pred_labels_top5.append(neighbor_labels[:5].numpy())

    true_labels_tensor = torch.tensor(true_labels)
    pred_labels_tensor = torch.tensor(pred_labels)

    pred_labels_top5_nparray = np.array(pred_labels_top5)
    pred_labels_top5_tensor = torch.tensor(pred_labels_top5_nparray)
    top5_correct = []
    for i, true_label in enumerate(true_labels_tensor):
        if true_label in pred_labels_top5_tensor[i]:
            top5_correct.append(1)
        else:
            top5_correct.append(0)
    top5_accuracy = sum(top5_correct) / len(top5_correct)

    accuracy = accuracy_score(true_labels_tensor, pred_labels_tensor)
    f1 = f1_score(true_labels_tensor, pred_labels_tensor, average=average)
    precision = precision_score(true_labels_tensor, pred_labels_tensor, average=average, zero_division=0)

    return {"accuracy": accuracy, "accuracy_top5": top5_accuracy, "f1": f1, "precision": precision}


def pca(data: pd.DataFrame, **kwargs: Any) -> wandb.Image:  # generate a 2D plot of the embeddings
    _, _, embeddings_in, _, labels_in = get_partition_from_dataframe(data, partition="val")

    num_classes = len(torch.unique(labels_in))
    embeddings = embeddings_in.numpy()
    labels = labels_in.numpy()

    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(embeddings)
    embeddings = pca.transform(embeddings)
    # plot embeddings

    plt.figure()
    plot = sns.scatterplot(
        x=embeddings[:, 0], y=embeddings[:, 1], palette=sns.color_palette("hls", num_classes), hue=labels
    )
    # ignore outliers when calculating the axes limits
    x_min, x_max = np.percentile(embeddings[:, 0], [0.1, 99.9])
    y_min, y_max = np.percentile(embeddings[:, 1], [0.1, 99.9])
    plot.set_xlim(x_min, x_max)
    plot.set_ylim(y_min, y_max)
    plot.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    # plot.figure.savefig("pca.png")
    plot = wandb.Image(plot.figure)
    # print("pca done")
    plt.close("all")
    return plot


def tsne(
    data: pd.DataFrame, with_pca: bool = False, count: int = 2000, **kwargs: Any
) -> Optional[wandb.Image]:  # generate a 2D plot of the embeddings
    _, _, embeddings_in, _, labels_in = get_partition_from_dataframe(data, partition="val")

    embeddings = embeddings_in.numpy()
    labels = labels_in.numpy()

    indices = np.random.choice(len(embeddings), min(count, len(labels)), replace=False)
    embeddings = embeddings[indices]
    labels = labels[indices]
    num_classes = len(np.unique(labels))
    if len(labels) < 50:
        return None
    if with_pca:
        embeddings = sklearn.decomposition.PCA(n_components=50).fit_transform(embeddings)

    # tsne = TSNE(n_components=2, method="exact")
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure()
    plot = sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        palette=sns.color_palette("hls", num_classes),
        hue=labels,
    )
    # ignore outliers when calculating the axes limits
    x_min, x_max = np.percentile(embeddings[:, 0], [1, 99])
    y_min, y_max = np.percentile(embeddings[:, 1], [1, 99])
    plot.set_xlim(x_min, x_max)
    plot.set_ylim(y_min, y_max)
    # place the legend outside of the plot but readable
    plot.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.0)
    # plot.figure.savefig("tnse.png")
    plot = wandb.Image(plot.figure)
    # print("tsne done")
    plt.close("all")
    return plot


def estimate_sigma(samples: npt.NDArray[np.float32], labels: npt.NDArray[np.int32]) -> float:
    sigma_squared_estimates = []

    for m in np.unique(labels):
        class_samples = samples[labels == m]
        if class_samples.shape[0] < 2:
            continue
        S_m = np.cov(class_samples, rowvar=False, bias=False)
        sigma_squared_m = np.mean(np.diag(S_m))
        sigma_squared_estimates.append(sigma_squared_m)

    sigma_squared = np.mean(sigma_squared_estimates)
    sigma = np.sqrt(sigma_squared)

    return sigma


def calculate_margin(data: pd.DataFrame, k: int = 5, d: int = 2) -> float:
    X = np.stack(data["embedding"].values)
    y = data["label"].values

    # NOTE(rob2u):
    # we know ||x - mu||^2 ~ sigma**2 * chi^2(d) -> when we want to find the margin
    # we want to use the euclidean distance, so we take the square root

    sigma = estimate_sigma(X, np.array(y, dtype=np.int32))
    margin_sq = sigma**2 * chi2.ppf(0.95, d)
    return np.sqrt(margin_sq)


def openset_clustering(data: pd.DataFrame, k: int = 1) -> dict[str, float]:
    val_data = data[data["partition"] == "val"].copy()  # NOTE(rob2u): explicitly copy to avoid warning

    # NOTE(rob2u): average the embeddings for a single video
    val_data["VIDEO_ID"] = val_data["id"].apply(get_individual_video_id)
    val_data = val_data.groupby(["label", "VIDEO_ID"]).agg({"embedding": "mean"}).reset_index()
    margin = (
        calculate_margin(val_data, k=k, d=val_data["embedding"][0].shape[-1]) * 2
    )  # NOTE: multiply by 2 to get the full margin

    known_data = val_data.sample(frac=0.5, random_state=0)
    unknown_data = val_data.drop(known_data.index)

    X_known = np.stack(known_data["embedding"].values)
    y_known = known_data["label"].values
    X_unknown = np.stack(unknown_data["embedding"].values)
    y_unknown = unknown_data["label"].values

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_known, y_known)

    current_labels = set(y_known)
    predicted_labels = []

    def assign_label(new_point: npt.ArrayLike) -> int:
        distances, indices = knn.kneighbors([new_point])
        neighbor_labels = y_known[indices[0]]
        label_counts = Counter(neighbor_labels)
        # filter out neighbors that are not within the margin
        allowed_labels = [label for label, distance in zip(neighbor_labels, distances[0]) if distance < margin]
        if allowed_labels:
            return max(allowed_labels, key=lambda x: label_counts[x])
        else:
            new_label = max(current_labels) + 1
            current_labels.add(new_label)
            return new_label

    for new_point in X_unknown:
        predicted_label = assign_label(new_point)
        predicted_labels.append(predicted_label)
        X_known = np.vstack([X_known, new_point])
        y_known = np.append(y_known, predicted_label)
        knn.fit(X_known, y_known)

    return {
        "ARI": adjusted_rand_score(y_unknown, predicted_labels),
    }


if __name__ == "__main__":
    run = wandb.init(entity="gorillas", project="MNIST-EfficientNetV2", name="test_embeddings2")
    data = load_embeddings_from_wandb("run_MNISTTest5-2023-11-11-15-17-17_epoch_10:v0", run)
    results = evaluate_embeddings(
        data=data,
        embedding_name="run_MNISTTest5-2023-11-11-15-17-17_epoch_10:v0",
        metrics={
            "pca": pca,
            "tsne": tsne,
            "knn": knn,
        },
    )
    print(results)
