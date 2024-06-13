from functools import partial
from itertools import islice
from typing import Any, Dict, List, Literal, Optional, Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import sklearn
import torch
import torchmetrics as tm
import wandb
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader as Dataloader
from torchmetrics.functional import pairwise_euclidean_distance
from torchvision.transforms import ToPILImage

import gorillatracker.type_helper as gtypes
from gorillatracker.data.nlet import NletDataModule
from gorillatracker.data.utils import flatten_batch, lazy_batch_size
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


def log_train_images_to_wandb(run: Runner, trainer: L.Trainer, train_dataloader: Dataloader[gtypes.Nlet], n_samples: int = 1) -> None:
    """
    Log nlet images from the train dataloader to wandb.
    Visual sanity check to see if the dataloader works as expected.
    """
    # get first n_samples triplets from the train dataloader
    samples = get_n_samples_from_dataloader(train_dataloader, n_samples=n_samples)  # type: ignore
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

    samples = get_n_samples_from_dataloader(train_dataloader, n_samples=1)  # type: ignore
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


def evaluate_embeddings(
    data: pd.DataFrame,
    embedding_name: str,
    metrics: Dict[str, Any],
    train_embeddings: Optional[torch.Tensor] = None,
    train_labels: Optional[gtypes.MergedLabels] = None,
    kfold_k: Optional[int] = None,
    dataloader_name: str = "Unkonwn",
) -> Dict[str, Any]:  # data is DataFrame with columns: label and embedding
    assert (train_embeddings is not None and train_labels is not None) or (
        train_embeddings is None and train_labels is None
    )

    # Transform any type to numeric type labels
    val_labels = torch.tensor(data["label"])
    train_labels = train_labels.clone().detach() if train_labels is not None else torch.tensor([])  # type: ignore
    val_labels = val_labels.type(torch.int64)
    train_labels = train_labels.type(torch.int64)

    val_train_labels = torch.cat([val_labels, train_labels], dim=0)

    nval = len(val_labels)
    val_labels, train_labels = val_train_labels[:nval], val_train_labels[nval:]
    val_embeddings = np.stack(data["embedding"].apply(np.array)).astype(np.float32)
    val_embeddings = torch.tensor(val_embeddings)

    assert len(val_embeddings) > 0, "No validation embeddings given."

    results = {
        metric_name: metric(val_embeddings, val_labels, train_embeddings=train_embeddings, train_labels=train_labels)
        for metric_name, metric in metrics.items()
    }

    kfold_str_prefix = f"fold-{kfold_k}/" if kfold_k is not None else ""
    for metric_name, result in results.items():
        if isinstance(result, dict):
            for key, value in result.items():
                wandb.log({f"{dataloader_name}/{kfold_str_prefix}{embedding_name}/{metric_name}/{key}": value})
        else:
            wandb.log({f"{dataloader_name}/{kfold_str_prefix}{embedding_name}/{metric_name}/": result}, commit=True)
    return results


def knn(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    k: int = 5,
    use_train_embeddings: bool = False,
    train_embeddings: Optional[torch.Tensor] = None,
    train_labels: Optional[torch.Tensor] = None,
    average: Literal["micro", "macro", "weighted", "none"] = "weighted",
) -> Dict[str, Any]:
    if use_train_embeddings and (train_embeddings is None or train_labels is None):
        raise ValueError("If use_train_embeddings is set to True, train_embeddings/train_labels must be provided.")

    # NOTE(rob2u): necessary for sanity checking dataloader and val only (problem when not range 0:n-1)
    le = LinearSequenceEncoder()
    val_labels_encoded = torch.tensor(le.encode_list(val_labels.tolist()))

    if use_train_embeddings:
        train_labels_encoded = torch.tensor(le.encode_list(train_labels.tolist()))  # type: ignore
        # print("Using train embeddings for knn")
        return knn_with_train(
            val_embeddings,
            val_labels_encoded,
            k=k,
            train_embeddings=train_embeddings,  # type: ignore
            train_labels=train_labels_encoded,
            average=average,
        )
    else:
        return knn_naive(val_embeddings, val_labels_encoded, k=k, average=average)


def knn_with_train(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"],
    k: int = 5,
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
    val_embeddings = val_embeddings.clone().detach()
    val_labels = torch.tensor(val_labels.tolist())
    train_embeddings = train_embeddings.clone().detach()
    train_labels = torch.tensor(train_labels.tolist())

    combined_embeddings = torch.cat([train_embeddings, val_embeddings], dim=0)
    combined_labels = torch.cat([train_labels, val_labels], dim=0)

    num_classes: int = int(torch.max(combined_labels).item() + 1)
    assert num_classes == len(np.unique(combined_labels))
    if num_classes < k:
        k = num_classes

    distance_matrix = pairwise_euclidean_distance(combined_embeddings)

    distance_matrix.fill_diagonal_(float("inf"))

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
    assert val_classification_matrix.shape == (len(val_embeddings), num_classes)

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


def knn_naive(
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    average: Literal["micro", "macro", "weighted", "none"],
    k: int = 5,
) -> Dict[str, Any]:
    num_classes = len(torch.unique(val_labels))
    if num_classes < k:
        print(f"Number of classes {num_classes} is smaller than k {k} -> setting k to {num_classes}")
        k = num_classes

    # convert embeddings and labels to tensors
    val_embeddings = val_embeddings.clone().detach()
    val_labels = torch.tensor(val_labels.tolist())

    distance_matrix = pairwise_euclidean_distance(val_embeddings)

    # Ensure distances on the diagonal are set to a large value so they are ignored
    distance_matrix.fill_diagonal_(float("inf"))

    # Find the indices of the closest embeddings for each embedding
    classification_matrix = torch.zeros((len(val_embeddings), k))

    _, closest_indices = torch.topk(distance_matrix, k, largest=False, sorted=True)
    assert closest_indices.shape == (len(val_embeddings), k)

    closest_labels = val_labels[closest_indices]
    assert closest_labels.shape == closest_indices.shape

    classification_matrix = torch.zeros((len(val_embeddings), num_classes))
    for i in range(num_classes):
        classification_matrix[:, i] = torch.sum(closest_labels == i, dim=1) / k
    assert classification_matrix.shape == (len(val_embeddings), num_classes)

    accuracy = tm.functional.accuracy(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert accuracy is not None
    accuracy_top5 = tm.functional.accuracy(
        classification_matrix,
        val_labels,
        task="multiclass",
        num_classes=num_classes,
        top_k=5 if num_classes >= 5 else num_classes,
    )
    assert accuracy_top5 is not None
    auroc = tm.functional.auroc(classification_matrix, val_labels, task="multiclass", num_classes=num_classes)
    assert auroc is not None
    f1 = tm.functional.f1_score(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert f1 is not None
    precision = tm.functional.precision(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average=average
    )
    assert precision is not None

    return {
        "accuracy": accuracy.item(),
        "accuracy_top5": accuracy_top5.item(),
        "auroc": auroc.item(),
        "f1": f1.item(),
        "precision": precision.item(),
    }


def pca(
    embeddings_in: torch.Tensor, labels_in: torch.Tensor, **kwargs: Any
) -> wandb.Image:  # generate a 2D plot of the embeddings
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
    embeddings_in: torch.Tensor, labels_in: torch.Tensor, with_pca: bool = False, count: int = 1000, **kwargs: Any
) -> Optional[wandb.Image]:  # generate a 2D plot of the embeddings
    num_classes = len(torch.unique(labels_in))
    embeddings = embeddings_in.numpy()
    labels = labels_in.numpy()

    indices = np.random.choice(len(embeddings), min(count, len(labels)), replace=False)
    embeddings = embeddings[indices]
    labels = labels[indices]
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


if __name__ == "__main__":
    # Test the EmbeddingAnalyzer and Accuracy metric
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
