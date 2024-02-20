from functools import partial
from itertools import islice
from typing import Any, Dict, List, Optional, Tuple, Union

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
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from torchmetrics.functional import pairwise_euclidean_distance
from torchvision.transforms import ToPILImage

import gorillatracker.type_helper as gtypes

# TODO: What is the wandb run type?
Runner = Any


def log_as_wandb_table(embeddings_table: pd.DataFrame, run: Runner) -> None:
    tmp = embeddings_table.apply(
        lambda row: pd.concat([pd.Series([row["label"]]), pd.Series(row["embedding"])]), axis=1
    )
    tmp.columns = ["label"] + [f"embedding_{i}" for i in range(len(embeddings_table["embedding"].iloc[0]))]
    run.log({"embeddings": wandb.Table(dataframe=tmp)})  # type: ignore


class LogEmbeddingsToWandbCallback(L.Callback):
    """
    A pytorch lightning callback that saves embeddings to wandb and logs them.

    Args:
        every_n_val_epochs: Save embeddings every n epochs as a wandb artifact (of validation set).
        log_share: Log embeddings to wandb every n epochs.
    """

    def __init__(
        self, every_n_val_epochs: int, knn_with_train: bool, wandb_run: Runner, dm: L.LightningDataModule
    ) -> None:
        super().__init__()
        self.embedding_artifacts: List[str] = []
        self.every_n_val_epochs = every_n_val_epochs
        self.knn_with_train = knn_with_train
        self.run = wandb_run
        dm.setup("fit")
        self.train_dataloader = dm.train_dataloader()

    def _get_train_embeddings_for_knn(self, trainer: L.Trainer) -> Tuple[torch.Tensor, gtypes.MergedLabels]:
        assert trainer.model is not None, "Model must be initalized before validation phase."
        train_embedding_batches = []
        train_labels = []
        for batch in self.train_dataloader:
            images, labels = batch
            anchor_images = images[0].to(trainer.model.device)
            embeddings = trainer.model(anchor_images)
            train_embedding_batches.append(embeddings)
            anchor_labels = labels[0]
            train_labels.extend(anchor_labels)
        train_embeddings = torch.cat(train_embedding_batches, dim=0)
        assert len(train_embeddings) == len(train_labels)
        return train_embeddings.cpu(), tuple(train_labels)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        embeddings_table = pl_module.embeddings_table
        current_step = trainer.global_step

        assert trainer.max_epochs is not None

        table = wandb.Table(columns=embeddings_table.columns.to_list(), data=embeddings_table.values)  # type: ignore
        artifact = wandb.Artifact(
            name="run_{0}_step_{1}".format(self.run.name, current_step),
            type="embeddings",
            metadata={"step": current_step},
            description="Embeddings from step {}".format(current_step),
        )
        artifact.add(table, "embeddings_table_step_{}".format(current_step))
        self.run.log_artifact(artifact)
        self.embedding_artifacts.append(artifact.name)

        train_embeddings, train_labels = (
            self._get_train_embeddings_for_knn(trainer) if self.knn_with_train else (None, None)
        )

        metrics = {
            "knn5": partial(knn, k=5),
            "knn": partial(knn, k=1),
            "pca": pca,
            "tsne": tsne,
            "fc_layer": fc_layer,
        }
        metrics |= (
            {
                "knn5-with-train": partial(knn, k=5, use_train_embeddings=True),
                "knn-with-train": partial(knn, k=1, use_train_embeddings=True),
            }
            if self.knn_with_train
            else {}
        )
        # log to wandb
        evaluate_embeddings(
            data=embeddings_table,
            embedding_name="val/embeddings",
            metrics=metrics,
            train_embeddings=train_embeddings,  # type: ignore
            train_labels=train_labels,
        )
        # clear the table where the embeddings are stored
        # pl_module.embeddings_table = pd.DataFrame(columns=pl_module.embeddings_table_columns)  # rese t embeddings table

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        log_train_images_to_wandb(self.run, trainer, n_samples=1)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        log_grad_cam_images_to_wandb(self.run, trainer)


# now add stuff to evaluate the embeddings / the model that created the embeddings
# 1. add a fully connected layer to the model that takes the embeddings as input and outputs the labels -> then train this model -> evaluate false positive, false negative, accuracy, ...)
# 2. use different kinds of clustering algorithms to cluster the embeddings -> evaluate (false positive, false negative, accuracy, ...)
# 3. use some kind of FLDA ({(m_1 - m_2)^2/(s_1^2 + s_2^2)} like metric to evaluate the quality of the embeddings
# 4. try kNN with different k values to evaluate the quality of the embeddings
# 5. enjoy
def load_embeddings_from_wandb(embedding_name: str, run: Runner) -> pd.DataFrame:
    """Load embeddings from wandb Artifact."""
    # Data is a pandas Dataframe with columns: label, embedding_0, embedding_1, ... loaded from wandb from the
    artifact = run.use_artifact(embedding_name, type="embeddings")
    data_table = artifact.get("embeddings_table_epoch_10")  # TODO
    data = pd.DataFrame(data=data_table.data, columns=data_table.columns)
    return data


def tensor_to_image(tensor: torch.Tensor) -> PIL.Image.Image:
    return ToPILImage()(tensor.cpu()).convert("RGB")


def get_n_samples_from_dataloader(
    dataloader: gtypes.BatchNletDataLoader, n_samples: int = 1
) -> List[Tuple[Tuple[torch.Tensor, ...], Tuple[Union[str, int], ...]]]:
    samples: List[Tuple[Tuple[torch.Tensor, ...], Tuple[Union[str, int], ...]]] = []
    for batch in dataloader:
        images, labels = batch
        row_batch = zip(zip(*images), zip(*labels))
        take_max_n = n_samples - len(samples)
        samples.extend(list(islice(row_batch, take_max_n)))
        if len(samples) == n_samples:
            break
    return samples


def log_train_images_to_wandb(run: Runner, trainer: L.Trainer, n_samples: int = 1) -> None:
    """
    Log nlet images from the train dataloader to wandb.
    Visual sanity check to see if the dataloader works as expected.
    """
    # get first n_samples triplets from the train dataloader
    samples = get_n_samples_from_dataloader(trainer.train_dataloader, n_samples=n_samples)  # type: ignore
    for i, sample in enumerate(samples):
        # a row (nlet) can either be (ap, p, n) OR (ap, p, n, an)
        row_meaning = ("positive_anchor", "positive", "negative", "negative_anchor")
        row_images, row_labels = sample
        img_label_meaning = zip(row_images, row_labels, row_meaning)
        artifacts = [
            wandb.Image(tensor_to_image(img), caption=f"{meaning} label={label}")
            for img, label, meaning in img_label_meaning
        ]
        run.log({f"epoch_{trainer.current_epoch}_nlet_{1+i}": artifacts})


def log_grad_cam_images_to_wandb(run: Runner, trainer: L.Trainer) -> None:
    # NOTE(liamvdv): inverse grad cam support to model since we might not be using
    #                a model which grad cam does not support.
    # NOTE(liamvdv): Transform models may have different interpretations.
    assert trainer.model is not None, "Must only call log_grad_cam_images... after model was initialized."
    if not hasattr(trainer.model, "get_grad_cam_layer"):
        return
    target_layer = trainer.model.get_grad_cam_layer()
    get_reshape_transform = getattr(trainer.model, "get_grad_cam_reshape_transform", lambda: None)
    cam = GradCAM(model=trainer.model, target_layers=[target_layer], reshape_transform=get_reshape_transform())

    samples = get_n_samples_from_dataloader(trainer.train_dataloader, n_samples=1)  # type: ignore
    wandb_images: List[wandb.Image] = []
    for sample in samples:
        # a row (nlet) can either be (ap, p, n) OR (ap, p, n, an)
        row_images, row_labels = sample
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
    train_embeddings: Optional[npt.NDArray[np.float_]] = None,
    train_labels: Optional[gtypes.MergedLabels] = None,
) -> Dict[str, Any]:  # data is DataFrame with columns: label and embedding
    assert (train_embeddings is not None and train_labels is not None) or (
        train_embeddings is None and train_labels is None
    )

    # Transform any type to numeric type labels
    le = LabelEncoder()
    _val_train_labels = np.concatenate([data["label"], train_labels]) if train_labels is not None else data["label"]
    val_train_labels = le.fit_transform(_val_train_labels)
    nval = len(data["label"])
    val_labels, train_labels = val_train_labels[:nval], val_train_labels[nval:]
    val_embeddings = np.stack(data["embedding"].apply(np.array)).astype(np.float32)

    assert len(val_embeddings) > 0, "No validation embeddings given."

    results = {
        metric_name: metric(val_embeddings, val_labels, train_embeddings=train_embeddings, train_labels=train_labels)
        for metric_name, metric in metrics.items()
    }

    for metric_name, result in results.items():
        if isinstance(result, dict):
            for key, value in result.items():
                wandb.log({f"{embedding_name}/{metric_name}/{key}": value})
        else:
            wandb.log({f"{embedding_name}/{metric_name}": result})

    return results


def fc_layer(
    embeddings: torch.Tensor,
    labels: gtypes.MergedLabels,
    batch_size: int = 64,
    epochs: int = 300,
    seed: int = 42,
    **kwargs: Any,
) -> Dict[str, float]:
    num_classes = (
        int(np.max(labels)) + 1
    )  # NOTE(liamvdv): not len() because train_labels also gets intermediate ids assigned.
    model = torch.nn.Sequential(
        torch.nn.Linear(embeddings.shape[1], 100),
        torch.nn.Sigmoid(),
        torch.nn.Linear(100, num_classes),
    )
    for param in model.parameters():
        param.requires_grad_(True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    # acitvate gradients
    with torch.set_grad_enabled(True):
        for epoch in range(epochs):
            loss_sum = 0.0
            embeddings_copy, labels_copy = sklearn.utils.shuffle(
                embeddings, labels, random_state=seed + epoch, n_samples=len(embeddings)
            )

            for i in range(0, len(embeddings_copy), batch_size):
                batch_embeddings = torch.tensor(embeddings_copy[i : i + batch_size])
                batch_labels = torch.tensor(labels_copy[i : i + batch_size], dtype=torch.long)

                assert batch_embeddings.shape[0] == batch_labels.shape[0]
                outputs = model(batch_embeddings)
                assert outputs.shape[0] == batch_embeddings.shape[0]
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.requires_grad_(True)
                loss.backward()
                # apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_sum += loss.item()

            loss_mean = loss_sum / (len(embeddings_copy) / batch_size)
            loss_sum = 0.0
            if epoch % 100 == 0:
                print(f"Loss: {loss_mean} in Epoch {epoch}")

    final_outputs = None
    embeddings = torch.tensor(embeddings)
    labels = torch.tensor(labels)
    with torch.no_grad():
        final_outputs = torch.nn.functional.softmax(model(embeddings), dim=1)

    accuracy = tm.functional.accuracy(
        final_outputs, labels, task="multiclass", num_classes=num_classes, average="weighted"
    )
    assert accuracy is not None
    accuracy_top5 = tm.functional.accuracy(final_outputs, labels, task="multiclass", num_classes=num_classes, top_k=5)
    assert accuracy_top5 is not None
    auroc = tm.functional.auroc(final_outputs, labels, task="multiclass", num_classes=num_classes)
    assert auroc is not None
    f1 = tm.functional.f1_score(final_outputs, labels, task="multiclass", num_classes=num_classes, average="weighted")
    assert f1 is not None
    return {"accuracy": accuracy.item(), "accuracy_top5": accuracy_top5.item(), "auroc": auroc.item(), "f1": f1.item()}


def knn(
    val_embeddings: torch.Tensor,
    val_labels: gtypes.MergedLabels,
    k: int = 5,
    use_train_embeddings: bool = False,
    train_embeddings: Optional[npt.NDArray[np.float_]] = None,
    train_labels: Optional[gtypes.MergedLabels] = None,
) -> Dict[str, Any]:
    if use_train_embeddings:
        print("Using train embeddings for knn")
        return knn_with_train(
            val_embeddings,
            val_labels,
            k=k,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
        )
    else:
        return knn_naive(val_embeddings, val_labels, k=k)


def knn_with_train(
    val_embeddings: torch.Tensor,
    val_labels: gtypes.MergedLabels,
    k: int = 5,
    train_embeddings: Optional[npt.NDArray[np.float_]] = None,
    train_labels: Optional[gtypes.MergedLabels] = None,
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
    val_embeddings = torch.tensor(val_embeddings)
    val_labels = torch.tensor(val_labels)
    train_embeddings = torch.tensor(train_embeddings)  # type: ignore
    train_labels = torch.tensor(train_labels)

    combined_embeddings = torch.cat([train_embeddings, val_embeddings], dim=0)  # type: ignore
    combined_labels = torch.cat([train_labels, val_labels], dim=0)

    num_classes: int = torch.max(combined_labels).item() + 1  # type: ignore
    assert num_classes == len(np.unique(combined_labels))
    if num_classes < k:
        k = num_classes

    distance_matrix = pairwise_euclidean_distance(combined_embeddings)

    distance_matrix.fill_diagonal_(float("inf"))

    _, closest_indices = torch.topk(distance_matrix, k, largest=False)
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
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average="weighted"
    )
    assert accuracy is not None
    accuracy_top5 = tm.functional.accuracy(
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, top_k=5
    )
    assert accuracy_top5 is not None
    auroc = tm.functional.auroc(val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes)
    assert auroc is not None
    f1 = tm.functional.f1_score(
        val_classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average="weighted"
    )
    assert f1 is not None
    print("knn done")
    return {"accuracy": accuracy.item(), "accuracy_top5": accuracy_top5.item(), "auroc": auroc.item(), "f1": f1.item()}


def knn_naive(val_embeddings: torch.Tensor, val_labels: gtypes.MergedLabels, k: int = 5) -> Dict[str, Any]:
    num_classes = np.max(val_labels).item() + 1
    if num_classes < k:
        print(f"Number of classes {num_classes} is smaller than k {k} -> setting k to {num_classes}")
        k = num_classes

    # convert embeddings and labels to tensors
    val_embeddings = torch.tensor(val_embeddings)
    val_labels = torch.tensor(val_labels)

    distance_matrix = pairwise_euclidean_distance(val_embeddings)

    # Ensure distances on the diagonal are set to a large value so they are ignored
    distance_matrix.fill_diagonal_(float("inf"))

    # Find the indices of the closest embeddings for each embedding
    classification_matrix = torch.zeros((len(distance_matrix), k))
    for i in range(k):
        closest_indices = torch.argmin(distance_matrix, dim=1)
        closest_labels = val_labels[closest_indices]
        # Set the distance to the closest embedding to a large value so it is ignored
        distance_matrix[torch.arange(len(distance_matrix)), closest_indices] = float("inf")
        classification_matrix[:, i] = closest_labels
    # Calculate the most common label for each embedding
    # transform classification_matrix of shape (n,k) to (n,num_classes) where num_classes is the number of unique labels
    # the idea is that in the end the classification_matrix contains the probability for each class for each embedding
    classification_matrix_cpy = classification_matrix.clone()
    classification_matrix = torch.zeros((len(classification_matrix), num_classes))
    for i in range(num_classes):
        classification_matrix[:, i] = torch.sum(classification_matrix_cpy == i, dim=1) / k

    accuracy = tm.functional.accuracy(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average="weighted"
    )
    assert accuracy is not None
    accuracy_top5 = tm.functional.accuracy(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, top_k=5
    )
    assert accuracy_top5 is not None
    auroc = tm.functional.auroc(classification_matrix, val_labels, task="multiclass", num_classes=num_classes)
    assert auroc is not None
    f1 = tm.functional.f1_score(
        classification_matrix, val_labels, task="multiclass", num_classes=num_classes, average="weighted"
    )
    assert f1 is not None
    print("knn done")
    return {"accuracy": accuracy.item(), "accuracy_top5": accuracy_top5.item(), "auroc": auroc.item(), "f1": f1.item()}


def pca(
    embeddings: torch.Tensor, labels: torch.Tensor, **kwargs: Any
) -> wandb.Image:  # generate a 2D plot of the embeddings
    num_classes = len(np.unique(labels))
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
    print("pca done")
    return plot


def tsne(
    embeddings: torch.Tensor, labels: torch.Tensor, with_pca: bool = False, count: int = 1000, **kwargs: Any
) -> Optional[wandb.Image]:  # generate a 2D plot of the embeddings
    num_classes = len(np.unique(labels))
    # downsample the embeddings and also the labels to 1000 samples
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
    print("tsne done")
    return plot


def flda_metric(embeddings: torch.Tensor, labels: gtypes.MergedLabels, **kwargs: Any) -> float:  # TODO: test
    # num_classes = len(np.unique(labels))
    # (m_1 - m_2)^2/(s_1^2 + s_2^2)
    mean_var_map = {label: [0.0, 0.0] for label in np.unique(labels)}
    ratio_sum = 0.0

    for label in np.unique(labels):
        class_embeddings = embeddings[labels == label]
        mean = np.mean(class_embeddings, axis=0)
        variance = np.var(class_embeddings, axis=0)
        mean_var_map[label] = [mean, variance]

    for label in np.unique(labels):
        for label2 in np.unique(labels):
            if label == label2:
                continue
            mean1, var1 = mean_var_map[label]
            mean2, var2 = mean_var_map[label2]
            # mean and variance are vectors so use euclidean distance
            ratio = np.linalg.norm(mean1 - mean2) / (np.linalg.norm(var1) + np.linalg.norm(var2))
            # ratio = np.(mean1 - mean2) / (var1 + var2)
            ratio_sum += ratio  # type: ignore

    return ratio_sum / 2  # because we calculate the ratio twice for each pair


def kmeans(
    embeddings: torch.Tensor, num_clusters: int = 2, **kwargs: Any
) -> Tuple[Any, Any]:  # TODO: log some kind of normalized mutual information score
    k_means = sklearn.cluster.KMeans(n_clusters=num_clusters)
    outputs = k_means.fit_predict(embeddings)
    return k_means.cluster_centers_, outputs


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
            "flda": flda_metric,
            "fc_layer": fc_layer,
            # "kmeans": kmeans
        },
    )
    print(results)
