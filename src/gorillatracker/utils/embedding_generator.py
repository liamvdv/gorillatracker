import pandas as pd
import torch
from lightning import Trainer
from PIL import Image
from torch.utils.data import DataLoader

import gorillatracker.type_helper as gtypes
from gorillatracker.model import BaseModule

def generate_embeddings(
    model: BaseModule, dataloader: DataLoader[gtypes.Nlet]
) -> tuple[list[gtypes.Id], torch.Tensor, torch.Tensor]:
    model.eval()
    model.freeze()
    trainer = Trainer()
    batched_predictions = trainer.predict(model, dataloader)
    ids, embeddings, labels = zip(*batched_predictions)
    print(len(ids[0][0]))
    print(embeddings[0].shape)
    print(len(labels[0][0]))
    flat_ids = [id for sublist in ids for id in sublist[0]]
    print(flat_ids)
    # flat_ids = list(sum(flat_ids, ()))
    concatenated_embeddings = torch.cat(embeddings)
    flat_labels = [label for sublist in labels for label in sublist[0]]
    # labels = tuple(lst[0] for lst in labels)
    print(flat_labels)
    # concatenated_labels = torch.cat(labels)
    print(len(flat_ids), len(concatenated_embeddings), len(flat_labels))
    return flat_ids, concatenated_embeddings, flat_labels

def generate_ssl_embeddings(
    model: BaseModule, dataloader: DataLoader[gtypes.Nlet]
) -> tuple[list[gtypes.Id], torch.Tensor, torch.Tensor]:
    model.eval()
    model.freeze()
    trainer = Trainer()
    batched_predictions = trainer.predict(model, dataloader)
    ids, embeddings, labels = zip(*batched_predictions)
    flat_ids = [id for sublist in ids for id in sublist]
    # flat_ids = list(sum(flat_ids, ()))
    concatenated_embeddings = torch.cat(embeddings)
    # labels = tuple(lst[0].reshape(1) for lst in labels)
    flat_labels = [label for sublist in labels for label in sublist]
    print(flat_labels)
    # labels = tuple(lst[0] for lst in labels)
    # concatenated_labels = torch.cat(labels)
    print(len(flat_ids), len(concatenated_embeddings), len(flat_labels))
    return flat_ids, concatenated_embeddings, flat_labels


def df_from_predictions(predictions: tuple[list[gtypes.Id], torch.Tensor, torch.Tensor]) -> pd.DataFrame:
    prediction_df = pd.DataFrame(columns=["id", "embedding", "label", "label_string"])
    ids = []
    embeddings = []
    labels = []
    label_strings = []

    for id, embedding, label in zip(*predictions):
        ids.append(id)
        embeddings.append(embedding.numpy())
        labels.append(label)
        if isinstance(id, str):
            label_strings.append(id.split("/")[-1].split("_")[0])
        else:
            label_strings.append(str(label.item()))

    # Create the DataFrame once
    prediction_df = pd.DataFrame({
        "id": ids,
        "embedding": embeddings,
        "label": labels,
        "label_string": label_strings,
    })

    prediction_df.reset_index(drop=False, inplace=True)
    return prediction_df
