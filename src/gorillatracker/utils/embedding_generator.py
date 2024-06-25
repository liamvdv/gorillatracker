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
    flat_ids = [id for sublist in ids for id in sublist]
    concatenated_embeddings = torch.cat(embeddings)
    concatenated_labels = torch.cat(labels)
    return flat_ids, concatenated_embeddings, concatenated_labels


def df_from_predictions(predictions: tuple[list[str], torch.Tensor, torch.Tensor]) -> pd.DataFrame:
    """
    Returns a DataFrame with the following columns:
    - id: str|int
    - embedding: torch.Tensor
    - label: int
    - input: PIL.Image
    - label_string: str
    """
    ids, embeddings, labels = predictions

    id_list = []
    embedding_list = []
    label_list = []
    input_list = []
    label_string_list = []

    for id, embedding, label in zip(ids, embeddings, labels):
        input_img = Image.open(id)
        id_list.append(id)
        embedding_list.append(embedding)
        label_list.append(int(label))
        input_list.append(input_img)
        label_string_list.append(str(label.item()))

    prediction_df = pd.DataFrame(
        {
            "id": id_list,
            "embedding": embedding_list,
            "label": label_list,
            "input": input_list,
            "label_string": label_string_list,
        }
    )

    prediction_df.reset_index(drop=True, inplace=True)

    return prediction_df
