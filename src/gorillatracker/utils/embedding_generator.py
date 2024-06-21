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


def df_from_predictions(predictions: tuple[list[gtypes.Id], torch.Tensor, torch.Tensor]) -> pd.DataFrame:
    prediction_df = pd.DataFrame(columns=["id", "embedding", "label", "input", "label_string"])
    for id, embedding, label in zip(*predictions):
        input_img = Image.open(id)
        prediction_df = pd.concat(
            [
                prediction_df,
                pd.DataFrame(
                    {
                        "id": [id],
                        "embedding": [embedding],
                        "label": [label],
                        "input": [input_img],
                        "label_string": [str(label.item())],
                    }
                ),
            ]
        )

    prediction_df.reset_index(drop=False, inplace=True)
    return prediction_df
