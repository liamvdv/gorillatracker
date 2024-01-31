from typing import Any

import torch
import wandb
import pandas as pd
from tqdm import tqdm


# load model from wandb artifact
def load_model_from_artifact(artifact: wandb.Artifact, model: Any, device: str) -> Any: # TODO
    model_state_dict = artifact.get('model_state_dict').state_dict()
    model.load_state_dict(model_state_dict)
    model.to(device)
    return model

# generate embeddings (dataframe) from model and dataset
def generate_embeddings(model, dataset, device): # TODO
    model.eval()
    embeddings = []
    df = pd.DataFrame(columns=['embedding', 'label', 'input', 'label_string'])
    with torch.no_grad():
        for batch_inputs, batch_labels in tqdm(dataset): # TODO
            batch_inputs = batch_inputs.to(device)
            embeddings = model(batch_inputs)
            for i in range(len(batch_inputs)):
                df.append({'embedding': embeddings[i], 'label': batch_labels[i], 'input': batch_inputs[i], 'label_string': dataset.mapping[batch_labels[i]]}, ignore_index=True, inplace=True)
    df.reset_index(drop=False, inplace=True)
    return df

