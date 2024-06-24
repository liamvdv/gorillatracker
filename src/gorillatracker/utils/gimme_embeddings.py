from pathlib import Path
from gorillatracker.data.nlet import NletDataModule, SupervisedDataset, build_onelet
from gorillatracker.utils.embedding_generator import generate_embeddings, df_from_predictions
from gorillatracker.utils.wandb_loader import get_model_from_run, get_run

# python3 train.py --config_path cfgs/swinv2_cxl.yml

# gorillas/Embedding-SwinV2Base-CXL-Open/model-yl2lx567:v12
data_dir = "/workspaces/gorillatracker/data/supervised/splits/cxl_faces_square_all-in-val"
run_url = "https://wandb.ai/gorillas/Embedding-SwinV2-CXL-Open/runs/kqs3myy5"

run = get_run(run_url)
model = get_model_from_run(run)
data_module = NletDataModule(  
    data_dir=Path(data_dir),
    dataset_class=SupervisedDataset,
    nlet_builder=build_onelet,
    batch_size=32,
    workers=10,
    model_transforms=model.get_training_transforms(),
    training_transforms=lambda x: x,
    dataset_names=["Inference"],
)

data_module.setup("validate")
dataloader = data_module.val_dataloader()

predications = generate_embeddings(model, dataloader)
df = df_from_predictions(predications)
df.to_pickle("embeddings.pkl")