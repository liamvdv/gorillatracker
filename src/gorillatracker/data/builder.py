from pathlib import Path
from typing import Optional, Type

import gorillatracker.type_helper as gtypes
from gorillatracker.data.bristol import BristolDataset
from gorillatracker.data.cxl import CXLDataset, KFoldCXLDataset
from gorillatracker.data.nlet import (
    FlatNletBuilder,
    NletDataModule,
    NletDataset,
    build_onelet,
    build_quadlet,
    build_triplet,
)
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig

BristolDatasetId = "gorillatracker.datasets.bristol.BristolDataset"
CXLDatasetId = "gorillatracker.datasets.cxl.CXLDataset"
KFoldCXLDatasetId = "gorillatracker.datasets.kfold_cxl.KFoldCXLDataset"  # TODO change this cxl.KFoldCXLDataset
SSLDatasetId = "gorillatracker.datasets.ssl.SSLDataset"

dataset_registry: dict[str, Type[NletDataset]] = {
    BristolDatasetId: BristolDataset,
    CXLDatasetId: CXLDataset,
    KFoldCXLDatasetId: KFoldCXLDataset,
    SSLDatasetId: SSLDataset,
}

nlet_requirements: dict[str, FlatNletBuilder] = {
    "softmax": build_onelet,
    "offline": build_triplet,
    "online": build_quadlet,
}


def build_data_module(
    dataset_class_id: str,
    data_dir: Path,
    batch_size: int,
    loss_mode: str,
    workers: int,
    model_transforms: gtypes.TensorTransform,
    training_transforms: gtypes.TensorTransform,
    additional_eval_datasets_ids: list[str] = [],
    additional_eval_data_dirs: list[Path] = [],
    ssl_config: Optional[SSLConfig] = None,
) -> NletDataModule:
    assert dataset_class_id in dataset_registry, f"Dataset class {dataset_class_id} not found in registry"
    assert all(
        [cls_id in dataset_registry for cls_id in additional_eval_datasets_ids]
    ), f"Dataset class not found in registry: {additional_eval_datasets_ids}"
    assert len(additional_eval_datasets_ids) == len(
        additional_eval_data_dirs
    ), "Length mismatch between eval datasets and dirs"

    if dataset_class_id == SSLDatasetId:
        assert ssl_config is not None, "ssl_config must be set for SSLDataset"

    dataset_class = dataset_registry[dataset_class_id]
    eval_datasets = [dataset_registry[cls_id] for cls_id in additional_eval_datasets_ids]

    nlet_builder = next((builder for mode, builder in nlet_requirements.items() if loss_mode.startswith(mode)), None)
    assert nlet_builder is not None, f"Invalid loss mode: {loss_mode}"

    return NletDataModule(
        data_dir=data_dir,
        dataset_class=dataset_class,
        nlet_builder=nlet_builder,
        batch_size=batch_size,
        workers=workers,
        model_transforms=model_transforms,
        training_transforms=training_transforms,
        eval_datasets=eval_datasets,
        eval_data_dirs=additional_eval_data_dirs,
        ssl_config=ssl_config,
    )
