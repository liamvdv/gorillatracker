from pathlib import Path
from typing import Literal, Optional, Type

import gorillatracker.type_helper as gtypes
from gorillatracker.data.nlet import (
    CrossEncounterSupervisedDataset,
    CrossEncounterSupervisedKFoldDataset,
    FlatNletBuilder,
    HardCrossEncounterSupervisedDataset,
    HardCrossEncounterSupervisedKFoldDataset,
    NletDataModule,
    NletDataset,
    SupervisedDataset,
    SupervisedKFoldDataset,
    build_onelet,
    build_pair,
    build_quadlet,
    build_triplet,
)
from gorillatracker.data.ssl import SSLDataset
from gorillatracker.ssl_pipeline.ssl_config import SSLConfig

HardCrossEncounterSupervisedKFoldDatasetId = "gorillatracker.datasets.kfold_cxl.HardCrossEncounterKFoldCXLDataset"
HardCrossEncounterSupervisedDatasetId = "gorillatracker.datasets.cxl.HardCrossEncounterCXLDataset"
CrossEncounterKFoldSupervisedDatasetId = "gorillatracker.datasets.kfold_cxl.CrossEncounterKFoldCXLDataset"
CrossEncounterSupervisedDatasetId = "gorillatracker.datasets.cxl.CrossEncounterCXLDataset"
BristolDatasetId = "gorillatracker.datasets.bristol.BristolDataset"
CXLDatasetId = "gorillatracker.datasets.cxl.CXLDataset"
CZooDatasetId = "gorillatracker.datasets.chimp.CZooDataset"
CTaiDatasetId = "gorillatracker.datasets.chimp.CTaiDataset"
Cows2021DatasetId = "gorillatracker.datasets.cows2021.Cows2021Dataset"
SeaturtleDatasetId = "gorillatracker.datasets.seaturtle.SeaturtleDataset"
ATRWDatasetId = "gorillatracker.datasets.atrw.ATRWDataset"
KFoldCZooDatasetId = "gorillatracker.datasets.chimp.KFoldCZooDataset"
KFoldCTaiDatasetId = "gorillatracker.datasets.chimp.KFoldCTaiDataset"
KFoldCXLDatasetId = "gorillatracker.datasets.kfold_cxl.KFoldCXLDataset"  # TODO change this cxl.KFoldCXLDataset
KFoldCows2021DatasetId = "gorillatracker.datasets.cows2021.KFoldCows2021Dataset"
KFoldSeaturtleDatasetId = "gorillatracker.datasets.seaturtle.KFoldSeaturtleDataset"
KFoldATRWDatasetId = "gorillatracker.datasets.atrw.KFoldATRWDataset"
SSLDatasetId = "gorillatracker.datasets.ssl.SSLDataset"

dataset_registry: dict[str, Type[NletDataset]] = {
    BristolDatasetId: SupervisedDataset,
    CXLDatasetId: SupervisedDataset,
    KFoldCXLDatasetId: SupervisedKFoldDataset,
    HardCrossEncounterSupervisedKFoldDatasetId: HardCrossEncounterSupervisedKFoldDataset,
    HardCrossEncounterSupervisedDatasetId: HardCrossEncounterSupervisedDataset,
    CrossEncounterKFoldSupervisedDatasetId: CrossEncounterSupervisedKFoldDataset,
    CrossEncounterSupervisedDatasetId: CrossEncounterSupervisedDataset,
    SSLDatasetId: SSLDataset,
    CZooDatasetId: SupervisedDataset,
    CTaiDatasetId: SupervisedDataset,
    Cows2021DatasetId: SupervisedDataset,
    SeaturtleDatasetId: SupervisedDataset,
    ATRWDatasetId: SupervisedDataset,
    KFoldCZooDatasetId: SupervisedKFoldDataset,
    KFoldCTaiDatasetId: SupervisedKFoldDataset,
    KFoldCows2021DatasetId: SupervisedKFoldDataset,
    KFoldSeaturtleDatasetId: SupervisedKFoldDataset,
    KFoldATRWDatasetId: SupervisedKFoldDataset,
}

nlet_requirements: dict[str, FlatNletBuilder] = {
    "softmax": build_onelet,
    "ntxent": build_pair,
    "offline": build_triplet,
    "online": build_quadlet,
    "distillation": build_triplet,
    "mae_mse": build_onelet,
}


def force_nlet_builder(builder_identifier: Literal["onelet", "pair", "triplet", "quadlet"]) -> None:
    if builder_identifier:
        global nlet_requirements
        nlet_requirements = {
            "softmax": build_onelet if builder_identifier == "onelet" else build_triplet,
            "ntxent": build_pair if builder_identifier == "pair" else build_pair,
            "offline": build_triplet if builder_identifier == "triplet" else build_quadlet,
            "online": build_quadlet if builder_identifier == "quadlet" else build_onelet,
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
    dataset_names: list[str] = [],
    ssl_config: Optional[SSLConfig] = None,
) -> NletDataModule:
    assert dataset_class_id in dataset_registry, f"Dataset class {dataset_class_id} not found in registry"
    assert all(
        [cls_id in dataset_registry for cls_id in additional_eval_datasets_ids]
    ), f"Dataset class not found in registry: {additional_eval_datasets_ids}"
    assert len(additional_eval_datasets_ids) == len(
        additional_eval_data_dirs
    ), "Length mismatch between eval datasets and dirs"
    # additional_dataset_names can be empty, but if not, it must have the same length as additional_eval_datasets_ids
    assert (
        not dataset_names or len(dataset_names) == len(additional_eval_datasets_ids) + 1
    ), "Length mismatch between dataset_names and eval datasets"

    if dataset_class_id == SSLDatasetId:
        assert ssl_config is not None, "ssl_config must be set for SSLDataset"

    dataset_class = dataset_registry[dataset_class_id]
    eval_datasets = [dataset_registry[cls_id] for cls_id in additional_eval_datasets_ids]
    dataset_ids = [cls_id.split(".")[-1] for cls_id in ([dataset_class_id] + additional_eval_datasets_ids)]
    dataset_names = dataset_names if dataset_names else dataset_ids
    print(f"Dataset names: {dataset_names}")

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
        dataset_ids=dataset_ids,
        dataset_names=dataset_names,
        eval_data_dirs=additional_eval_data_dirs,
        ssl_config=ssl_config,
    )
