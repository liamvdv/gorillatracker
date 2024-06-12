"""
split = 70% train; 15% test; 15% eval

openset/
    train/
    eval/
    test/
    
    at least 2 individuals are not seen in training.
    C1: at least 1 individual is in eval that is not seen in training
    C2: at least 1 individual is in test that is not seen in training
    C3: the unseen individuals in C1 and C2 must not be the same.    

    Splits are done on individual level. This might produce more photos in test/eval
    that by %.
    
openset-strict/
    train/
    eval/
    test/
    
    train, eval and test are disjoint.
    
openset-strict-half-known/
    train/
    eval/
    test/
    
    train, eval and test are NOT disjoint.
    - half of eval and train are images of individuals that are not seen in training.
    - other half of eval and train are images of individuals that are seen in training.


closedset/
    train/
    eval/
    test/

    all individuals are seen in training.
    C1: This means eval and test MUST NOT contain at least 1 photo of every individual. 
    
    Splits are done on a photos.
"""

import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple, TypeVar, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

"""
As of Python 3.7, the insertion order of keys is preserved in all standard dictionary types in Python, including collections.defaultdict. This change was made part of the language specification in Python 3.7, meaning that dictionaries maintain the order in which keys are first inserted.
"""
assert sys.version_info >= (
    3,
    7,
), "code relies on Python 3.7 explicit dict key insertion ordering guarantee for collections.defaultdict"

Value = Union[Path, List["Entry"]]
Label = Union[str, int]
Metadata = Dict[Any, Any]
T = TypeVar("T")

partitions = ["train", "val", "test"]


@dataclass
class Entry:
    value: Union[Value, List["Entry"]]
    label: Label
    metadata: Metadata


Labeler = Callable[[Path], Label]

# Importers of images, Path is unique identifier.


def read_files(dirpath: Path) -> List[Entry]:
    entries = []
    for filename in os.listdir(dirpath):
        filepath = dirpath / filename
        entry = Entry(filepath, str(filename), {})
        entries.append(entry)
    return entries


def read_dataset_partition(dirpath: Path, labeler: Labeler) -> List[Entry]:
    return [Entry(value, labeler(value), {}) for value in dirpath.glob("*")]


def read_ground_truth(full_images_dirpath: Path) -> List[Entry]:
    """
    Assumed directory structure:
    dirpath/
        <label>_<...>.png
        or
        <label>_<...>.jpg
    """
    entries = []
    image_paths = list(full_images_dirpath.glob("*.jpg"))
    image_paths = image_paths + list(full_images_dirpath.glob("*.png"))
    for image_path in image_paths:
        if "_" in image_path.name:
            label = image_path.name.split("_")[0]
        else:
            label = image_path.name.split("-")[0]
        entry = Entry(image_path, label, {})
        entries.append(entry)
    return entries


# Business Logic


def group(images: List[Entry]) -> List[Entry]:
    """preserves order:
    group order determined by first seen element with group id.
    inner group order in order of occurance in array.
    """
    individuals = defaultdict(list)
    for img in images:
        individuals[img.label].append(img)
    return [Entry(images, label, {}) for label, images in individuals.items()]


def ungroup(individuals: List[Entry]) -> List[Entry]:
    """preserves order"""
    assert all(
        isinstance(individual.value, list) for individual in individuals
    ), "Sanity check that is is not called on Paths."
    return [img for individual in individuals for img in individual.value]  # type: ignore


def consistent_random_permutation(arr: List[T], unique_key: Callable[[Any], Any], seed: int) -> List[T]:
    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(len(arr), generator=g).tolist()
    arr = sorted(arr, key=unique_key)
    return [arr[i] for i in idxs]


def compute_split(samples: int, train: int, val: int, test: int, kfold: bool) -> Tuple[int, int, int]:
    if train + val + test != 100:
        raise ValueError("sum of train, val and test must be 100")

    # sum must add; if partial, count towards train
    train_count = train / 100 * samples
    val_count = val / 100 * samples
    test_count = test / 100 * samples

    if not kfold:
        # we'll enfore at least 1
        train_count = max(train_count, 1)
        val_count = max(val_count, 1)
        test_count = max(test_count, 1)

    val_count = int(val_count)
    test_count = int(test_count)
    train_count = samples - (val_count + test_count)
    return train_count, val_count, test_count


# You must ensure this is set to True when pushed. Do not keep a TEST = False
# version on main.
TEST = False


def copy(src: Path, dst: Path) -> None:
    if TEST:
        print(f"{src} -> {dst}")
    else:
        shutil.copyfile(src, dst)


def write_entries(entries: List[Entry], outdir: Path) -> None:
    os.makedirs(outdir, exist_ok=True)
    for entry in entries:
        assert isinstance(entry.value, Path)
        newpath = outdir / entry.value.name
        copy(entry.value, newpath)


def splitter(
    images: List[Entry],
    mode: Literal["openset", "closedset"],
    train: int,
    val: int,
    test: int,
    seed: int,
    min_train_count: int = 3,
) -> Tuple[List[Entry], List[Entry], List[Entry]]:
    assert train + val + test == 100, "train, val, test must sum to 100."
    random.seed(seed)
    # This shuffe is preserved throughout groups and ungroups.
    # because python dicts are ordered by insertion.
    # We can now simply pop from the start / end to get random images.
    images = consistent_random_permutation(images, lambda x: x.value, seed)
    train_bucket: List[Entry] = []
    val_bucket: List[Entry] = []
    test_bucket: List[Entry] = []

    individums = group(images)
    if mode == "closedset":
        train_count, val_count, test_count = compute_split(len(images), train, val, test, False)
        for individum in individums:
            assert isinstance(individum.value, list)
            n = len(individum.value)
            # only take min_train_count images from each individual and update the individual's value
            selection, individum.value = individum.value[:min_train_count], individum.value[min_train_count:]
            if len(selection) != min_train_count:
                print(
                    f"WARN: individual {individum.label} has less than min_train_count={min_train_count} images. It has {n} images. You may consider discarding it."
                )
            train_bucket.extend(selection)
        # need to shuffle again because we ungroup() will return images of individuals sequentially
        rest = consistent_random_permutation(ungroup(individums), lambda x: x.value, seed + 1)
        val_bucket, rest = rest[:val_count], rest[val_count:]
        test_bucket, rest = rest[:test_count], rest[test_count:]
        train_bucket.extend(rest)
    elif mode == "openset":
        # At least one unseen individum in test and eval.
        train_count, val_count, test_count = compute_split(len(individums), train, val, test, False)
        print(
            f"Unique Individuals (Image Count Bias): Total={len(individums)}, Train={train_count}, Val={val_count}, Test={test_count}"
        )
        train_bucket = ungroup(individums[:train_count])
        val_bucket = ungroup(individums[train_count : train_count + val_count])
        test_bucket = ungroup(individums[train_count + val_count :])

        n = len(train_bucket)

        for individual in group(train_bucket):
            assert isinstance(individual.value, list)
            n = len(individual.value)
            if n < min_train_count:
                print(
                    f"WARN: train set: individual {individual.label} has less than min_train_count={min_train_count} images. It has {n} images. You may consider discarding it."
                )

    return train_bucket, val_bucket, test_bucket


def kfold_splitter(
    images: List[Entry],
    mode: Literal["openset", "closedset"],
    trainval: int,
    test: int,
    seed: int,
    k: int = 5,
) -> Tuple[List[List[Entry]], List[Entry]]:
    assert trainval + test == 100, "trainval, test must sum to 100."
    random.seed(seed)
    # This shuffe is preserved throughout groups and ungroups.
    # because python dicts are ordered by insertion.
    # We can now simply pop from the start / end to get random images.
    images = consistent_random_permutation(images, lambda x: x.value, seed)
    fold_buckets: List[List[Entry]] = [[] for _ in range(k)]
    test_bucket: List[Entry] = []

    fold_bucket_iterator = 0

    individums = group(images)
    if mode == "closedset":
        trainval_count, _, test_count = compute_split(len(images), trainval, 0, test, True)
        fold_bucket_size = trainval_count // k

        # TODO: smarter way of splitting the images into k folds
        for i in range(k):
            fold_buckets[i] = images[fold_bucket_iterator : fold_bucket_iterator + fold_bucket_size]
            fold_bucket_iterator += fold_bucket_size

        test_bucket = images[fold_bucket_iterator:]

    elif mode == "openset":
        # At least one unseen individum in test and eval.
        # NOTE number of actual images in each fold (and test) can vary due to different number of images per individual.
        trainval_count, _, test_count = compute_split(len(individums), trainval, 0, test, True)
        fold_bucket_size = trainval_count // k

        sorted_individums = sorted(individums, key=lambda x: len(x.value), reverse=True)  # type: ignore

        sizes = [0 for _ in range(k)]
        num_images_fold = [0 for _ in range(k)]

        test_size = 0

        print(
            f"Unique Individuals (Image Count Bias): Total={len(individums)}, Train={trainval_count}, Test={test_count}"
        )
        for j in range(fold_bucket_size):
            for i in range(k):
                smallest_fold_indices = np.where(sizes == np.min(sizes))[0]
                individual = sorted_individums[fold_bucket_iterator]

                min_ind = smallest_fold_indices[0]
                for i in smallest_fold_indices:
                    if num_images_fold[i] < num_images_fold[min_ind]:
                        min_ind = i

                fold_buckets[min_ind].extend(ungroup([individual]))
                sizes[min_ind] += 1
                num_images_fold[min_ind] += len(individual.value)  # type: ignore
                fold_bucket_iterator += 1
            test_bucket.extend(ungroup([sorted_individums[fold_bucket_iterator]]))
            test_size += 1
            fold_bucket_iterator += 1

        test_bucket.extend(ungroup(individums[fold_bucket_iterator:]))
        test_size += len(individums[fold_bucket_iterator:])

    for i, fold_bucket in enumerate(fold_buckets):
        print(f"Fold {i}: {len(fold_bucket)} images, {sizes[i]} individuals")
    print(f"Test: {len(test_bucket)} images, {test_size} individuals")

    return fold_buckets, test_bucket


def stats_and_confirm(name: str, full: List[Entry], train: List[Entry], val: List[Entry], test: List[Entry]) -> None:
    print(f"Stats for {name}")
    print(f"Distribution over {len(full)} images:")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    if input("Continue? (yes,N)") != "yes":
        print("abort.")
        exit(0)


def generate_split(
    dataset_dir: Path = Path("data/ground_truth/cxl/full_images"),
    output_dir: Path = Path("data/splits"),
    mode: Literal["openset", "closedset"] = "closedset",
    seed: int = 42,
    train: int = 70,
    val: int = 15,
    test: int = 15,
    min_train_count: int = 3,
) -> Path:
    name = f"{str(dataset_dir).replace('/', '-')}-{mode}-mintraincount-{min_train_count}-seed-{seed}-train-{train}-val-{val}-test-{test}"
    output_dir = output_dir / name

    images = read_ground_truth(dataset_dir)

    train_bucket, val_bucket, test_bucket = splitter(
        images,
        mode=mode,
        train=train,
        val=val,
        test=test,
        seed=seed,
        min_train_count=min_train_count,
    )
    stats_and_confirm(name, images, train_bucket, val_bucket, test_bucket)
    write_entries(train_bucket, output_dir / "train")
    write_entries(val_bucket, output_dir / "val")
    write_entries(test_bucket, output_dir / "test")
    return output_dir


def generate_simple_split(
    dataset_dir: Path = Path("data/ground_truth/cxl/full_images"),
    output_dir: Path = Path("data/splits"),
    seed: int = 42,
    train: int = 70,
    val: int = 15,
    test: int = 15,
) -> Path:
    name = f"{str(dataset_dir).replace('/', '-')}-seed-{seed}-train-{train}-val-{val}-test-{test}"
    files = read_files(dataset_dir)
    files = consistent_random_permutation(files, lambda x: x.value, seed)
    output_dir = output_dir / name
    train_count, val_count, test_count = compute_split(len(files), train, val, test, False)
    train_set = files[:train_count]
    val_set = files[train_count : train_count + val_count]
    test_set = files[train_count + val_count :]
    stats_and_confirm(name, files, train_set, val_set, test_set)
    write_entries(train_set, output_dir / "train")
    write_entries(val_set, output_dir / "val")
    write_entries(test_set, output_dir / "test")
    return output_dir


def generate_kfold_split(
    dataset_dir: Path = Path("data/ground_truth/cxl/face_images"),
    output_dir: Path = Path("data/splits"),
    mode: Literal["openset", "closedset"] = "closedset",
    seed: int = 42,
    trainval: int = 80,
    test: int = 20,
    k: int = 5,
) -> Path:
    name = f"{str(dataset_dir).replace('/', '-')}-kfold-{mode}-seed-{seed}-trainval-{trainval}-test-{test}-k-{k}"
    output_dir = output_dir / name

    images = read_ground_truth(dataset_dir)

    fold_buckets, test_bucket = kfold_splitter(
        images,
        mode=mode,
        trainval=trainval,
        test=test,
        seed=seed,
        k=k,
    )
    # stats_and_confirm(name, images, fold_buckets, [], test_bucket)
    for i, fold_bucket in enumerate(fold_buckets):
        write_entries(fold_bucket, output_dir / f"fold-{i}")
    write_entries(test_bucket, output_dir / "test")
    return output_dir


def copy_corresponding_images(data_dir: str, img_dir: str = "ground_truth/cxl/full_images") -> None:
    """
    Copy each image from img_dir to data_dir if a file with a corresponding name exists in data_dir.
    """
    data_dir_path = Path(f"data/{data_dir}")
    img_dir_path = Path(f"data/{img_dir}")
    to_copy = []

    for data_file in os.listdir(data_dir_path):
        base_name = Path(data_file).stem
        corresponding_img_file = img_dir_path / (base_name + ".png")
        assert corresponding_img_file.exists()
        to_copy.append(corresponding_img_file)

    assert len(to_copy) == len(os.listdir(data_dir_path))

    for img_file in to_copy:
        copy(img_file, data_dir_path / img_file.name)


def read_dataset(dirpath: Path, labeler: Labeler) -> Dict[str, List[Entry]]:
    return {partition: read_dataset_partition(dirpath / partition, labeler) for partition in partitions}


def _merge_dataset_splits(target: str, ds1: str, ds2: str, ds1_labeler: Labeler, ds2_labeler: Labeler) -> None:
    ds1_path = Path("data", ds1)
    ds2_path = Path("data", ds2)
    target_path = Path("data", target)
    ds1_entries = read_dataset(ds1_path, ds1_labeler)
    ds2_entries = read_dataset(ds2_path, ds2_labeler)

    def process(entry: Entry) -> Entry:
        assert isinstance(entry.value, Path)
        value = Path(f"{entry.label}__{entry.value.name}")  # concat file name to preserve information
        return Entry(value, entry.label, entry.metadata | {"original": entry.value})

    # Logical Merge
    merged_entries = {
        partition: list(map(process, ds1_entries[partition] + ds2_entries[partition])) for partition in partitions
    }

    # Assertions
    for partition in partitions:
        # Entry Uniqueness
        value_set = set([str(entry.value) for entry in merged_entries[partition]])
        if len(merged_entries[partition]) != len(value_set):
            raise ValueError(f"WARN: {partition} no longer unique. Merging produces collisions.")

        # Label Uniqueness
        ds1_label_set = set([entry.label for entry in ds1_entries[partition]])
        ds2_label_set = set([entry.label for entry in ds2_entries[partition]])
        intersection = ds1_label_set & ds2_label_set
        if len(intersection) > 0:
            raise ValueError(
                f"WARN: {partition}: Labels {intersection} exist in both datasets. Merging produces collisions."
            )
    stats_and_confirm(
        target,
        merged_entries["train"] + merged_entries["val"] + merged_entries["test"],
        merged_entries["train"],
        merged_entries["val"],
        merged_entries["test"],
    )
    # Physical Merge
    for partition in partitions:
        if not TEST:
            (target_path / partition).mkdir(parents=True, exist_ok=True)  # idempotent
        for entry in merged_entries[partition]:
            assert isinstance(entry.value, Path)
            copy(entry.metadata["original"], target_path / partition / entry.value)


def merge_labeler(x: Path) -> str:
    return x.name.split("__")[0]


def common_labeler(x: Path) -> str:
    return x.name.split("_")[0]


def get_labeler_for_dataset(ds: str) -> Labeler:
    # DO NOT REORDER
    if "splits/merged" in ds:
        return merge_labeler
    elif "bristol" in ds:
        return common_labeler
    elif "rohan-cxl" in ds:
        return common_labeler
    elif "cxl" in ds:
        return common_labeler
    else:
        raise ValueError(f"Labeler unknown for dataset: {ds}")


def merge_dataset_splits(ds1: str, ds2: str) -> None:
    ds1_labeler = get_labeler_for_dataset(ds1)
    ds2_labeler = get_labeler_for_dataset(ds2)
    target = f"splits/merged-{ds1.replace('/', '-')}-{ds2.replace('/', '-')}"
    _merge_dataset_splits(target, ds1, ds2, ds1_labeler, ds2_labeler)


if __name__ == "__main__":
    dir = generate_kfold_split(
        dataset_dir=Path("/workspaces/gorillatracker/data/ground_truth/bristol/full_images"),
        output_dir=Path("compressed/test"),
        mode="openset",
        seed=42,
        trainval=80,
        test=20,
        k=5,
    )

# NOTE(liamvdv): Images per Individual heavly screwed. Image distribution around 73 / 17 / 10 for Individual Distribution 50 / 25 / 25.
# dir = generate_split(
#     dataset_dir="data/ground_truth/cxl/face_images", mode="openset", seed=42, reid_factor_test=0, reid_factor_val=0, train=50, val=25, test=25
# )
