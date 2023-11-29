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
    

closedset/
    train/
    eval/
    test/

    all individuals are seen in training.
    C1: This means eval and test MUST NOT contain at least 1 photo of every individual. 
    
    Splits are done on a photos.
"""
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Union

import torch

"""
As of Python 3.7, the insertion order of keys is preserved in all standard dictionary types in Python, including collections.defaultdict. This change was made part of the language specification in Python 3.7, meaning that dictionaries maintain the order in which keys are first inserted.
"""
assert sys.version_info >= (
    3,
    7,
), "code relies on Python 3.7 explicit dict key insertion ordering guarantee for collections.defaultdict"

Value = Path
Label = Union[str, int]
Metadata = dict


@dataclass
class Entry:
    value: Union[Value, List["Entry"]]
    label: Label
    metadata: Metadata


# Importers of images, Path is unique identifier.


def read_files(dirpath: str) -> List[Entry]:
    entries = []
    for filename in os.listdir(dirpath):
        filepath = Path(dirpath, filename)
        entry = Entry(filepath, filename, {})
        entries.append(entry)
    return entries


def read_ground_truth_cxl(full_images_dirpath: str) -> List[Entry]:
    entries = []
    for filename in os.listdir(full_images_dirpath):
        assert filename.endswith(".png")
        label, rest = re.split(r"[_\s]", filename, maxsplit=1)
        entry = Entry(Path(full_images_dirpath, filename), label, {})
        entries.append(entry)
    return entries


def read_ground_truth_bristol() -> List[Entry]:
    raise NotImplementedError()


# Business Logic


def group(images: List[Entry]) -> List[Entry]:
    """preservers order:
    group order determined by first seen element with group id.
    inner group order in order of occurance in array.
    """
    individuals = defaultdict(list)
    for img in images:
        individuals[img.label].append(img)
    return [Entry(images, label, {}) for label, images in individuals.items()]


def ungroup(individuals: List[Entry]) -> List[Entry]:
    """preservers order"""
    return [img for individual in individuals for img in individual.value]


def ungroup_with_must_include_mask(individuals: List[Entry], k: int) -> Tuple[List[Entry], List[bool]]:
    """preservers order"""
    images = []
    must_include_mask = []
    for individual in individuals:
        for i, img in enumerate(individual.value):
            images.append(img)
            must_include_mask.append(i < k)
    return images, must_include_mask


def consistent_random_permutation(arr: list, unique_key, seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(len(arr), generator=g).tolist()
    arr = sorted(arr, key=unique_key)
    return [arr[i] for i in idxs]


def compute_split(samples: int, train: int, val: int, test: int):
    if train + val + test != 100:
        raise ValueError("sum of train, val and test must be 100")

    # sum must add; if partial, count towards train
    train_count = train / 100 * samples
    val_count = val / 100 * samples
    test_count = test / 100 * samples

    # we'll enfore at least 1
    train_count = max(train_count, 1)
    val_count = max(val_count, 1)
    test_count = max(test_count, 1)

    val_count = int(val_count)
    test_count = int(test_count)
    train_count = samples - (val_count + test_count)
    assert train_count >= 1
    return train_count, val_count, test_count


# You must ensure this is set to True when pushed. Do not keep a TEST = False
# version on main.
TEST = True


def copy(src: Path, dst: Path):
    if TEST:
        print(f"{src} -> {dst}")
    else:
        shutil.copyfile(src, dst)


def write_entries(entries: List[Entry], outdir: Path):
    os.makedirs(outdir, exist_ok=True)
    for entry in entries:
        assert isinstance(entry.value, Value)
        newpath = outdir / entry.value.name
        copy(entry.value, newpath)


def sample_and_remove(arr, samples: int, mask=None):
    if mask is None:
        mask = [True] * len(arr)
    assert len(arr) == len(mask)
    view = [arr[i] for i, b in enumerate(mask) if b]
    assert len(view) > samples, "masked array has less entries than required samples"
    sampled = random.sample(view, samples)
    remaining = [e for e in arr if e not in sampled]
    return sampled, remaining


def splitter(
    images: List[Entry],
    mode: Literal["openset", "openset"],
    train: int,
    val: int,
    test: int,
    seed: int,
    min_train_count=3,
    reid_factor_val: int = None,
    reid_factor_test: int = None,
):
    assert train + val + test == 100, "train, val, test must sum to 100."
    random.seed = seed
    # This shuffe is preserved throughout groups and ungroups.
    # because python dicts are ordered by insertion.
    # We can now simply pop from the start / end to get random images.
    images = consistent_random_permutation(images, lambda x: x.value, seed)
    train_bucket = []
    val_bucket = []
    test_bucket = []

    individums = group(images)
    if mode == "closedset":
        train_count, val_count, test_count = compute_split(len(images), train, val, test)
        for individum in individums:
            n = len(individum.value)
            selection, individum.value = individum.value[:min_train_count], individum.value[min_train_count:]
            if len(selection) != min_train_count:
                print(
                    f"WARN: individual {individum.label} has less than min_train_count={min_train_count} images. It has {n} images. You may consider discarding it."
                )
            train_bucket.extend(selection)
        rest = ungroup(individums)
        val_bucket, rest = rest[:val_count], rest[val_count:]
        assert len(val_bucket) == val_count, "Dataset too small: Not enough images left to fill validation."
        test_bucket, rest = rest[:test_count], rest[test_count:]
        assert len(test_bucket) == test_count, "Dataset too small: Not enough images left to fill test."
        train_bucket.extend(rest)
    elif mode == "openset":
        # At least one unseen individuum in test and eval.
        train_count, val_count, test_count = compute_split(len(individums), 70, 15, 15)
        train_bucket, train_must_include_mask = ungroup_with_must_include_mask(
            individums[:train_count], k=min_train_count
        )
        val_bucket = ungroup(individums[train_count : train_count + val_count])
        test_bucket = ungroup(individums[train_count + val_count :])
        # NOTE(liamvdv): We'll now move back a fraction of images from train_bucket to val and test
        #                to ensure we also look at reidentification of known faces.
        n = len(train_bucket)
        reid_count_val = int(reid_factor_val / 100 * n)
        reid_count_test = int(reid_factor_test / 100 * n)
        inv_mask = [not v for v in train_must_include_mask]
        reid_samples, train_bucket = sample_and_remove(train_bucket, reid_count_val + reid_count_test, mask=inv_mask)
        assert len(reid_samples) == reid_count_test + reid_count_val, "sanity check"
        val_bucket += reid_samples[:reid_count_val]
        test_bucket += reid_samples[reid_count_val:]

        for individual in group(train_bucket):
            n = len(individual.value)
            if n < min_train_count:
                print(
                    f"WARN: train set: individual {individual.label} has less than min_train_count={min_train_count} images. It has {n} images. You may consider discarding it."
                )
    return train_bucket, val_bucket, test_bucket


def stats_and_confirm(name, full, train, val, test):
    print(f"Stats for {name}")
    print(f"Distribution over {len(full)} images:")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test: {len(test)}")
    if input("Continue? (yes,N)") != "yes":
        print("abort.")
        exit(0)


def generate_split(
    dataset="ground_truth/cxl/full_images",
    mode: Literal["openset", "closedset"] = "closedset",
    seed=42,
    train=70,
    val=15,
    test=15,
    min_train_count=3,
    reid_factor_val: int = None,
    reid_factor_test: int = None,
):
    reid_factors = "-"
    if mode == "openset":
        assert isinstance(reid_factor_val, int) and isinstance(
            reid_factor_test, int
        ), "must pass reid_factor_test if mode=openset"
        reid_factors = f"-reid-val-{reid_factor_val}-test-{reid_factor_test}"
    name = f"splits/{dataset.replace('/', '-')}-{mode}{reid_factors}-mintraincount-{min_train_count}-seed-{seed}-train-{train}-val-{val}-test-{test}"
    outdir = Path(f"data/{name}")
    images = read_ground_truth_cxl(f"data/{dataset}")
    train, val, test = splitter(
        images,
        mode=mode,
        train=train,
        val=val,
        test=test,
        seed=seed,
        min_train_count=min_train_count,
        reid_factor_test=reid_factor_test,
        reid_factor_val=reid_factor_val,
    )
    stats_and_confirm(name, images, train, val, test)
    write_entries(train, outdir / "train")
    write_entries(val, outdir / "val")
    write_entries(test, outdir / "test")
    return outdir


def generate_simple_split(
    dataset="ground_truth/cxl/full_images",
    seed=42,
    train=70,
    val=15,
    test=15,
):
    name = f"splits/{dataset.replace('/', '-')}-seed-{seed}-train-{train}-val-{val}-test-{test}"
    files = read_files(f"data/{dataset}")
    files = consistent_random_permutation(files, lambda x: x.value, seed)
    outdir = Path(f"data/{name}")
    train_count, val_count, test_count = compute_split(len(files), train, val, test)
    train_set = files[:train_count]
    val_set = files[train_count : train_count + val_count]
    test_set = files[train_count + val_count :]
    stats_and_confirm(name, files, train_set, val_set, test_set)
    write_entries(train_set, outdir / "train")
    write_entries(val_set, outdir / "val")
    write_entries(test_set, outdir / "test")


def copy_corresponding_images(data_dir: str, img_dir="ground_truth/cxl/full_images"):
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


# if __name__ == "__main__":
#     dir = generate_simple_split(dataset="ground_truth/cxl/full_images_body_bbox", seed=42)
#     copy_corresponding_images("splits/ground_truth-cxl-full_images_body_bbox-seed-42-train-70-val-15-test-15/train")

#     dir = generate_split(
#         dataset="ground_truth/rohan-cxl/face_images", mode="openset", seed=43, reid_factor_test=10, reid_factor_val=10
#     )
#     dir = generate_split(dataset="ground_truth/rohan-cxl/face_images", mode="closedset", seed=42)

#     dir = generate_split(
#         dataset="ground_truth/cxl/full_images", mode="openset", seed=43, reid_factor_test=10, reid_factor_val=10
#     )
#     dir = generate_split(dataset="ground_truth/cxl/full_images", mode="closedset", seed=42)
