from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DatasetName = Literal["bcss", "luad"]
SplitName = Literal["train", "val", "test"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = {
    "bcss": ("tumor", "stroma", "lymphocyte", "necrosis"),
    "luad": ("tumor_epithelial", "necrosis", "lymphocyte", "tumor_stroma"),
}

BCSS_PALETTE = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (153, 0, 255),
    4: (255, 255, 255),
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    train_dir: Path
    val_img_dir: Path
    val_mask_dir: Path
    test_img_dir: Path
    test_mask_dir: Path


def resolve_dataset_paths(data_root: str | Path, dataset: DatasetName) -> DatasetPaths:
    """Resolve dataset folders, accepting both train/ and training/ conventions."""
    data_root = Path(data_root)
    folder_name = {"bcss": "BCSS-WSSS", "luad": "LUAD-HistoSeg"}[dataset]
    root = data_root / folder_name
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    train_dir = root / "training"
    if not train_dir.exists():
        train_dir = root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Expected train/training folder under {root}")

    paths = DatasetPaths(
        root=root,
        train_dir=train_dir,
        val_img_dir=root / "val" / "img",
        val_mask_dir=root / "val" / "mask",
        test_img_dir=root / "test" / "img",
        test_mask_dir=root / "test" / "mask",
    )
    for path in (paths.val_img_dir, paths.val_mask_dir, paths.test_img_dir, paths.test_mask_dir):
        if not path.exists():
            raise FileNotFoundError(f"Expected folder not found: {path}")
    return paths


def parse_image_level_label(filename: str | Path, dataset: DatasetName) -> torch.Tensor:
    """Parse patch-level multi-hot labels from WSSS filenames.

    BCSS uses contiguous labels like [1100].
    LUAD commonly uses separated labels like [1, 0, 0, 1].
    """
    name = Path(filename).name
    match = re.search(r"\[([^\]]+)\]", name)
    if not match:
        raise ValueError(f"No [label] block found in filename: {name}")

    label_str = match.group(1)
    if dataset == "bcss":
        digits = [int(ch) for ch in label_str if ch in "01"]
    elif dataset == "luad":
        digits = [int(x) for x in re.findall(r"[01]", label_str)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if len(digits) != 4:
        raise ValueError(f"Expected 4 labels for {dataset}, got {digits} from {name}")
    return torch.tensor(digits, dtype=torch.float32)


def load_preprocessor_stats(path: str | Path | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if path is None:
        return IMAGENET_MEAN, IMAGENET_STD
    path = Path(path)
    config_path = path / "preprocessor_config.json" if path.is_dir() else path.parent / "preprocessor_config.json"
    if not config_path.exists():
        return IMAGENET_MEAN, IMAGENET_STD
    config = json.loads(config_path.read_text(encoding="utf-8"))
    mean = tuple(float(x) for x in config.get("image_mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in config.get("image_std", IMAGENET_STD))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Expected 3-channel image_mean/image_std in {config_path}")
    return mean, std


def pil_to_normalized_tensor(
    image: Image.Image,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def load_mask(path: str | Path, ignore_background: bool = True, ignore_index: int = 255) -> torch.Tensor:
    mask = torch.from_numpy(np.asarray(Image.open(path), dtype=np.uint8).copy()).long()
    if ignore_background:
        mask = mask.clone()
        mask[mask == 4] = ignore_index
    return mask


class ImageLevelDataset(Dataset):
    """Training dataset with image-level labels only."""

    def __init__(
        self,
        data_root: str | Path,
        dataset: DatasetName = "bcss",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.dataset = dataset
        self.paths = resolve_dataset_paths(data_root, dataset)
        self.images = sorted(self.paths.train_dir.glob("*.png"))
        self.transform = transform or pil_to_normalized_tensor
        if not self.images:
            raise FileNotFoundError(f"No training PNG files found in {self.paths.train_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        return {
            "image": self.transform(image),
            "label": parse_image_level_label(path.name, self.dataset),
            "name": path.name,
            "height": height,
            "width": width,
        }


class SegmentationDataset(Dataset):
    """Validation/test dataset with dense masks."""

    def __init__(
        self,
        data_root: str | Path,
        dataset: DatasetName = "bcss",
        split: Literal["val", "test"] = "val",
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
        ignore_background: bool = True,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.paths = resolve_dataset_paths(data_root, dataset)
        self.image_dir = self.paths.val_img_dir if split == "val" else self.paths.test_img_dir
        self.mask_dir = self.paths.val_mask_dir if split == "val" else self.paths.test_mask_dir
        self.images = sorted(self.image_dir.glob("*.png"))
        self.transform = transform or pil_to_normalized_tensor
        self.ignore_background = ignore_background
        if not self.images:
            raise FileNotFoundError(f"No {split} PNG files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.images[index]
        mask_path = self.mask_dir / path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing for {path.name}: {mask_path}")
        image = Image.open(path).convert("RGB")
        width, height = image.size
        return {
            "image": self.transform(image),
            "mask": load_mask(mask_path, ignore_background=self.ignore_background),
            "name": path.name,
            "height": height,
            "width": width,
        }


def make_image_level_loader(
    data_root: str | Path,
    dataset: DatasetName = "bcss",
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        ImageLevelDataset(data_root=data_root, dataset=dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def make_segmentation_loader(
    data_root: str | Path,
    dataset: DatasetName = "bcss",
    split: Literal["val", "test"] = "val",
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        SegmentationDataset(data_root=data_root, dataset=dataset, split=split),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
