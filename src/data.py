"""Pipeline data SIMPLIcity: indeks file, dataset PyTorch, transform, dan loader.

Label diturunkan dari nama file (kode // 100) sehingga tidak perlu anotasi
terpisah. Setiap split divalidasi terhadap rentang offset yang ditetapkan tugas.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import (
    CLASS_NAMES,
    DATASET_ROOT,
    EXPECTED_TOTAL,
    IMAGE_EXTENSIONS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    SPLIT_OFFSET_RANGE,
    canonical_label,
)


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    offset: int


def scan_split(split: str, root: Optional[Path] = None) -> List[Sample]:
    """Kumpulkan seluruh sampel satu split beserta label hasil parsing nama file."""
    root = root or DATASET_ROOT
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Folder split tidak ditemukan: {split_dir}")

    low, high = SPLIT_OFFSET_RANGE[split]
    samples: List[Sample] = []
    problems: List[str] = []

    for path in sorted(split_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            label, offset = canonical_label(path.stem)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not low <= offset < high:
            problems.append(
                f"{path.name} berada di luar rentang offset split {split} "
                f"({low}-{high - 1})"
            )
            continue
        samples.append(Sample(path=path, label=label, offset=offset))

    if problems:
        preview = "\n  - ".join(problems[:10])
        raise ValueError(f"Ditemukan {len(problems)} file bermasalah:\n  - {preview}")
    return samples


def summarize_split(split: str) -> Dict[str, object]:
    """Ringkasan jumlah sampel per kelas untuk satu split."""
    samples = scan_split(split)
    counts = Counter(s.label for s in samples)
    return {
        "split": split,
        "total": len(samples),
        "expected_total": EXPECTED_TOTAL.get(split),
        "per_class": {CLASS_NAMES[i]: counts.get(i, 0) for i in range(NUM_CLASSES)},
        "offsets": (min(s.offset for s in samples), max(s.offset for s in samples)),
    }


class SimplicityDataset(Dataset):
    """Dataset gambar SIMPLIcity dengan label dari kode nama file."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_path: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("Daftar sampel kosong")
        self.samples = list(samples)
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        target = sample.label
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_path:
            return image, target, str(sample.path)
        return image, target

    @property
    def labels(self) -> List[int]:
        return [s.label for s in self.samples]


# ------------------------------------------------------------------ transform
def build_transform(
    image_size: int = 224,
    train: bool = True,
    augment: str = "standard",
) -> Callable:
    """Bangun transform torchvision.

    augment: "standard" (flip + rotasi kecil + jitter ringan),
             "light" (hanya flip horizontal),
             "none" (resize + normalize saja).
    """
    if not train or augment == "none":
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    ops: List[Callable] = [
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
    ]
    if augment == "standard":
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, fill=0),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03),
        ]
    elif augment == "light":
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        raise ValueError(f"Mode augmentasi tidak dikenal: {augment!r}")

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_dataloader(
    split: str,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    train: Optional[bool] = None,
    augment: str = "standard",
    shuffle: Optional[bool] = None,
    seed: int = 42,
    return_path: bool = False,
) -> Tuple[DataLoader, SimplicityDataset]:
    """Buat DataLoader untuk satu split."""
    samples = scan_split(split)
    is_train = split == "train" if train is None else train
    dataset = SimplicityDataset(
        samples,
        transform=build_transform(image_size=image_size, train=is_train, augment=augment),
        return_path=return_path,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
    return loader, dataset


def set_seed(seed: int = 42) -> None:
    """Reproducibility untuk python, numpy, dan torch."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":  # pragma: no cover
    for name in ("train", "validation", "test"):
        info = summarize_split(name)
        print(f"[{info['split']}] total={info['total']} (harapan {info['expected_total']})")
        print("   ", info["per_class"])
