"""Konfigurasi terpusat proyek klasifikasi 10 kategori SIMPLIcity dataset.

Semua path, pemetaan label, dan hyperparameter default didefinisikan di sini
supaya skrip training, evaluasi, dan plotting memakai sumber kebenaran yang sama.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- path proyek
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATASET_ROOT: Path = PROJECT_ROOT / "SIMPLIcity Dataset"
OUTPUT_ROOT: Path = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR: Path = OUTPUT_ROOT / "checkpoints"
FIGURE_DIR: Path = OUTPUT_ROOT / "figures"
LOG_DIR: Path = OUTPUT_ROOT / "logs"
REPORT_DIR: Path = OUTPUT_ROOT / "reports"
PREDICTION_DIR: Path = OUTPUT_ROOT / "predictions"
BENCHMARK_DIR: Path = OUTPUT_ROOT / "benchmark"

# ------------------------------------------------------------------- kategori
# Urutan kelas mengikuti rentang kode file pada dataset (kode // 100).
CLASS_NAMES: List[str] = [
    "people",     # 0-99
    "beach",      # 100-199
    "building",   # 200-299
    "bus",        # 300-399
    "dinosaur",   # 400-499
    "elephant",   # 500-599
    "flower",     # 600-699
    "horse",      # 700-799
    "mountain",   # 800-899
    "food",       # 900-999
]
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: Dict[int, str] = {i: name for name, i in CLASS_TO_INDEX.items()}

# rentang kode global per kategori: (inklusif bawah, eksklusif atas)
CLASS_CODE_RANGE: Dict[str, Tuple[int, int]] = {
    name: (i * 100, (i + 1) * 100) for i, name in enumerate(CLASS_NAMES)
}

# ------------------------------------------------- pembagian split via kode
# Berdasarkan instruksi tugas: train = x01-x49, validation = x50-x79, test = x80-x99
SPLIT_OFFSET_RANGE: Dict[str, Tuple[int, int]] = {
    "train": (0, 50),        # x00-x49 (instruksi: x01-x49; dataset menyertakan x00)
    "validation": (50, 80),  # x50-x79
    "test": (80, 100),       # x80-x99
}

# Jumlah gambar per kelas untuk validasi integritas data.
# Catatan: instruksi menyebut train "x01-x49", tetapi file di dataset memakai
# offset x00-x49 (50 berkas per kelas). Nilai di bawah mengikuti data nyata;
# pemindaian tetap menolak offset di luar rentang split masing-masing.
EXPECTED_PER_CLASS: Dict[str, int] = {"train": 50, "validation": 30, "test": 20}
EXPECTED_TOTAL: Dict[str, int] = {
    split: EXPECTED_PER_CLASS[split] * NUM_CLASSES for split in EXPECTED_PER_CLASS
}

IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ImageNet statistics (dipakai untuk model transfer learning)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def canonical_label(file_stem: str) -> Tuple[int, int]:
    """Ubah nama file (tanpa ekstensi) menjadi (index_kelas, offset_dalam_kelas).

    Contoh: ``"347"`` -> ``(3, 47)`` artinya kelas ``bus`` (300-399) offset 47.
    """
    code = int(file_stem)
    if not 0 <= code <= 999:
        raise ValueError(f"Kode gambar di luar rentang 0-999: {file_stem!r}")
    return code // 100, code % 100


def split_of_offset(offset: int) -> str:
    """Tentukan nama split dari offset kode (x01-x49 / x50-x79 / x80-x99)."""
    for split, (low, high) in SPLIT_OFFSET_RANGE.items():
        if low <= offset < high:
            return split
    raise ValueError(f"Offset {offset} tidak termasuk split mana pun")


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameter default untuk eksperimen utama."""

    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0          # Windows: 0 paling aman saat memakai DataLoader
    epochs: int = 30
    head_epochs: int = 8          # fase 1: backbone dibekukan
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3
    patience: int = 8
    mixed_precision: bool = True
    class_names: Tuple[str, ...] = field(default=tuple(CLASS_NAMES))


def ensure_output_dirs() -> None:
    """Buat seluruh folder output bila belum ada."""
    for path in (
        CHECKPOINT_DIR,
        FIGURE_DIR,
        LOG_DIR,
        REPORT_DIR,
        PREDICTION_DIR,
        BENCHMARK_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
