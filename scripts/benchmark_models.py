"""Seleksi arsitektur: linear probe (frozen backbone) pada ketiga split.

Prosedur ini memberi perbandingan bobot pretrained yang adil dan cepat:
fitur ImageNet diekstrak sekali, lalu classifier linear dilatih dengan
LogisticRegression di atas fitur tersebut. Model dengan akurasi validasi
tertinggi dipilih untuk dilatih penuh (fine-tuning) pada tahap berikutnya.

Jalankan:
    python scripts/benchmark_models.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import BENCHMARK_DIR, CLASS_NAMES, NUM_CLASSES  # noqa: E402
from src.data import build_dataloader, set_seed  # noqa: E402
from src.models import build_feature_extractor, count_parameters  # noqa: E402

CANDIDATES: List[str] = [
    "custom_cnn",
    "resnet18",
    "resnet34",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3_large",
    "densenet121",
    "convnext_tiny",
]
PROBE_CANDIDATES = [c for c in CANDIDATES if c != "custom_cnn"]


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Ambil representasi penultimate untuk seluruh batch."""
    model.eval()
    feats, labels = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        feats.append(out.detach().float().cpu().flatten(1).numpy())
        labels.append(targets.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def probe_architecture(
    name: str,
    device: torch.device,
    image_size: int,
    batch_size: int,
    max_iter: int,
) -> Dict[str, object]:
    """Latih classifier linear di atas fitur beku satu backbone."""
    started = time.perf_counter()
    model, feat_dim = build_feature_extractor(name, pretrained=True)
    model = model.to(device)

    loaders = {}
    for split in ("train", "validation", "test"):
        loaders[split], _ = build_dataloader(
            split,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=0,
            train=False,
            shuffle=False,
        )

    x_train, y_train = extract_features(model, loaders["train"], device)
    x_val, y_val = extract_features(model, loaders["validation"], device)
    x_test, y_test = extract_features(model, loaders["test"], device)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=1.0),
    )
    clf.fit(x_train, y_train)
    val_acc = accuracy_score(y_val, clf.predict(x_val))
    test_acc = accuracy_score(y_test, clf.predict(x_test))
    elapsed = time.perf_counter() - started

    result = {
        "model": name,
        "feature_dim": int(x_train.shape[1]),
        "params_total": count_parameters(model, trainable_only=False),
        "val_accuracy": round(float(val_acc), 4),
        "test_accuracy": round(float(test_acc), 4),
        "seconds": round(elapsed, 1),
    }
    print(f"  {name:22s} val={val_acc:.4f}  test={test_acc:.4f}  "
          f"dim={result['feature_dim']}  {elapsed:5.1f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Kelas: {', '.join(CLASS_NAMES)}")

    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    targets = args.models or PROBE_CANDIDATES

    results: List[Dict[str, object]] = []
    print("\n== Linear probe (backbone beku, classifier logistic regression) ==")
    for name in targets:
        try:
            results.append(
                probe_architecture(
                    name,
                    device=device,
                    image_size=args.image_size,
                    batch_size=args.batch_size,
                    max_iter=args.max_iter,
                )
            )
        except Exception as exc:  # noqa: BLE001 - lanjutkan kandidat berikutnya
            print(f"  {name:22s} GAGAL: {type(exc).__name__}: {exc}", flush=True)

    results.sort(key=lambda r: r["val_accuracy"], reverse=True)
    payload = {
        "protocol": "frozen-backbone linear probe, 224px, ImageNet weights",
        "image_size": args.image_size,
        "splits": {"train": 50, "validation": 30, "test": 20},
        "results": results,
        "best": results[0]["model"] if results else None,
    }
    out_path = BENCHMARK_DIR / "architecture_probe.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n== Peringkat akurasi validasi ==")
    for rank, r in enumerate(results, start=1):
        print(f"{rank:2d}. {r['model']:22s} val={r['val_accuracy']:.4f}  "
              f"test={r['test_accuracy']:.4f}  params={r['params_total']:,}")
    print(f"\nTerbaik: {payload['best']}")
    print(f"Disimpan: {out_path}")


if __name__ == "__main__":
    main()
