"""Evaluasi model terlatih pada split test + artefak laporan.

Contoh:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/resnet34_2phase_s42.pt
    python scripts/evaluate.py --run-id resnet34_2phase_s42 --split test

Menghasilkan:
    outputs/reports/<run_id>_metrics.json
    outputs/reports/<run_id>_classification_report.txt
    outputs/figures/<run_id>_confusion_matrix.png
    outputs/predictions/<run_id>_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLASS_NAMES,
    FIGURE_DIR,
    NUM_CLASSES,
    PREDICTION_DIR,
    PROJECT_ROOT as ROOT,
    REPORT_DIR,
    ensure_output_dirs,
)
from src.data import build_dataloader  # noqa: E402
from src.models import build_model  # noqa: E402


@torch.no_grad()
def predict_all(model: nn.Module, loader, device: torch.device):
    """Kumpulkan probabilitas, label, dan path untuk seluruh split."""
    model.eval()
    probs_all, labels_all, paths_all = [], [], []
    for batch in loader:
        if len(batch) == 3:
            images, targets, paths = batch
        else:
            images, targets = batch
            paths = [""] * len(targets)
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        probs_all.append(probs.cpu().numpy())
        labels_all.append(targets.numpy())
        paths_all.extend(paths)
    return np.concatenate(probs_all), np.concatenate(labels_all), paths_all


def plot_confusion(
    cm: np.ndarray,
    run_id: str,
    split: str,
    out_path: Path,
) -> None:
    """Simpan confusion matrix (nilai absolut + normalisasi baris)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sums, 1), dtype=float)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (jumlah) — {split}")
    axes[0].set_xlabel("Prediksi")
    axes[0].set_ylabel("Label sebenarnya")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=False,
                vmin=0, vmax=1, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1])
    axes[1].set_title(f"Confusion Matrix (normalisasi baris) — {split}")
    axes[1].set_xlabel("Prediksi")
    axes[1].set_ylabel("Label sebenarnya")

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)

    fig.suptitle(f"{run_id}", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=None, help="path file .pt")
    parser.add_argument("--run-id", default=None, help="alternatif: cari di outputs/checkpoints")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ensure_output_dirs()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
    elif args.run_id:
        ckpt_path = CHECKPOINT_DIR / f"{args.run_id}.pt"
    else:
        raise SystemExit("Sebutkan --checkpoint atau --run-id")
    if not ckpt_path.is_file():
        raise SystemExit(f"Checkpoint tidak ditemukan: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(ckpt_path, map_location=device)
    run_id = payload.get("run_id", ckpt_path.stem)
    model_name = payload["model_name"]
    image_size = payload.get("image_size", 224)
    class_names: List[str] = payload.get("class_names", CLASS_NAMES)

    print(f"Checkpoint : {ckpt_path.name}")
    print(f"Run ID     : {run_id}")
    print(f"Model      : {model_name} @ {image_size}px")
    print(f"Epoch      : {payload.get('epoch')}  val_loss={payload.get('val_loss'):.4f}  "
          f"val_acc={payload.get('val_acc'):.4f}")

    model = build_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model = model.to(device)

    loader, dataset = build_dataloader(
        args.split, image_size=image_size, batch_size=args.batch_size,
        train=False, return_path=True,
    )
    probs, y_true, paths = predict_all(model, loader, device)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    top3 = float(top_k_accuracy_score(y_true, probs, k=3, labels=list(range(NUM_CLASSES))))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))

    metrics: Dict[str, object] = {
        "run_id": run_id,
        "model": model_name,
        "split": args.split,
        "image_size": image_size,
        "n_samples": int(len(y_true)),
        "accuracy": round(float(acc), 4),
        "top3_accuracy": round(top3, 4),
        "macro_precision": round(float(macro[0]), 4),
        "macro_recall": round(float(macro[1]), 4),
        "macro_f1": round(float(macro[2]), 4),
        "weighted_precision": round(float(weighted[0]), 4),
        "weighted_recall": round(float(weighted[1]), 4),
        "weighted_f1": round(float(weighted[2]), 4),
        "per_class": {
            class_names[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(NUM_CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
    }

    metrics_path = REPORT_DIR / f"{run_id}_metrics_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report_txt = classification_report(
        y_true, y_pred, labels=list(range(NUM_CLASSES)),
        target_names=class_names, digits=4, zero_division=0,
    )
    report_path = REPORT_DIR / f"{run_id}_classification_report_{args.split}.txt"
    report_path.write_text(
        f"Run: {run_id}\nModel: {model_name}\nSplit: {args.split}\n"
        f"Akurasi: {acc:.4f}  Top-3: {top3:.4f}\n\n" + report_txt,
        encoding="utf-8",
    )

    fig_path = FIGURE_DIR / f"{run_id}_confusion_matrix_{args.split}.png"
    plot_confusion(cm, run_id, args.split, fig_path)

    pred_path = PREDICTION_DIR / f"{run_id}_{args.split}.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "true_label", "true_name", "pred_label",
                         "pred_name", "confidence", "correct"])
        for path, t, p in zip(paths, y_true, probs):
            pred_idx = int(p.argmax())
            writer.writerow([
                Path(path).name, int(t), class_names[int(t)], pred_idx,
                class_names[pred_idx], f"{float(p[pred_idx]):.6f}", int(t == pred_idx),
            ])

    print("\n" + report_txt)
    print(f"Akurasi   : {acc:.4f}")
    print(f"Top-3     : {top3:.4f}")
    print(f"Macro F1  : {macro[2]:.4f}")
    print(f"\nMetrik    : {metrics_path}")
    print(f"Report    : {report_path}")
    print(f"Confusion : {fig_path}")
    print(f"Prediksi  : {pred_path}")


if __name__ == "__main__":
    main()
