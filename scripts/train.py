"""Training CNN 2 fase (head-only lalu fine-tune) untuk 10 kategori SIMPLIcity.

Contoh:
    python scripts/train.py --model resnet34
    python scripts/train.py --model resnet34 --epochs 30 --head-epochs 8 --seed 42

Menghasilkan:
    outputs/checkpoints/<run_id>.pt        bobot terbaik (val_loss terendah)
    outputs/checkpoints/<run_id>.json      metadata checkpoint
    outputs/logs/<run_id>_history.csv      metrik per epoch
    configs/<run_id>.json                  konfigurasi lengkap run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import (  # noqa: E402
    CHECKPOINT_DIR,
    CLASS_NAMES,
    LOG_DIR,
    NUM_CLASSES,
    PROJECT_ROOT as ROOT,
    TrainConfig,
    ensure_output_dirs,
)
from src.data import build_dataloader, set_seed  # noqa: E402
from src.models import build_model, count_parameters, unfreeze_backbone  # noqa: E402

HISTORY_FIELDS = [
    "epoch",
    "phase",
    "lr",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "val_f1_macro",
    "epoch_seconds",
]


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Satu epoch penuh. optimizer=None berarti mode evaluasi.

    Return: (rata-rata loss, akurasi, semua label, semua prediksi)
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    use_amp = scaler is not None and device.type == "cuda"

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == targets).sum().item()
        total_samples += batch_size
        all_labels.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy, np.array(all_labels), np.array(all_preds)


def main() -> None:
    cfg = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="resnet34", help="nama arsitektur di src/models.py")
    parser.add_argument("--run-id", default=None, help="default: {model}_2phase_s{seed}")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--head-epochs", type=int, default=cfg.head_epochs)
    parser.add_argument("--batch-size", type=int, default=cfg.batch_size)
    parser.add_argument("--image-size", type=int, default=cfg.image_size)
    parser.add_argument("--lr-head", type=float, default=cfg.lr_head)
    parser.add_argument("--lr-finetune", type=float, default=cfg.lr_finetune)
    parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay)
    parser.add_argument("--label-smoothing", type=float, default=cfg.label_smoothing)
    parser.add_argument("--dropout", type=float, default=cfg.dropout)
    parser.add_argument("--patience", type=int, default=cfg.patience)
    parser.add_argument("--augment", default="standard", choices=["standard", "light", "none"])
    parser.add_argument("--seed", type=int, default=cfg.seed)
    parser.add_argument("--unfreeze-blocks", type=int, default=None,
                        help="batasi fine-tuning ke N blok terakhir (default: semua)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="latih dari bobot acak (untuk baseline pembanding)")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    pretrained = not args.no_pretrained
    run_id = args.run_id or f"{args.model}_2phase_s{args.seed}"
    ensure_output_dirs()
    (ROOT / "configs").mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run ID     : {run_id}")
    print(f"Device     : {device}")
    print(f"Model      : {args.model} (pretrained={pretrained})")

    train_loader, train_ds = build_dataloader(
        "train", image_size=args.image_size, batch_size=args.batch_size,
        augment=args.augment, seed=args.seed,
    )
    val_loader, val_ds = build_dataloader(
        "validation", image_size=args.image_size, batch_size=args.batch_size,
        train=False, seed=args.seed,
    )
    print(f"Train      : {len(train_ds)} gambar, {len(train_loader)} batch")
    print(f"Validation : {len(val_ds)} gambar, {len(val_loader)} batch")

    model = build_model(args.model, num_classes=NUM_CLASSES,
                        pretrained=pretrained, dropout=args.dropout).to(device)
    print(f"Parameter  : {count_parameters(model, trainable_only=False):,} total, "
          f"{count_parameters(model, trainable_only=True):,} trainable (fase 1)")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = None if (args.no_amp or device.type != "cuda") else torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict[str, object]] = []
    started_all = time.perf_counter()

    # ------------------------------------------------------------- fase 1
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.head_epochs, 1),
    )
    print(f"\n== Fase 1: head only ({args.head_epochs} epoch, lr={args.lr_head}) ==")

    for epoch in range(1, args.head_epochs + 1):
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device,
                                          optimizer=optimizer, scaler=scaler)
        va_loss, va_acc, va_y, va_p = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
        record = {
            "epoch": len(history) + 1, "phase": "head", "lr": round(lr_now, 8),
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss": round(va_loss, 6), "val_acc": round(va_acc, 6),
            "val_f1_macro": round(float(val_f1), 6), "epoch_seconds": round(elapsed, 2),
        }
        history.append(record)
        print(f"[head {epoch:2d}/{args.head_epochs}] lr={lr_now:.2e} "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  acc {tr_acc:.4f}/{va_acc:.4f}  "
              f"({elapsed:.1f}s)", flush=True)

        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, len(history)

    # ------------------------------------------------------------- fase 2
    print(f"\n== Fase 2: fine-tune ({args.epochs - args.head_epochs} epoch, "
          f"lr={args.lr_finetune}) ==")
    unfreeze_backbone(model, last_blocks=args.unfreeze_blocks)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_finetune, weight_decay=args.weight_decay,
    )
    remaining = max(args.epochs - args.head_epochs, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)

    for epoch in range(1, remaining + 1):
        t0 = time.perf_counter()
        lr_now = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, device,
                                          optimizer=optimizer, scaler=scaler)
        va_loss, va_acc, va_y, va_p = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_f1 = f1_score(va_y, va_p, average="macro", zero_division=0)
        record = {
            "epoch": len(history) + 1, "phase": "finetune", "lr": round(lr_now, 8),
            "train_loss": round(tr_loss, 6), "train_acc": round(tr_acc, 6),
            "val_loss": round(va_loss, 6), "val_acc": round(va_acc, 6),
            "val_f1_macro": round(float(val_f1), 6), "epoch_seconds": round(elapsed, 2),
        }
        history.append(record)
        print(f"[ft   {epoch:2d}/{remaining}] lr={lr_now:.2e} "
              f"loss {tr_loss:.4f}/{va_loss:.4f}  acc {tr_acc:.4f}/{va_acc:.4f}  "
              f"({elapsed:.1f}s)", flush=True)

        if va_loss < best_val_loss - 1e-6:
            best_val_loss, best_epoch = va_loss, len(history)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_name": args.model,
                    "state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "epoch": len(history),
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "image_size": args.image_size,
                    "run_id": run_id,
                },
                CHECKPOINT_DIR / f"{run_id}.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping pada epoch {len(history)} "
                      f"(tidak ada perbaikan val_loss {args.patience} epoch)")
                break

    total_seconds = time.perf_counter() - started_all

    # ------------------------------------------------------------- artefak
    hist_path = LOG_DIR / f"{run_id}_history.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)

    config_payload = {
        "run_id": run_id,
        "model": args.model,
        "pretrained": pretrained,
        "epochs_requested": args.epochs,
        "head_epochs": args.head_epochs,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 6),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr_head": args.lr_head,
        "lr_finetune": args.lr_finetune,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "augment": args.augment,
        "patience": args.patience,
        "unfreeze_blocks": args.unfreeze_blocks,
        "seed": args.seed,
        "amp": scaler is not None,
        "device": str(device),
        "train_seconds": round(total_seconds, 1),
        "splits": {"train": len(train_ds), "validation": len(val_ds)},
        "history_csv": str(hist_path.relative_to(ROOT)),
    }
    (ROOT / "configs" / f"{run_id}.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )
    (CHECKPOINT_DIR / f"{run_id}.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    print(f"\nSelesai dalam {total_seconds:.1f}s")
    print(f"Best epoch {best_epoch}  val_loss {best_val_loss:.4f}")
    print(f"Checkpoint : outputs/checkpoints/{run_id}.pt")
    print(f"History    : outputs/logs/{run_id}_history.csv")
    print(f"Config     : configs/{run_id}.json")


if __name__ == "__main__":
    main()
