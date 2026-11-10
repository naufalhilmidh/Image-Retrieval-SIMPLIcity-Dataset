"""Gambar kurva gradient descent: training vs validation.

Inti instruksi #3 — memvisualisasikan proses penurunan loss selama training
dan membandingkannya dengan validation, untuk membaca apakah model konvergen,
underfit, atau overfit.

Contoh:
    python scripts/plot_gradient_descent.py --run-id resnet34_2phase_s42
    python scripts/plot_gradient_descent.py --run-id resnet34_2phase_s42 --smooth 3

Menghasilkan:
    outputs/figures/<run_id>_gradient_descent.png   panel loss, akurasi, lr, gap
    outputs/figures/<run_id>_gradient_descent_loss.png  satu panel loss saja
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import FIGURE_DIR, LOG_DIR, PROJECT_ROOT as ROOT, ensure_output_dirs  # noqa: E402


def load_history(path: Path) -> List[Dict[str, float]]:
    """Baca CSV history per epoch menjadi list of dict (nilai numerik)."""
    if not path.is_file():
        raise SystemExit(f"History tidak ditemukan: {path}")
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            row: Dict[str, float] = {}
            for key, value in raw.items():
                if key == "phase":
                    row["phase"] = value  # type: ignore[assignment]
                    continue
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    row[key] = 0.0
            rows.append(row)
    if not rows:
        raise SystemExit(f"History kosong: {path}")
    return rows


def smooth(values: Sequence[float], window: int) -> List[float]:
    """Moving average sederhana untuk meredam noise antar epoch."""
    if window <= 1:
        return list(values)
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def phase_boundary(history: List[Dict[str, float]]) -> float | None:
    """Nomor epoch tempat fase head -> finetune berganti."""
    for row in history:
        if row.get("phase") == "finetune":
            return row["epoch"] - 0.5
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history", default=None, help="path CSV (override --run-id)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="window moving average untuk kurva mentah (default: 1 = tanpa smoothing)")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    ensure_output_dirs()
    hist_path = Path(args.history) if args.history else LOG_DIR / f"{args.run_id}_history.csv"
    if args.history and not Path(args.history).is_absolute():
        hist_path = ROOT / args.history

    history = load_history(hist_path)
    epochs = [int(r["epoch"]) for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_acc = [r["train_acc"] for r in history]
    val_acc = [r["val_acc"] for r in history]
    lrs = [r["lr"] for r in history]

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])
    boundary = phase_boundary(history)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": args.dpi,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    def annotate_phases(ax) -> None:
        if boundary is not None:
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(boundary, ax.get_ylim()[1], " fine-tune →", va="top", ha="left",
                    fontsize=8, color="gray")

    # --- panel 1: loss ---------------------------------------------------
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, color="#1f77b4", marker="o", markersize=3,
            linewidth=1.4, alpha=0.45, label="train (mentah)")
    ax.plot(epochs, val_loss, color="#d62728", marker="o", markersize=3,
            linewidth=1.4, alpha=0.45, label="validation (mentah)")
    if args.smooth > 1:
        ax.plot(epochs, smooth(train_loss, args.smooth), color="#1f77b4",
                linewidth=2.4, label=f"train (MA-{args.smooth})")
        ax.plot(epochs, smooth(val_loss, args.smooth), color="#d62728",
                linewidth=2.4, label=f"validation (MA-{args.smooth})")
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=90, marker="*",
               color="#ff7f0e", zorder=5, edgecolor="black", linewidth=0.6,
               label=f"val_loss terbaik ({val_loss[best_idx]:.4f})")
    annotate_phases(ax)
    ax.set_title("Gradient Descent — Loss per Epoch (Train vs Validation)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(fontsize=8, loc="upper right")

    # --- panel 2: akurasi -------------------------------------------------
    ax = axes[0, 1]
    ax.plot(epochs, [v * 100 for v in train_acc], color="#1f77b4", marker="o",
            markersize=3, linewidth=1.6, label="train accuracy")
    ax.plot(epochs, [v * 100 for v in val_acc], color="#d62728", marker="s",
            markersize=3, linewidth=1.6, label="validation accuracy")
    ax.scatter([epochs[best_idx]], [val_acc[best_idx] * 100], s=90, marker="*",
               color="#ff7f0e", zorder=5, edgecolor="black", linewidth=0.6,
               label=f"val_acc@{epochs[best_idx]} ({val_acc[best_idx] * 100:.2f}%)")
    annotate_phases(ax)
    ax.set_title("Akurasi per Epoch", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Akurasi (%)")
    ax.set_ylim(max(0, min(min(train_acc), min(val_acc)) * 100 - 10), 101)
    ax.legend(fontsize=8, loc="lower right")

    # --- panel 3: learning rate -------------------------------------------
    ax = axes[1, 0]
    ax.plot(epochs, lrs, color="#2ca02c", marker="o", markersize=3, linewidth=1.6)
    annotate_phases(ax)
    ax.set_title("Jadwal Learning Rate (Cosine Annealing)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_yscale("log")

    # --- panel 4: generalization gap --------------------------------------
    ax = axes[1, 1]
    gap = [v - t for t, v in zip(train_loss, val_loss)]
    colors = ["#d62728" if g > 0 else "#2ca02c" for g in gap]
    ax.bar(epochs, gap, color=colors, alpha=0.75, width=0.7)
    ax.axhline(0, color="black", linewidth=0.9)
    annotate_phases(ax)
    ax.set_title("Generalization Gap (val_loss − train_loss)", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Selisih Loss")
    ax.text(0.02, 0.96, "positif = val lebih buruk dari train (indikasi overfit)",
            transform=ax.transAxes, fontsize=7.5, va="top", style="italic", color="#555")

    fig.suptitle(f"Kurva Training — {args.run_id}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_main = FIGURE_DIR / f"{args.run_id}_gradient_descent.png"
    fig.savefig(out_main, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # --- versi ringkas: satu panel loss -----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(epochs, train_loss, color="#1f77b4", marker="o", markersize=4,
            linewidth=1.8, label="Training Loss")
    ax.plot(epochs, val_loss, color="#d62728", marker="s", markersize=4,
            linewidth=1.8, label="Validation Loss")
    ax.scatter([epochs[best_idx]], [val_loss[best_idx]], s=140, marker="*",
               color="#ff7f0e", zorder=5, edgecolor="black", linewidth=0.7,
               label=f"Best val_loss = {val_loss[best_idx]:.4f} (epoch {epochs[best_idx]})")
    annotate_phases(ax)
    ax.set_title(f"Gradient Descent — {args.run_id}", fontweight="bold", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    fig.tight_layout()
    out_simple = FIGURE_DIR / f"{args.run_id}_gradient_descent_loss.png"
    fig.savefig(out_simple, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Epoch dijalankan   : {len(history)}")
    print(f"Train loss akhir   : {train_loss[-1]:.4f}")
    print(f"Val loss akhir     : {val_loss[-1]:.4f}")
    print(f"Best val_loss      : {val_loss[best_idx]:.4f} (epoch {epochs[best_idx]})")
    print(f"Best val_acc       : {val_acc[best_idx]:.4f}")
    print(f"Gambar (4 panel)   : {out_main}")
    print(f"Gambar (loss saja) : {out_simple}")


if __name__ == "__main__":
    main()
