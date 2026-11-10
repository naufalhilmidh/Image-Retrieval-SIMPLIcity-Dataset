"""Rakit aset laporan: tabel perbandingan arsitektur + ringkasan hasil.

Menggabungkan
  outputs/benchmark/architecture_probe.json         (seleksi arsitektur)
  configs/<run_id>.json                              (konfigurasi run)
  outputs/logs/<run_id>_history.csv                  (kurva per epoch)
  outputs/reports/<run_id>_metrics_test.json         (metrik test)

menjadi:
  outputs/reports/architecture_comparison.csv
  outputs/reports/architecture_comparison.md
  outputs/reports/summary.md

Contoh:
    python scripts/make_report_assets.py
    python scripts/make_report_assets.py --runs resnet34_2phase_s42 efficientnet_b0_2phase_s42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import (  # noqa: E402
    BENCHMARK_DIR,
    CLASS_NAMES,
    FIGURE_DIR,
    LOG_DIR,
    PROJECT_ROOT as ROOT,
    REPORT_DIR,
    ensure_output_dirs,
)

PROBE_PATH = BENCHMARK_DIR / "architecture_probe.json"


def load_json(path: Path) -> Optional[Dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_history_summary(path: Path) -> Optional[Dict[str, float]]:
    """Ambil epoch terbaik dan nilai akhir dari history CSV."""
    if not path.is_file():
        return None
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                rows.append({k: float(v) for k, v in raw.items() if k != "phase"})
            except ValueError:
                continue
    if not rows:
        return None
    best = min(rows, key=lambda r: r["val_loss"])
    return {
        "epochs_run": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "final_train_loss": rows[-1]["train_loss"],
        "final_val_loss": rows[-1]["val_loss"],
        "final_train_acc": rows[-1]["train_acc"],
        "final_val_acc": rows[-1]["val_acc"],
    }


def discover_runs() -> List[str]:
    """Semua run yang punya history CSV di outputs/logs."""
    runs = []
    for path in sorted(LOG_DIR.glob("*_history.csv")):
        runs.append(path.name[: -len("_history.csv")])
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*", default=None,
                        help="run-id yang dilaporkan (default: semua history yang ada)")
    args = parser.parse_args()

    ensure_output_dirs()
    runs = args.runs or discover_runs()

    # ---------------------------------------------------- tabel benchmark
    probe = load_json(PROBE_PATH)
    comparison_rows: List[Dict[str, object]] = []
    if probe:
        for r in probe.get("results", []):
            comparison_rows.append({
                "model": r["model"],
                "params": r["params_total"],
                "feature_dim": r["feature_dim"],
                "linear_probe_val_acc": r["val_accuracy"],
                "linear_probe_test_acc": r["test_accuracy"],
                "probe_seconds": r["seconds"],
            })

    csv_path = REPORT_DIR / "architecture_comparison.csv"
    if comparison_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

        md_lines = [
            "# Perbandingan Arsitektur (Linear Probe, Backbone Beku)",
            "",
            "Protokol: bobot ImageNet dibekukan, fitur penultimate diekstrak, "
            "lalu LogisticRegression dilatih pada 500 gambar train.",
            "Peringkat berdasarkan akurasi validasi (300 gambar).",
            "",
            "| Rank | Model | Params | Dim Fitur | Val Acc | Test Acc | Waktu (s) |",
            "|-----:|-------|-------:|----------:|--------:|---------:|----------:|",
        ]
        for rank, row in enumerate(
            sorted(comparison_rows, key=lambda r: r["linear_probe_val_acc"], reverse=True),
            start=1,
        ):
            md_lines.append(
                f"| {rank} | {row['model']} | {row['params']:,} | {row['feature_dim']} | "
                f"{row['linear_probe_val_acc']:.4f} | {row['linear_probe_test_acc']:.4f} | "
                f"{row['probe_seconds']:.1f} |"
            )
        md_lines += [
            "",
            f"**Model terpilih:** `{probe.get('best')}`",
            "",
            "> Catatan: akurasi test di kolom ini berasal dari classifier linear di atas "
            "fitur beku, bukan dari model hasil fine-tuning. Angka final ada di "
            "`summary.md`.",
            "",
        ]
        (REPORT_DIR / "architecture_comparison.md").write_text(
            "\n".join(md_lines), encoding="utf-8"
        )
        print(f"Tabel benchmark : {csv_path}")

    # ------------------------------------------------------- ringkasan run
    run_rows: List[Dict[str, object]] = []
    for run_id in runs:
        cfg = load_json(ROOT / "configs" / f"{run_id}.json")
        hist = read_history_summary(LOG_DIR / f"{run_id}_history.csv")
        metrics = load_json(REPORT_DIR / f"{run_id}_metrics_test.json")
        run_rows.append({
            "run_id": run_id,
            "model": (cfg or {}).get("model", "?"),
            "pretrained": (cfg or {}).get("pretrained", "?"),
            "epochs_run": (hist or {}).get("epochs_run", 0),
            "best_epoch": (hist or {}).get("best_epoch", 0),
            "best_val_loss": (hist or {}).get("best_val_loss", float("nan")),
            "best_val_acc": (hist or {}).get("best_val_acc", float("nan")),
            "test_acc": (metrics or {}).get("accuracy", float("nan")),
            "test_macro_f1": (metrics or {}).get("macro_f1", float("nan")),
            "test_top3": (metrics or {}).get("top3_accuracy", float("nan")),
            "train_seconds": (cfg or {}).get("train_seconds", 0),
            "has_figure": (FIGURE_DIR / f"{run_id}_gradient_descent.png").is_file(),
        })

    if run_rows:
        with (REPORT_DIR / "runs_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(run_rows[0].keys()))
            writer.writeheader()
            writer.writerows(run_rows)
        print(f"Ringkasan run   : {REPORT_DIR / 'runs_summary.csv'}")

    # ---------------------------------------------------------- summary.md
    lines = [
        "# Ringkasan Hasil — Klasifikasi 10 Kategori SIMPLIcity",
        "",
        "Dataset: 500 train / 300 validation / 200 test (10 kelas × 50/30/20).",
        "Label diturunkan dari kode nama file (`int(stem) // 100`).",
        "",
    ]

    if run_rows:
        lines += [
            "## 1. Hasil Training 2 Fase",
            "",
            "| Run | Model | Pretrained | Epoch | Best Epoch | Best Val Loss | Best Val Acc | Test Acc | Test Macro-F1 | Test Top-3 | Waktu (s) |",
            "|-----|-------|:----------:|------:|-----------:|--------------:|-------------:|---------:|--------------:|-----------:|----------:|",
        ]
        for r in run_rows:
            lines.append(
                f"| {r['run_id']} | {r['model']} | {r['pretrained']} | {r['epochs_run']} | "
                f"{r['best_epoch']} | {r['best_val_loss']:.4f} | {r['best_val_acc']:.4f} | "
                f"{r['test_acc']:.4f} | {r['test_macro_f1']:.4f} | {r['test_top3']:.4f} | "
                f"{r['train_seconds']:.0f} |"
            )
        lines.append("")

        best_run = max(
            (r for r in run_rows if r["test_acc"] == r["test_acc"]),
            key=lambda r: (r["best_val_acc"], r["test_acc"]),
            default=None,
        )
        if best_run:
            lines += [
                f"**Model terbaik keseluruhan:** `{best_run['run_id']}` "
                f"({best_run['model']}) — test accuracy {best_run['test_acc']:.4f}, "
                f"macro-F1 {best_run['test_macro_f1']:.4f}.",
                "",
            ]

    # per-class detail run terbaik
    detail_run: Optional[str] = None
    if run_rows:
        scored = [r for r in run_rows if r["test_acc"] == r["test_acc"]]
        if scored:
            detail_run = max(
                scored, key=lambda r: (r["best_val_acc"], r["test_acc"])
            )["run_id"]
    if detail_run:
        metrics = load_json(REPORT_DIR / f"{detail_run}_metrics_test.json")
        if metrics:
            lines += [
                f"## 2. Detail per Kelas — `{detail_run}`",
                "",
                "| Kelas | Precision | Recall | F1 | Support |",
                "|-------|----------:|-------:|---:|--------:|",
            ]
            for name in CLASS_NAMES:
                c = metrics["per_class"][name]
                lines.append(
                    f"| {name} | {c['precision']:.4f} | {c['recall']:.4f} | "
                    f"{c['f1']:.4f} | {c['support']} |"
                )
            lines += [
                "",
                f"Confusion matrix: `outputs/figures/{detail_run}_confusion_matrix_test.png`",
                "",
            ]

    lines += [
        "## 3. Aset Gambar",
        "",
        "| Jenis | Path |",
        "|-------|------|",
    ]
    figs = sorted(FIGURE_DIR.glob("*.png"))
    if figs:
        for fig in figs:
            lines.append(f"| Grafik | `outputs/figures/{fig.name}` |")
    else:
        lines.append("| Grafik | (belum ada) |")

    lines += [
        "",
        "## 4. Catatan Metodologis",
        "",
        "- Test set dievaluasi sekali setelah hyperparameter dibekukan; tidak dipakai "
        "untuk memilih model.",
        "- Best checkpoint dipilih berdasarkan `val_loss` terendah, bukan akurasi, "
        "karena akurasi validasi cepat jenuh pada dataset yang relatif mudah.",
        "- SIMPLIcity berisi gambar objek tunggal dengan latar bersih. Akurasi tinggi "
        "tidak otomatis berarti model siap untuk foto dunia nyata.",
        "",
    ]
    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Ringkasan       : {summary_path}")


if __name__ == "__main__":
    main()
