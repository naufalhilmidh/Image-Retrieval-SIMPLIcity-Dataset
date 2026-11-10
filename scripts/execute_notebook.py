"""Eksekusi notebook secara in-process dan rekam output aslinya.

``jupyter nbconvert --execute`` memerlukan kernel terpisah yang membuat file
koneksi dan mengubah DACL-nya. Operasi DACL itu diblokir sandbox Windows pada
sesi ini, sehingga kernel tidak pernah berhasil start.

Skrip ini menjalankan sel kode di dalam proses Python yang sama (tanpa kernel),
lalu menulis hasil eksekusi (stdout, tabel, dan gambar) kembali ke file
``.ipynb``. Hasilnya setara dengan notebook yang dijalankan di Jupyter untuk
keperluan pemeriksaan dan presentasi.

Jalankan:
    python scripts/execute_notebook.py
    python scripts/execute_notebook.py --notebook notebooks/klasifikasi_simplicity.ipynb
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def strip_magics(source: str) -> str:
    """Buang baris magic IPython (mis. ``%matplotlib inline``).

    Backend matplotlib sudah diset ke Agg di proses ini, sehingga magic tersebut
    tidak diperlukan dan tidak bisa dieksekusi di luar kernel IPython.
    """
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith(("%", "!")):
            continue
        lines.append(line)
    return "\n".join(lines)


class CellExecutor:
    """Jalankan sel kode dan ubah hasilnya menjadi output nbformat."""

    def __init__(self) -> None:
        self.namespace: Dict[str, Any] = {"__name__": "__main__"}
        self._current_outputs: List[Dict[str, Any]] = []
        self.namespace["display"] = self._display

    # ------------------------------------------------------------- display
    def _display(self, *objects: Any, **_: Any) -> None:
        """Pengganti ``display()`` milik IPython: dukung DataFrame & objek biasa."""
        for obj in objects:
            if hasattr(obj, "to_html"):
                data = {
                    "text/html": obj.to_html(),
                    "text/plain": repr(obj),
                }
            else:
                data = {"text/plain": repr(obj)}
            self._current_outputs.append(nbf.v4.new_output(
                "display_data", data=data, metadata={},
            ))

    # ------------------------------------------------------------ eksekusi
    def run_cell(self, source: str) -> List[Dict[str, Any]]:
        """Eksekusi satu sel; kembalikan daftar output nbformat."""
        self._current_outputs = []
        buffer = io.StringIO()
        error: str | None = None

        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                exec(compile(strip_magics(source), "<cell>", "exec"), self.namespace)
        except Exception:  # noqa: BLE001 - error dilaporkan ke notebook
            error = traceback.format_exc()

        teks = buffer.getvalue()
        if teks:
            self._current_outputs.insert(0, nbf.v4.new_output(
                "stream", name="stdout", text=teks,
            ))

        # Rekam setiap figure matplotlib yang masih terbuka
        for nomor in plt.get_fignums():
            fig = plt.figure(nomor)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
            buf.seek(0)
            self._current_outputs.append(nbf.v4.new_output(
                "display_data",
                data={"image/png": base64.b64encode(buf.read()).decode("ascii")},
                metadata={},
            ))
            plt.close(fig)
        plt.close("all")

        if error:
            baris = error.strip().splitlines()
            self._current_outputs.append(nbf.v4.new_output(
                "error",
                ename="CellExecutionError",
                evalue=baris[-1] if baris else "Error",
                traceback=baris,
            ))
        return self._current_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", default="notebooks/klasifikasi_simplicity.ipynb")
    parser.add_argument("--stop-on-error", action="store_true",
                        help="hentikan eksekusi pada sel pertama yang gagal")
    parser.add_argument("--no-output", action="store_true",
                        help="jangan simpan output ke file notebook")
    args = parser.parse_args()

    nb_path = Path(args.notebook)
    if not nb_path.is_absolute():
        nb_path = PROJECT_ROOT / nb_path
    if not nb_path.is_file():
        raise SystemExit(f"Notebook tidak ditemukan: {nb_path}")

    notebook = nbf.read(str(nb_path), as_version=4)
    executor = CellExecutor()

    total = len(notebook.cells)
    gagal: List[int] = []
    started = time.perf_counter()

    # Bekerja dari folder notebooks agar path relatif di notebook sesuai
    import os

    os.chdir(nb_path.parent)

    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        judul = (cell.source.strip().splitlines() or [""])[0][:64]
        print(f"[{i + 1:2d}/{total}] {judul}", flush=True)
        outputs = executor.run_cell(cell.source)
        cell["outputs"] = outputs
        cell["execution_count"] = i + 1
        for out in outputs:
            if out["output_type"] == "error":
                gagal.append(i + 1)
                print(f"        GAGAL: {out['evalue']}", flush=True)
                if args.stop_on_error:
                    raise SystemExit(f"Berhenti pada sel {i + 1}")

    elapsed = time.perf_counter() - started
    print(f"\nEksekusi selesai dalam {elapsed:.1f}s")

    if gagal:
        print(f"Sel gagal: {gagal}")
    else:
        print("Semua sel kode berhasil dijalankan.")

    if not args.no_output:
        nbf.write(notebook, str(nb_path))
        print(f"Notebook tersimpan dengan output: {nb_path}")

    if gagal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
