"""Verifikasi isi notebook hasil eksekusi (output, gambar, tabel, error)."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB = PROJECT_ROOT / "notebooks" / "klasifikasi_simplicity.ipynb"

nb = nbf.read(str(NB), as_version=4)
kode = [c for c in nb.cells if c.cell_type == "code"]
md = [c for c in nb.cells if c.cell_type == "markdown"]

gambar = tabel = teks = error = kosong = 0
for cell in kode:
    outs = cell.get("outputs") or []
    if not outs:
        kosong += 1
    for o in outs:
        tipe = o.get("output_type")
        if tipe == "error":
            error += 1
        elif tipe == "stream":
            teks += 1
        data = o.get("data") or {}
        if "image/png" in data:
            gambar += 1
        if "text/html" in data:
            tabel += 1

ukuran = NB.stat().st_size / 1024
print(f"File            : {NB.name}")
print(f"Ukuran          : {ukuran:.0f} KB")
print(f"Sel markdown    : {len(md)}")
print(f"Sel kode        : {len(kode)}")
print(f"Sel tanpa output: {kosong}")
print(f"Output gambar   : {gambar}")
print(f"Output tabel    : {tabel}")
print(f"Output teks     : {teks}")
print(f"Output error    : {error}")

if error:
    print("\nSel yang error:")
    for i, cell in enumerate(kode):
        if any(o.get("output_type") == "error" for o in cell.get("outputs") or []):
            print(f"  sel {i}: {cell.source.splitlines()[0][:70]}")

sys.exit(1 if error else 0)
