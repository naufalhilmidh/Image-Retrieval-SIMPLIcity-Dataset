"""Cetak cuplikan output teks notebook untuk pemeriksaan isi."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
nb = nbf.read(str(PROJECT_ROOT / "notebooks" / "klasifikasi_simplicity.ipynb"), as_version=4)

kode = [c for c in nb.cells if c.cell_type == "code"]
for cell in kode:
    teks = []
    for o in cell.get("outputs") or []:
        if o.get("output_type") == "stream":
            teks.append(o.get("text", ""))
    if not teks:
        continue
    judul = cell.source.strip().splitlines()[0][:70]
    isi = "".join(teks).strip()
    if not isi:
        continue
    print("=" * 76)
    print("SEL:", judul)
    print("-" * 76)
    print(isi[:1400])
    print()
