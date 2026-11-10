"""Bootstrap environment: arahkan semua cache unduhan ke dalam folder proyek.

Sandbox Windows pada sesi ini hanya memberi akses tulis di dalam workspace,
sehingga cache default (``%USERPROFILE%\\.cache\\torch`` dan
``~/.keras/models``) gagal diakses. ``configure_cache_dirs()`` wajib dipanggil
SEBELUM ``import torch`` / ``import tensorflow``.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / "outputs" / ".cache"


def configure_cache_dirs() -> Path:
    """Set TORCH_HOME/KERAS_HOME/MPLCONFIGDIR ke folder yang bisa ditulis."""
    torch_home = CACHE_ROOT / "torch"
    keras_home = CACHE_ROOT / "keras"
    mpl_dir = CACHE_ROOT / "matplotlib"
    for path in (torch_home, keras_home, mpl_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("KERAS_HOME", str(keras_home))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    return CACHE_ROOT
