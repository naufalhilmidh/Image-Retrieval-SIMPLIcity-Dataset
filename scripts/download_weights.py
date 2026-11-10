"""Pra-unduh bobot pretrained dengan resume + timeout (tahan jaringan lambat).

``torch.hub.load_state_dict_from_url`` memakai satu koneksi tanpa timeout yang
memadai, sehingga unduhan bisa menggantung tanpa batas pada jaringan yang tidak
stabil. Skrip ini mengunduh per potongan (HTTP Range), mencoba ulang, dan
melanjutkan dari byte terakhir bila koneksi terputus.

Jalankan:
    python scripts/download_weights.py
    python scripts/download_weights.py --models resnet34 efficientnet_b0
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env_setup import configure_cache_dirs  # noqa: E402

configure_cache_dirs()

from src.config import PROJECT_ROOT as ROOT  # noqa: E402

CHECKPOINT_CACHE = ROOT / "outputs" / ".cache" / "torch" / "hub" / "checkpoints"
BASE = "https://download.pytorch.org/models"

WEIGHT_URLS: Dict[str, str] = {
    "resnet18": f"{BASE}/resnet18-f37072fd.pth",
    "resnet34": f"{BASE}/resnet34-b627a593.pth",
    "resnet50": f"{BASE}/resnet50-11ad3fa6.pth",
    "efficientnet_b0": f"{BASE}/efficientnet_b0_rwightman-7f5810bc.pth",
    "mobilenet_v3_large": f"{BASE}/mobilenet_v3_large-8738ca79.pth",
    "densenet121": f"{BASE}/densenet121-a639ec97.pth",
    "convnext_tiny": f"{BASE}/convnext_tiny-983f1562.pth",
}

CHUNK = 1 << 20  # 1 MiB per permintaan


def remote_size(url: str, timeout: int = 30) -> int | None:
    """Ambil Content-Length tanpa mengunduh isi."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def download_resumable(
    url: str,
    dest: Path,
    retries: int = 6,
    timeout: int = 30,
) -> Path:
    """Unduh ``url`` ke ``dest`` dengan resume per chunk dan backoff."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        size = remote_size(url)
        if size is not None and dest.stat().st_size == size:
            print(f"  sudah lengkap: {dest.name} ({size / 1e6:.1f} MB)")
            return dest

    expected = remote_size(url)
    print(f"  target: {dest.name}"
          + (f" ({expected / 1e6:.1f} MB)" if expected else " (ukuran tidak diketahui)"))

    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expected is not None and have >= expected:
            break
        try:
            headers = {"Range": f"bytes={have}-{have + CHUNK - 1}"} if have else {}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data:
                break
            with dest.open("ab") as fh:
                fh.write(data)
            have = dest.stat().st_size
            if expected:
                print(f"    {have / 1e6:6.1f}/{expected / 1e6:.1f} MB "
                      f"({100 * have / expected:5.1f}%)", flush=True)
        except (urllib.error.URLError, TimeoutError, OSError, ConnectionError) as exc:
            print(f"    percobaan {attempt}/{retries} gagal: {type(exc).__name__}")
            time.sleep(min(2 ** attempt, 20))

    size = dest.stat().st_size if dest.exists() else 0
    if expected is not None and size != expected:
        raise RuntimeError(
            f"Unduhan tidak lengkap: {dest.name} {size}/{expected} byte. "
            f"Jalankan ulang skrip ini untuk melanjutkan."
        )
    print(f"  selesai: {dest.name} ({size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=None,
                        help=f"subset dari: {sorted(WEIGHT_URLS)}")
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--clean-partials", action="store_true",
                        help="hapus file .partial sisa unduhan yang gagal")
    args = parser.parse_args()

    if args.clean_partials:
        for partial in CHECKPOINT_CACHE.glob("*.partial"):
            partial.unlink()
            print(f"dihapus: {partial.name}")

    targets: List[str] = args.models or list(WEIGHT_URLS)
    unknown = [m for m in targets if m not in WEIGHT_URLS]
    if unknown:
        raise SystemExit(f"Model tidak dikenal: {unknown}. Pilihan: {sorted(WEIGHT_URLS)}")

    CHUNK_DIR = CHECKPOINT_CACHE
    print(f"Cache: {CHUNK_DIR}")
    for name in targets:
        print(f"\n[{name}]")
        url = WEIGHT_URLS[name]
        dest = CHUNK_DIR / Path(url).name
        try:
            download_resumable(url, dest, retries=args.retries)
        except RuntimeError as exc:
            print(f"  GAGAL: {exc}")

    # bersihkan file .partial yang tertinggal
    for partial in CHUNK_DIR.glob("*.partial"):
        try:
            partial.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
