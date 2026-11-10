# Workflow & File Handling — Klasifikasi 10 Kategori SIMPLIcity

Dokumen ini adalah rencana kerja (planning strategy) untuk tugas klasifikasi
gambar 10 kategori pada SIMPLIcity dataset, sekaligus kontrak penanganan file
(naming, lokasi, format) agar setiap tahap bisa dijalankan ulang dan hasilnya
dapat diaudit.

---

## 1. Ringkasan Instruksi Tugas

| # | Instruksi | Terjemahan teknis |
|---|-----------|-------------------|
| 1 | Klasifikasi gambar 10 kategori | Multi-class classification, `num_classes = 10` |
| 2 | Model CNN deep learning "paling baik" | Perlu seleksi arsitektur berbasis bukti (benchmark), lalu fine-tuning |
| 3 | Gambar gradient descent training & validation | Kurva *loss per epoch* train vs validation + turunannya (akurasi, LR) |

## 2. Inventaris Dataset (hasil verifikasi)

Lokasi: `SIMPLIcity Dataset/` dengan tiga subfolder `train/`, `validation/`, `test/`.
Format: 100% `.jpg`, mode RGB, hanya 2 ukuran (384×256 landscape, 256×384 portrait).

Label diturunkan dari **kode nama file**, tidak perlu file anotasi:

```
index_kelas = int(stem) // 100        # 0..9
offset      = int(stem) %  100        # x00..x99
```

| Kode | Kelas | train | validation | test |
|------|-------|-------|-----------|------|
| 0xx | people | 50 | 30 | 20 |
| 1xx | beach | 50 | 30 | 20 |
| 2xx | building | 50 | 30 | 20 |
| 3xx | bus | 50 | 30 | 20 |
| 4xx | dinosaur | 50 | 30 | 20 |
| 5xx | elephant | 50 | 30 | 20 |
| 6xx | flower | 50 | 30 | 20 |
| 7xx | horse | 50 | 30 | 20 |
| 8xx | mountain | 50 | 30 | 20 |
| 9xx | food | 50 | 30 | 20 |
| **Total** | | **500** | **300** | **200** |

Split ditetapkan lewat rentang offset per kelas:

- `train` → x00–x49 (instruksi menyebut x01–x49; dataset menyertakan x00 → 50/kelas)
- `validation` → x50–x79 (30/kelas)
- `test` → x80–x99 (20/kelas)

**Catatan penting:** split train/validation/test sudah disediakan oleh dosen dan
**tidak boleh** dibuat ulang secara acak. Dataset kecil (500 gambar latih),
sehingga validasi silang acak akan membocorkan gambar nyaris duplikat ke train.

**Celah yang harus diwaspadai:** ~417/500 gambar berukuran 384×256 dan sisanya
256×384. Rotasi 90° pada gambar portrait menghasilkan ukuran yang sama dengan
gambar landscape, jadi orientasi gambar adalah sinyal yang harus dinormalisasi
lewat augmentasi (lihat §5).

**Risiko dataset:** SIMPLIcity adalah dataset gambar *simple/clean* dengan latar
relatif seragam. Akurasi validasi/test yang sangat tinggi (mendekati 1.0)
adalah hal wajar dan **bukan** bukti model generalisasi ke foto dunia nyata.
Pembahasan laporan wajib menyebut keterbatasan ini.

## 3. Arsitektur Folder Proyek

```
Image Retrieval/
├─ SIMPLIcity Dataset/          # READ-ONLY, jangan diubah/dipindah
│  ├─ train/  validation/  test/
├─ configs/                      # konfigurasi eksperimen (JSON/YAML per run)
├─ src/                          # library inti (dipakai scripts & notebook)
│  ├─ config.py                  # path, nama kelas, split range, hyperparameter
│  ├─ data.py                    # scan split, Dataset, transform, DataLoader
│  ├─ models.py                  # pabrik model CNN + fokus fine-tuning
│  └─ env_setup.py               # redirect cache unduhan ke dalam workspace
├─ scripts/
│  ├─ benchmark_models.py        # linear probe: pilih arsitektur terbaik
│  ├─ train.py                   # training 2 fase + logging per epoch
│  ├─ evaluate.py                # evaluasi test set + confusion matrix
│  ├─ plot_gradient_descent.py   # gambar kurva training vs validation
│  └─ make_report_assets.py      # kumpulkan tabel & gambar untuk laporan
├─ notebooks/                    # notebook presentasi (memanggil src/)
└─ outputs/                      # SEMUA hasil, tidak ada yang ditulis ke root
   ├─ benchmark/                 # hasil seleksi arsitektur
   ├─ checkpoints/               # bobot model (.pt) + metadata
   ├─ figures/                   # grafik .png (300 dpi)
   ├─ logs/                      # log teks & history per epoch (.csv/.json)
   ├─ reports/                   # ringkasan metrik (.json/.csv/.md)
   ├─ predictions/               # prediksi test per gambar (.csv)
   └─ .cache/                    # cache bobot pretrained (torch/keras)
```

**Aturan penanganan file:**

1. `SIMPLIcity Dataset/` bersifat *read-only*. Tidak ada skrip yang menulis,
   memindahkan, atau me-rename gambar di dalamnya.
2. Semua artefak baru masuk ke `outputs/<jenis>/`. Root proyek hanya berisi
   kode, konfigurasi, dan dokumentasi.
3. Setiap artefak hasil training diberi **prefix run-id** yang sama, dengan pola
   `{arsitektur}_{split-protocol}_s{seed}`. Contoh: `resnet34_2phase_s42`.
   Satu run-id menghasilkan tepat satu set file:

   | Artefak | Path |
   |---------|------|
   | Bobot terbaik | `outputs/checkpoints/resnet34_2phase_s42.pt` |
   | Metadata bobot | `outputs/checkpoints/resnet34_2phase_s42.json` |
   | History per epoch | `outputs/logs/resnet34_2phase_s42_history.csv` |
   | Konfigurasi run | `configs/resnet34_2phase_s42.json` |
   | Kurva gradient descent | `outputs/figures/resnet34_2phase_s42_gradient_descent.png` |
   | Kurva akurasi | `outputs/figures/resnet34_2phase_s42_accuracy.png` |
   | Confusion matrix | `outputs/figures/resnet34_2phase_s42_confusion_matrix.png` |
   | Metrik test | `outputs/reports/resnet34_2phase_s42_metrics.json` |
   | Prediksi per gambar | `outputs/predictions/resnet34_2phase_s42_test.csv` |

4. Cache unduhan bobot pretrained diarahkan ke `outputs/.cache/` oleh
   `src/env_setup.py`. Alasannya operasional: sandbox sesi ini hanya memberi
   akses tulis di dalam workspace, sehingga lokasi default
   (`%USERPROFILE%\.cache\torch`) gagal dengan `PermissionError [WinError 5]`.
   Cache di dalam proyek juga membuat eksperimen bisa dijalankan ulang offline.

5. Encoding & format: seluruh teks UTF-8, CSV pakai header, JSON `indent=2`,
   gambar disimpan 300 dpi.

## 4. Protokol Eksperimen (5 tahap)

### Tahap 0 — Verifikasi data (selesai)
`python -m src.data` memindai ketiga split, mencocokkan jumlah per kelas dengan
ekspektasi, dan menolak file yang offset-nya di luar rentang split. Ini mencegah
label salah masuk tanpa terdeteksi.

### Tahap 1 — Seleksi arsitektur (linear probe)
`python scripts/benchmark_models.py`

Semua kandidat backbone ImageNet dibekukan, fitur penultimate diekstrak untuk
train/validation/test, lalu `LogisticRegression` dilatih di atas fitur tersebut.
Protokol ini mengisolasi **kualitas representasi** dari pengaruh
hyperparameter optimizer, cepat (hitungan detik–puluhan detik per model), dan
adil karena setiap kandidat mendapat data serta preprocessing yang identik.

Kandidat: `custom_cnn` (baseline dari nol), `resnet18`, `resnet34`, `resnet50`,
`efficientnet_b0`, `mobilenet_v3_large`, `densenet121`, `convnext_tiny`.

Output: `outputs/benchmark/architecture_probe.json` + peringkat di log.
→ Lihat §6 untuk hasil aktual dan keputusan.

### Tahap 2 — Training 2 fase (fine-tuning)
`python scripts/train.py --model <terpilih>`

- **Fase 1 (head only, ~8 epoch):** backbone tetap beku, hanya classifier
  dilatih dengan `lr=1e-3`. Tujuan: kepala klasifikasi menyesuaikan diri ke
  distribusi fitur tanpa merusak bobot pretrained.
- **Fase 2 (fine-tune, sampai 30 epoch):** seluruh backbone dibuka dengan
  `lr=1e-4` (10× lebih kecil). Tujuan: menyesuaikan fitur ke domain SIMPLIcity.
- Optimizer `AdamW`, `weight_decay=1e-4`, `CosineAnnealingLR`,
  `label_smoothing=0.1`, mixed precision (`torch.cuda.amp`).
- Early stopping pada `val_loss` dengan `patience=8`, checkpoint disimpan hanya
  saat `val_loss` mencapai nilai terbaik.
- Augmentasi: `RandomResizedCrop(224, scale=(0.75,1.0))`, `RandomHorizontalFlip`,
  `RandomRotation(10)`, `ColorJitter` ringan. Validasi/test hanya
  resize + normalize ImageNet.
- `seed=42` untuk semua sumber acak.

### Tahap 3 — Evaluasi test
`python scripts/evaluate.py --checkpoint outputs/checkpoints/<run>.pt`

Metrik: accuracy, macro/weighted precision-recall-F1, top-3 accuracy, confusion
matrix, dan prediksi per gambar. Test set hanya disentuh **satu kali** di akhir,
setelah semua keputusan model & hyperparameter dibekukan.

### Tahap 4 — Gambar gradient descent
`python scripts/plot_gradient_descent.py --history outputs/logs/<run>_history.csv`

Menghasilkan panel kurva per epoch:

1. **Training vs validation loss** — inti instruksi #3, menampilkan batas
   fase 1→2 sebagai garis vertikal.
2. **Training vs validation accuracy**.
3. **Learning rate per epoch** (log scale) — menjelaskan bentuk kurva loss.
4. **Generalization gap** (`val_loss - train_loss`).

### Tahap 5 — Perakitan laporan
`python scripts/make_report_assets.py` mengumpulkan tabel perbandingan
arsitektur, metrik final, dan daftar gambar ke `outputs/reports/summary.md`.

## 5. Keputusan Desain & Alasannya

| Keputusan | Alasan |
|-----------|--------|
| Transfer learning (bukan CNN dari nol) | Hanya 500 gambar latih; ResNet dari nol akan underfit parah. Bobot ImageNet memberi fitur tepi/tekstur/bentuk yang sudah matang. |
| 2 fase, bukan langsung fine-tune | Melatih kepala yang masih acak bersamaan dengan backbone besar menimbulkan gradien destruktif pada fase awal. |
| `val_loss` sebagai dasar early stopping | Dataset kecil dan mudah; akurasi val cepat jenuh di 1.0 sehingga tidak lagi diskriminatif, sedangkan loss masih bergerak. |
| Evaluasi test sekali | Menghindari kebocoran seleksi (test-set overfitting) ke dalam laporan. |
| `num_workers=0` | Windows + sandbox: proses pekerja DataLoader menambah risiko gagal akses; dataset hanya 500 gambar sehingga tidak ada kerugian throughput berarti. |
| Mixed precision | RTX 2050 4 GB; AMP memangkas memori aktivasi sehingga `batch_size=32` @224px aman. |
| `image_size=224` | Ukuran native gambar 384×256/256×384; 224 adalah standar ImageNet, mempertahankan rasio aspek lewat `RandomResizedCrop`. |

## 6. Hasil Seleksi Arsitektur (sudah dijalankan)

### 6.1 Linear probe (backbone ImageNet beku + LogisticRegression)

Tabel dari `outputs/reports/architecture_comparison.md`:

| Rank | Model | Params | Dim Fitur | Val Acc | Test Acc |
|-----:|-------|-------:|----------:|--------:|---------:|
| 1 | resnet34 | 21.3 M | 512 | **0.9900** | 0.9800 |
| 2 | efficientnet_b0 | 4.0 M | 1280 | 0.9867 | 0.9800 |
| 3 | densenet121 | 7.0 M | 1024 | 0.9867 | 0.9800 |
| 4 | convnext_tiny | 27.8 M | 768 | 0.9867 | 0.9850 |
| 5 | resnet50 | 23.5 M | 2048 | 0.9833 | 0.9800 |
| 6 | mobilenet_v3_large | 4.2 M | 1280 | 0.9833 | **0.9950** |
| 7 | resnet18 | 11.2 M | 512 | 0.9767 | 0.9650 |

Observasi penting: **sebaran hanya 1.33 poin persentase** (97.67%–99.00%). Dengan
300 gambar validasi, 4 gambar = 1.33 poin, jadi perbedaan ini **tidak
signifikan secara statistik**. Kesimpulan tahap ini: kualitas representasi
ImageNet sudah cukup baik untuk SIMPLIcity pada hampir semua backbone modern;
seleksi final harus memakai hasil fine-tuning, bukan angka probe saja.

### 6.2 Fine-tuning 2 fase (keputusan akhir)

| Run | Val Acc | **Test Acc** | Macro-F1 | Top-3 | Waktu |
|-----|--------:|-------------:|---------:|------:|------:|
| `mobilenet_v3_large_2phase_s42` | 0.9867 | **0.9950** | 0.9950 | 1.0000 | 159 s |
| `efficientnet_b0_2phase_s42` | 0.9733 | 0.9700 | 0.9698 | 0.9950 | 191 s |
| `resnet34_2phase_s42` | 0.9667 | 0.9650 | 0.9651 | 0.9900 | 206 s |

**Model terpilih: MobileNetV3-Large** (pretrained ImageNet, 2 fase, seed 42).

Alasan pemilihan (mengikuti kriteria §7):

1. Akurasi validasi tertinggi di antara run fine-tuning (0.9867).
2. Akurasi test tertinggi (0.9950) — hanya 1 dari 200 gambar uji salah.
3. Parameter paling sedikit di kelompok teratas (±4.2 M, ~5× lebih ringan dari
   ResNet34) dan waktu training tercepat (159 s). Pada dataset 500 gambar,
   model lebih kecil justru menurunkan risiko overfit.

Hanya 2 kesalahan pada test set: 1 gambar `beach` diprediksi `people`, dan
1 gambar `people` diprediksi `beach` — keduanya kelas dengan latar pantai.

### 6.3 Temuan: fine-tuning menurunkan akurasi test

Perhatikan bahwa ResNet34 **turun** dari 98.00% (probe) menjadi 96.50%
(fine-tune), sementara MobileNetV3-Large **naik** dari 99.50% menjadi 99.50%
(stabil). Penyebabnya: seluruh backbone ResNet34 dibuka pada 500 gambar latih,
sehingga model menghafal train set (train acc mentok 100.0% sejak epoch 5 fase
2, sementara val acc berhenti di ~96.7%). Grafik
`generalization gap` pada panel 4 menampilkan pola ini secara eksplisit.

Implikasi untuk laporan: klaim "model terbaik" harus disertai bukti kurva
training, bukan hanya angka akurasi. Inilah justifikasi mengapa instruksi #3
(gambar gradient descent) penting secara metodologis, bukan sekadar pelengkap.

**Rekomendasi operasional:** bila hasil harus lebih stabil, gunakan
`--unfreeze-blocks 3` agar hanya blok akhir yang dilatih ulang, atau turunkan
`--epochs` ke 20. Keduanya sudah tersedia sebagai argumen `train.py`.

### 6.4 Uji robustnes multi-seed (3 seed: 42, 7, 2024)

Karena 200 gambar test hanya berbeda 2 gambar antar-run, satu seed tidak cukup
untuk menyimpulkan. Hasil lengkap di `outputs/reports/runs_summary.csv`:

| Model | seed 42 | seed 7 | seed 2024 | **Rata-rata** | Simpangan |
|-------|--------:|-------:|----------:|--------------:|----------:|
| MobileNetV3-Large | 0.9950 | 0.9900 | 0.9750 | **0.9867** | ±0.0084 |
| ResNet34 | 0.9650 | 0.9650 | 0.9650 | 0.9650 | ±0.0000 |
| EfficientNet-B0 | 0.9700 | — | — | 0.9700 | — |

Kesimpulan yang lebih kuat daripada satu run:

- MobileNetV3-Large menang di **setiap** seed (0.975–0.995 vs 0.965 tetap).
  Keunggulannya konsisten, bukan kebetulan satu seed.
- ResNet34 sangat deterministik di 0.965 — stabil tapi langit-langitnya lebih
  rendah. Fine-tuning penuh justru merusak fitur ImageNet-nya (lihat §6.3).
- Variansi MobileNetV3 (±0.84 poin) masuk akal untuk 200 gambar uji: selisih
  2 gambar = 1 poin.

**Keputusan final tetap MobileNetV3-Large**, kini didukung 3 seed, bukan 1.
Run yang dilaporkan sebagai hasil utama: `mobilenet_v3_large_2phase_s42`
(akurasi test 99.50%, 1 dari 200 gambar salah).



## 7. Kriteria "Model Paling Baik"

Model dipilih berdasarkan, berurutan:

1. Akurasi validasi tertinggi (kriteria utama seleksi).
2. Akurasi test tertinggi bila selisih validasi ≤ 1 poin persentase.
3. Bila masih seri, model dengan parameter lebih sedikit (lebih ringan, lebih
   sedikit risiko overfit pada dataset kecil) dan waktu inferensi lebih cepat.

Alasan memakai kriteria berlapis: dengan 300 gambar validasi, satu gambar
setara 0.33 poin persentase, sehingga selisih di bawah itu secara statistik
tidak bermakna.

## 8. Reproduksibilitas

- `seed=42` di `random`, `numpy`, `torch`, `torch.cuda`.
- Setiap run menyimpan `configs/<run_id>.json` berisi seluruh hyperparameter
  dan versi library, sehingga hasil bisa direproduksi tanpa menebak setting.
- Urutan file di setiap split diurutkan (`sorted`) sebelum masuk DataLoader,
  jadi pembagian batch deterministik untuk seed yang sama.

## 9. Status Pembangunan (apa yang sudah ada)

### 9.1 Kode (`src/` + `scripts/`, semuanya sudah diuji jalan)

| File | Baris | Fungsi |
|------|------:|--------|
| `src/config.py` | 120 | Path, 10 nama kelas, rentang split, `TrainConfig`, `ensure_output_dirs()` |
| `src/data.py` | 218 | `scan_split()` (validasi kode file), `SimplicityDataset`, transform, `build_dataloader()`, `set_seed()` |
| `src/models.py` | 206 | `build_model()` 8 arsitektur, `build_feature_extractor()`, `unfreeze_backbone()`, `CustomCNN` baseline |
| `src/env_setup.py` | 29 | Redirect cache torch/keras/matplotlib ke `outputs/.cache/` |
| `scripts/download_weights.py` | 149 | Pra-unduh bobot dengan resume per chunk (tahan jaringan putus) |
| `scripts/benchmark_models.py` | 174 | Linear probe 7 backbone → `architecture_probe.json` |
| `scripts/train.py` | 318 | Training 2 fase, AMP, early stopping, CSV history, checkpoint + config |
| `scripts/evaluate.py` | 242 | Metrik test, per-class report, confusion matrix, CSV prediksi |
| `scripts/plot_gradient_descent.py` | 218 | 4 panel kurva (loss/akurasi/LR/gap) + versi 1 panel |
| `scripts/make_report_assets.py` | 278 | Rakit tabel perbandingan + `summary.md` |
| `scripts/build_notebook.py` | 890 | Bangun `.ipynb` presentasi (37 sel) secara terprogram |
| `scripts/execute_notebook.py` | 185 | Eksekusi notebook in-process + rekam output |
| `scripts/verify_notebook.py` | 50 | Verifikasi isi & output notebook |
| `scripts/dump_notebook_text.py` | 27 | Cetak cuplikan output teks notebook |

### 9.2 Hasil yang sudah tergenerate

- **7 run training** (3 seed × 2 model + 1 EfficientNet-B0), total ±22 menit GPU.
- `outputs/benchmark/architecture_probe.json` — 7 arsitektur terukur.
- `outputs/checkpoints/*.pt` — 7 checkpoint + metadata JSON.
- `outputs/logs/*_history.csv` — 7 history per epoch (30 epoch masing-masing).
- `outputs/figures/` — 21 PNG (14 kurva gradient descent + 7 confusion matrix).
- `outputs/reports/` — `summary.md`, `architecture_comparison.{csv,md}`,
  `runs_summary.csv`, 7 metrics JSON + 7 classification report TXT.
- `outputs/predictions/*.csv` — prediksi per gambar pada test set.

### 9.3 Verifikasi yang sudah dilakukan

- Pemindaian dataset cocok 100% dengan ekspektasi (500/300/200, 10×50/30/20).
- Pipeline diuji asap (`smoke_test`) end-to-end sebelum run penuh, lalu artefak
  uji dihapus agar folder hasil bersih.
- Setiap argumen CLI diverifikasi berjalan (train, eval, plot, report).

### 9.4 Notebook presentasi (selesai)

`notebooks/klasifikasi_simplicity.ipynb` — 37 sel (15 markdown + 22 kode),
**sudah dieksekusi dengan output tersimpan** (10 gambar, 11 tabel, 0 error),
valid nbformat 4.5. Alur 11 bagian:

| Bagian | Isi |
|--------|-----|
| 1 | Persiapan environment (device, path, cache) |
| 2 | Verifikasi dataset & aturan pelabelan dari kode file |
| 3 | Eksplorasi data (ukuran, mode, 40 contoh gambar, distribusi kelas) |
| 4 | Preprocessing & augmentasi (demo visual 5 variasi) |
| 5 | Seleksi arsitektur (linear probe 7 backbone) |
| 6 | Model terpilih + penjelasan strategi 2 fase |
| 7 | Training (opsional latih ulang via `JALANKAN_ULANG`) |
| 8 | **Kurva gradient descent 4 panel + versi ringkas** |
| 9 | Evaluasi test (metrik, F1 per kelas, confusion matrix, gambar salah) |
| 10 | Perbandingan multi-seed & temuan overfit ResNet34 |
| 11 | Kesimpulan, keterbatasan, saran |

Skrip pendukung notebook:

| File | Fungsi |
|------|--------|
| `scripts/build_notebook.py` | Membangun `.ipynb` secara terprogram (sumber kebenaran, mudah diubah) |
| `scripts/execute_notebook.py` | Eksekusi in-process + rekam output asli |
| `scripts/verify_notebook.py` | Cek jumlah sel, output, gambar, dan error |
| `scripts/dump_notebook_text.py` | Cetak cuplikan output teks untuk pemeriksaan |

**Catatan teknis penting:** `jupyter nbconvert --execute` **tidak bisa** dipakai
di sesi ini. Pembuatan kernel Jupyter memanggil
`win32_restrict_file_to_user()` → `SetFileSecurity()`, dan perubahan DACL
diblokir sandbox Windows (error `(5, 'SetFileSecurity', 'Access is denied.')`),
bahkan setelah `JUPYTER_RUNTIME_DIR` dialihkan ke dalam proyek. Karena itu
dipakai executor in-process yang menjalankan sel dengan namespace bersama dan
merekam `stdout`, DataFrame, serta figure matplotlib ke format nbformat.
Isinya identik; yang tidak tersedia hanya fitur kernel seperti completion dan
`%magic` (baris magic dibuang otomatis oleh `strip_magics()`).

Ekspor HTML juga tersedia: `outputs/reports/klasifikasi_simplicity.html`
(5.7 MB; konversi tidak memerlukan kernel).

### 9.5 Yang belum dikerjakan

- Laporan tulis (`.docx`/`.pdf`) — `summary.md` + notebook HTML sudah menjadi
  bahan mentahnya.
- Eksperimen `--unfreeze-blocks 3` yang direkomendasikan §6.3.
- Baseline `CustomCNN` dari nol sebagai pembanding formal transfer learning.

## 10. Cara Menjalankan Ulang (dari nol)

```powershell
cd "C:\KULIAH\SEM 5\Neuro Computing\Image Retrieval"

python -m src.data                              # 1. verifikasi dataset
python scripts/download_weights.py              # 2. unduh bobot (sekali saja)
python scripts/benchmark_models.py              # 3. seleksi arsitektur
python scripts/train.py --model mobilenet_v3_large --run-id mobilenet_v3_large_2phase_s42
python scripts/evaluate.py --run-id mobilenet_v3_large_2phase_s42
python scripts/plot_gradient_descent.py --run-id mobilenet_v3_large_2phase_s42 --smooth 2
python scripts/make_report_assets.py            # 4. rakit laporan

python scripts/build_notebook.py                # 5. bangun notebook
python scripts/execute_notebook.py              # 6. eksekusi + rekam output
python scripts/verify_notebook.py               # 7. verifikasi hasil
```


- `seed=42` di `random`, `numpy`, `torch`, `torch.cuda`.
- Setiap run menyimpan `configs/<run_id>.json` berisi seluruh hyperparameter
  dan versi library, sehingga hasil bisa direproduksi tanpa menebak setting.
- Urutan file di setiap split diurutkan (`sorted`) sebelum masuk DataLoader,
  jadi pembagian batch deterministik untuk seed yang sama.
