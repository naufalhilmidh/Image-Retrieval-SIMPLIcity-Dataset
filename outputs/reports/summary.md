# Ringkasan Hasil — Klasifikasi 10 Kategori SIMPLIcity

Dataset: 500 train / 300 validation / 200 test (10 kelas × 50/30/20).
Label diturunkan dari kode nama file (`int(stem) // 100`).

## 1. Hasil Training 2 Fase

| Run | Model | Pretrained | Epoch | Best Epoch | Best Val Loss | Best Val Acc | Test Acc | Test Macro-F1 | Test Top-3 | Waktu (s) |
|-----|-------|:----------:|------:|-----------:|--------------:|-------------:|---------:|--------------:|-----------:|----------:|
| efficientnet_b0_2phase_s42 | efficientnet_b0 | True | 30 | 25 | 0.5907 | 0.9733 | 0.9700 | 0.9698 | 0.9950 | 190 |
| mobilenet_v3_large_2phase_s2024 | mobilenet_v3_large | True | 30 | 30 | 0.5727 | 0.9833 | 0.9750 | 0.9747 | 1.0000 | 185 |
| mobilenet_v3_large_2phase_s42 | mobilenet_v3_large | True | 30 | 22 | 0.5771 | 0.9867 | 0.9950 | 0.9950 | 1.0000 | 159 |
| mobilenet_v3_large_2phase_s7 | mobilenet_v3_large | True | 30 | 28 | 0.5756 | 0.9867 | 0.9900 | 0.9900 | 1.0000 | 194 |
| resnet34_2phase_s2024 | resnet34 | True | 30 | 23 | 0.6030 | 0.9633 | 0.9650 | 0.9649 | 0.9900 | 191 |
| resnet34_2phase_s42 | resnet34 | True | 30 | 25 | 0.5977 | 0.9667 | 0.9650 | 0.9651 | 0.9900 | 206 |
| resnet34_2phase_s7 | resnet34 | True | 28 | 20 | 0.5912 | 0.9733 | 0.9650 | 0.9646 | 0.9950 | 219 |

**Model terbaik keseluruhan:** `mobilenet_v3_large_2phase_s42` (mobilenet_v3_large) — test accuracy 0.9950, macro-F1 0.9950.

## 2. Detail per Kelas — `mobilenet_v3_large_2phase_s42`

| Kelas | Precision | Recall | F1 | Support |
|-------|----------:|-------:|---:|--------:|
| people | 0.9524 | 1.0000 | 0.9756 | 20 |
| beach | 1.0000 | 0.9500 | 0.9744 | 20 |
| building | 1.0000 | 1.0000 | 1.0000 | 20 |
| bus | 1.0000 | 1.0000 | 1.0000 | 20 |
| dinosaur | 1.0000 | 1.0000 | 1.0000 | 20 |
| elephant | 1.0000 | 1.0000 | 1.0000 | 20 |
| flower | 1.0000 | 1.0000 | 1.0000 | 20 |
| horse | 1.0000 | 1.0000 | 1.0000 | 20 |
| mountain | 1.0000 | 1.0000 | 1.0000 | 20 |
| food | 1.0000 | 1.0000 | 1.0000 | 20 |

Confusion matrix: `outputs/figures/mobilenet_v3_large_2phase_s42_confusion_matrix_test.png`

## 3. Aset Gambar

| Jenis | Path |
|-------|------|
| Grafik | `outputs/figures/efficientnet_b0_2phase_s42_confusion_matrix_test.png` |
| Grafik | `outputs/figures/efficientnet_b0_2phase_s42_gradient_descent.png` |
| Grafik | `outputs/figures/efficientnet_b0_2phase_s42_gradient_descent_loss.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s2024_confusion_matrix_test.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s2024_gradient_descent.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s2024_gradient_descent_loss.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s42_confusion_matrix_test.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s42_gradient_descent.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s42_gradient_descent_loss.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s7_confusion_matrix_test.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s7_gradient_descent.png` |
| Grafik | `outputs/figures/mobilenet_v3_large_2phase_s7_gradient_descent_loss.png` |
| Grafik | `outputs/figures/resnet34_2phase_s2024_confusion_matrix_test.png` |
| Grafik | `outputs/figures/resnet34_2phase_s2024_gradient_descent.png` |
| Grafik | `outputs/figures/resnet34_2phase_s2024_gradient_descent_loss.png` |
| Grafik | `outputs/figures/resnet34_2phase_s42_confusion_matrix_test.png` |
| Grafik | `outputs/figures/resnet34_2phase_s42_gradient_descent.png` |
| Grafik | `outputs/figures/resnet34_2phase_s42_gradient_descent_loss.png` |
| Grafik | `outputs/figures/resnet34_2phase_s7_confusion_matrix_test.png` |
| Grafik | `outputs/figures/resnet34_2phase_s7_gradient_descent.png` |
| Grafik | `outputs/figures/resnet34_2phase_s7_gradient_descent_loss.png` |

## 4. Catatan Metodologis

- Test set dievaluasi sekali setelah hyperparameter dibekukan; tidak dipakai untuk memilih model.
- Best checkpoint dipilih berdasarkan `val_loss` terendah, bukan akurasi, karena akurasi validasi cepat jenuh pada dataset yang relatif mudah.
- SIMPLIcity berisi gambar objek tunggal dengan latar bersih. Akurasi tinggi tidak otomatis berarti model siap untuk foto dunia nyata.
