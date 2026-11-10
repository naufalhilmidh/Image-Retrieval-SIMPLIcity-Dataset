# Perbandingan Arsitektur (Linear Probe, Backbone Beku)

Protokol: bobot ImageNet dibekukan, fitur penultimate diekstrak, lalu LogisticRegression dilatih pada 500 gambar train.
Peringkat berdasarkan akurasi validasi (300 gambar).

| Rank | Model | Params | Dim Fitur | Val Acc | Test Acc | Waktu (s) |
|-----:|-------|-------:|----------:|--------:|---------:|----------:|
| 1 | resnet34 | 21,284,672 | 512 | 0.9900 | 0.9800 | 5.3 |
| 2 | efficientnet_b0 | 4,007,548 | 1280 | 0.9867 | 0.9800 | 4.9 |
| 3 | densenet121 | 6,953,856 | 1024 | 0.9867 | 0.9800 | 8.1 |
| 4 | convnext_tiny | 27,820,128 | 768 | 0.9867 | 0.9850 | 7.8 |
| 5 | resnet50 | 23,508,032 | 2048 | 0.9833 | 0.9800 | 7.4 |
| 6 | mobilenet_v3_large | 4,202,032 | 1280 | 0.9833 | 0.9950 | 28.5 |
| 7 | resnet18 | 11,176,512 | 512 | 0.9767 | 0.9650 | 5.0 |

**Model terpilih:** `resnet34`

> Catatan: akurasi test di kolom ini berasal dari classifier linear di atas fitur beku, bukan dari model hasil fine-tuning. Angka final ada di `summary.md`.
