# 📊 UTS Machine Learning: Linear & Logistic Regression from Scratch

Repositori ini berisi pengerjaan Ujian Tengah Semester (UTS) mata kuliah **Machine Learning**. Seluruh model, fungsi loss, metrik evaluasi, dan proses *train-test split* diimplementasikan secara mandiri menggunakan **NumPy** (*from scratch*) tanpa bergantung pada `scikit-learn` (kecuali untuk soal bonus perbandingan).

---

## 👨‍💻 Identitas Mahasiswa

* **Nama:** Rivaldo Ferdiansyah
* **NIM:** 241011400456
* **Kelas:** Reg B
* **Random State:** 56

---

## 📂 Struktur Repositori

```text
├── Ads_Data.csv              # Dataset Linear Regression (Biaya Iklan vs Penjualan)
├── Email_Data.csv            # Dataset Logistic Regression (Filter Spam)
├── uts_ml_rivaldo.ipynb      # Jupyter Notebook berisi kode lengkap pengerjaan UTS
└── README.md                 # Dokumentasi repositori
```

---

## 📌 Rincian Jawaban & Hasil Pengerjaan UTS

---

### Bagian A – Linear Regression (35%)

#### Soal A1 – Load & Visualisasi Data
* **Jumlah baris data**: 40 baris.
* **5 data pertama**:
  * `BiayaIklan = 0.8`, `Penjualan = 34.3`
  * `BiayaIklan = 0.9`, `Penjualan = 38.8`
  * `BiayaIklan = 1.0`, `Penjualan = 33.2`
  * `BiayaIklan = 1.3`, `Penjualan = 36.0`
  * `BiayaIklan = 1.7`, `Penjualan = 51.3`
* **Analisis Visual**: Scatter plot menunjukkan tren linier positif yang kuat antara biaya iklan dan penjualan. Semakin besar biaya iklan yang dikeluarkan, semakin tinggi pula total penjualan yang dihasilkan.

#### Soal A2 – Split Data
* Pembagian data menggunakan `test_size=0.2` dan `random_state=56`.
* **Shape Dataset**:
  * `X_train`: (32, 1) | `X_test`: (8, 1)
  * `y_train`: (32,) | `y_test`: (8,)

#### Soal A3 – Training dengan Normal Equation
* **Intercept ($\theta_0$)**: $28.5405$
* **Koefisien ($\theta_1$)**: $8.9851$
* **Persamaan Garis Regresi**:
  $$\text{Penjualan} = 28.5405 + 8.9851 \times \text{BiayaIklan}$$
* **Interpretasi**: Setiap kenaikan biaya iklan sebesar **1 juta Rupiah**, penjualan diprediksi akan meningkat sebesar **8.9851 juta Rupiah**, dengan asumsi faktor lain konstan (baseline penjualan tanpa iklan adalah 28.5405 juta Rupiah).

#### Soal A4 – Evaluasi Model

| Metrik Evaluasi | Data Training | Data Testing |
| :--- | :--- | :--- |
| **MAE** | 5.1345 | 2.2890 |
| **MSE** | 43.2469 | 8.2377 |
| **RMSE** | 6.5762 | 2.8701 |
| **$R^2$ Score** | 0.9276 | 0.9873 |

* **Analisis**: Nilai **RMSE** pada data testing sebesar **2.87** menunjukkan rata-rata besar kesalahan prediksi model terhadap data aktual. Nilai **$R^2$** sebesar **0.9873** pada data testing mengindikasikan bahwa sekitar **98.73%** variasi penjualan dapat dijelaskan oleh variabel biaya iklan. Model ini memiliki performa yang sangat baik dan kemampuan generalisasi yang tinggi.

#### Soal A5 – Gradient Descent & Learning Rate
* **Hasil Eksperimen (`n_iterations=5000`)**:
  * $\text{LR} = 0.001$: Terlalu lambat (belum konvergen optimal, final loss 45.07). Intercept = 25.72, Coef = 9.44.
  * $\text{LR} = 0.02$: Optimal/Pas (konvergen sempurna menyamai hasil normal equation). Intercept = 28.54, Coef = 8.99.
  * $\text{LR} = 0.1$: Gagal konvergen / **Divergen** (`RuntimeWarning: overflow encountered in square`), karena nilai learning rate yang terlalu besar menyebabkan pembaruan bobot melompati titik minimum dan nilai loss meledak tak berhingga.

#### Soal A6 – Visualisasi & Prediksi
* **Hasil Prediksi Penjualan**:
  1. Biaya Iklan $4.5$ juta $\rightarrow$ **$68.9735$** juta
  2. Biaya Iklan $7.0$ juta $\rightarrow$ **$91.4360$** juta
  3. Biaya Iklan $12.0$ juta $\rightarrow$ **$136.3616$** juta
* **Analisis Ekstrapolasi ($12$ juta)**: Nilai ini berada di luar rentang data training maksimum ($9.9$ juta). Prediksi ekstrapolasi **kurang dapat dipercaya secara mutlak** karena mengasumsikan hubungan linier berlanjut tanpa batas, padahal dalam bisnis riil biasanya terdapat titik jenuh (_diminishing returns_).

---

### Bagian B – Logistic Regression (40%)

#### Soal B1 – Load & Visualisasi Data
* **Total data**: 60 email.
* **Distribusi Target**: 
  * Bukan Spam (`0`): 35 email
  * Spam (`1`): 25 email
* **Analisis Visual**: Scatter plot `JumlahLink` vs `KataPromo` menunjukkan bahwa email spam cenderung memiliki kombinasi jumlah link dan kata promo yang lebih tinggi dibandingkan email bukan spam.

#### Soal B2 – Split Data & Training
* `test_size=0.25`, `random_state=56`.
* **Parameter Model (`learning_rate=0.1`, `n_iterations=10000`)**:
  * Intercept ($\theta_0$): $-9.0614$
  * Koefisien ($\theta_1, \theta_2$): $[0.8171, 0.8041]$
* **Persamaan Logit**:
  $$\text{Logit} = -9.0614 + 0.8171 \times \text{JumlahLink} + 0.8041 \times \text{KataPromo}$$
* **Grafik Log Loss**: Loss turun secara konsisten dan stabil mendekati akhir iterasi, menandakan proses pelatihan berhasil konvergen.

#### Soal B3 – Evaluasi Model
* **Confusion Matrix (Testing Data)**:
  $$\begin{pmatrix} 10 & 1 \\ 0 & 4 \end{pmatrix}$$
  *(TN = 10, FP = 1, FN = 0, TP = 4)*
* **Metrik Evaluasi**:
  * **Accuracy**: $0.9333$ ($93.33\%$)
  * **Precision**: $0.8000$
  * **Recall**: $1.0000$
  * **F1-score**: $0.8889$
* **Analisis Kesalahan**:
  * **False Positive (FP)**: 1 email normal masuk kategori spam.
  * **False Negative (FN)**: 0 email spam lolos ke inbox.
  * **Mana yang lebih merugikan?** Bagi pengguna umum, **False Positive** lebih merugikan karena pesan penting (seperti pekerjaan atau tagihan) terlewatkan di folder spam.

#### Soal B4 – Decision Boundary
* **Persamaan Decision Boundary ($P=0.5 \rightarrow \text{Logit}=0$)**:
  $$0 = -9.0614 + 0.8171 \times \text{JumlahLink} + 0.8041 \times \text{KataPromo}$$
  $$\text{KataPromo} = \frac{9.0614 - 0.8171 \times \text{JumlahLink}}{0.8041}$$
* **Visualisasi**: Garis batas keputusan memisahkan ruang fitur secara linier. Hanya terdapat 1 data testing yang salah diklasifikasikan di sekitar garis batas tersebut.

#### Soal B5 – Eksperimen Threshold

| Threshold | Accuracy | Precision | Recall | F1-score | Confusion Matrix |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0.3** | 0.8000 | 0.5714 | 1.0000 | 0.7273 | `[[8, 3], [0, 4]]` |
| **0.5** | 0.9333 | 0.8000 | 1.0000 | 0.8889 | `[[10, 1], [0, 4]]` |
| **0.7** | 0.9333 | 0.8000 | 1.0000 | 0.8889 | `[[10, 1], [0, 4]]` |

* **Analisis**: Menurunkan threshold ke 0.3 memperbanyak False Positive (menaikkan recall tetapi menurunkan precision). Threshold **0.5** memberikan keseimbangan terbaik antara akurasi, precision, dan recall untuk filter spam ini.

#### Soal B6 – Prediksi Email Baru (Threshold = 0.5)

| Email | JumlahLink | KataPromo | Prob_Spam | Pred_Spam |
| :---: | :---: | :---: | :---: | :---: |
| **E1** | 1 | 2 | 0.0013 | **0 (Bukan Spam)** |
| **E2** | 5 | 6 | 0.4622 | **0 (Bukan Spam)** |
| **E3** | 9 | 10 | 0.9982 | **1 (Spam)** |
| **E4** | 2 | 11 | 0.8050 | **1 (Spam)** |

---

### Bagian C – Pemahaman Konsep (15%)

* **Jawaban C1**: Logistic Regression dinamakan "regression" karena basis matematikanya menggunakan model regresi linier untuk menghitung log-odds dari target biner. **Fungsi sigmoid** berperan mengubah output kontinu dari regresi linier tersebut menjadi nilai probabilitas antara $0$ dan $1$.
* **Jawaban C2**: Normal equation menghitung solusi analitik secara langsung menggunakan matriks invers (sangat cepat untuk dataset kecil, namun lambat $O(n^3)$ untuk fitur besar). Gradient descent adalah metode optimasi iteratif berbasis turunan yang sangat ideal digunakan ketika ukuran dataset sangat besar.
* **Jawaban C3**: Evaluasi pada data testing wajib dilakukan untuk mengukur kemampuan generalisasi model terhadap data baru yang belum pernah dilihat saat pelatihan. **Overfitting** adalah kondisi ketika model terlalu menghafal data training sehingga gagal berkinerja baik pada data testing.
* **Jawaban C4**: Linear Regression menggunakan MSE karena targetnya berupa nilai kontinu dengan asumsi distribusi normal. Sebaliknya, Logistic Regression menggunakan log loss (binary cross-entropy) karena outputnya berupa probabilitas diskrit, di mana MSE akan menghasilkan fungsi cost non-konveks yang sulit dioptimalkan.
* **Jawaban C5**: Perbedaan angka bisa terjadi karena faktor **random_state** (pembagian data train-test yang berbeda jika tidak diset sama), inisialisasi bobot awal pada gradient descent, atau perbedaan pustaka/versi library yang digunakan.

---

### Bonus (+5) – Perbandingan dengan Scikit-Learn
* Hasil koefisien dari implementasi NumPy manual sama persis dengan `sklearn.linear_model.LinearRegression`. Untuk Logistic Regression, terdapat sedikit perbedaan kecil karena `sklearn` secara default menggunakan solver `lbfgs` dengan regularisasi L2 ($C=1.0$), sedangkan implementasi manual kita menggunakan gradient descent murni tanpa regularisasi (`penalty=None`).

---

## 🚀 Cara Menjalankan Notebook

1. Pastikan pustaka yang dibutuhkan sudah terinstal:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Buka Jupyter Notebook dan jalankan file `uts_ml_rivaldo.ipynb` secara berurutan (*Run All*).