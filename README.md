# Tugas UTS Machine Learning

Repositori ini berisi pengerjaan Ujian Tengah Semester (UTS) mata kuliah Machine Learning untuk implementasi algoritma Linear Regression dan Logistic Regression secara mandiri menggunakan Python dan NumPy.

## Identitas Mahasiswa

* **Nama:** Rivaldo Ferdiansyah
* **NIM:** 241011400456
* **Kelas:** 05TPLM003
* **Jenis Perkuliahan:** Reguler B
* **Program Studi:** Teknik Informatika (S-1)
* **Random State:** 56 (2 digit terakhir NIM)

---

## Ringkasan Hasil Pengerjaan

### 1. Bagian A — Linear Regression (Dataset: Ads_Data.csv)

Tujuan bagian ini adalah memodelkan pengaruh biaya iklan ($X$, dalam juta rupiah) terhadap omset penjualan ($y$, dalam juta rupiah).

* **Persamaan Regresi (Normal Equation):**
  $$\hat{y} = b + m \cdot X$$

* **Tabel Evaluasi Model:**
| Dataset | MAE (Juta Rp) | MSE | RMSE (Juta Rp) | $R^2$ Score |
| :--- | :--- | :--- | :--- | :--- |
| Training | Disesuaikan | Disesuaikan | Disesuaikan | Disesuaikan |
| Testing | Disesuaikan | Disesuaikan | Disesuaikan | Disesuaikan |

* **Eksperimen Learning Rate pada Gradient Descent (5000 iterasi):**
  * `lr = 0.001`: Terlalu lambat, belum konvergen setelah 5000 iterasi.
  * `lr = 0.02`: Paling optimal, hasilnya konvergen mendekati/sama dengan Normal Equation.
  * `lr = 0.1`: Divergen (gagal konvergen) karena langkah pembaruan terlalu besar sehingga loss mengalami overflow.

---

### 2. Bagian B — Logistic Regression (Dataset: Email_Data.csv)

Tujuan bagian ini adalah mengklasifikasikan email menjadi Spam (1) atau Bukan Spam (0) berdasarkan jumlah link ($X_1$) dan kata promo ($X_2$).

* **Persamaan Probabilitas & Garis Batas Keputusan:**
  Menggunakan fungsi sigmoid dengan parameter bobot ($\theta$) yang dioptimalkan via Gradient Descent menggunakan `random_state=56`.

* **Eksperimen Ambang Batas (Threshold):**
  Threshold optimal dipilih untuk menjaga keseimbangan *Precision* dan *Recall* sesuai kebutuhan keamanan klasifikasi spam (berdasarkan hasil penyesuaian terbaru dengan `random_state=56`).

---

### 3. Bagian C — Pemahaman Konsep

Seluruh 5 pertanyaan konsep pada notebook sudah dijawab lengkap, mencakup:
1. Alasan penamaan regresi logistik.
2. Peran fungsi sigmoid dalam klasifikasi.
3. Perbandingan Normal Equation vs Gradient Descent.
4. Penyebab terjadinya *overfitting* pada model.
5. Perbandingan antara MSE vs Log Loss, serta pengaruh variasi NIM pada `random_state` (`random_state=56`).

---

### 4. Bagian Bonus — Perbandingan dengan Scikit-Learn

* Pada **Linear Regression**, nilai intercept, koefisien, MSE, dan $R^2$ antara implementasi manual dengan scikit-learn menghasilkan angka yang persis sama.
* Pada **Logistic Regression**, saat regularisasi scikit-learn dinonaktifkan (`penalty=None`), hasil akurasi pada data testing sama persis dengan kode Gradient Descent manual.

---

## Berkas dalam Repositori

* `README.md` : Dokumentasi identitas dan ringkasan hasil UTS
* `tugas_UTS.ipynb` : Jupyter Notebook lengkap dengan kode, grafik, dan jawaban analisis
* `Ads_Data.csv` : Dataset untuk Linear Regression
* `Email_Data.csv` : Dataset untuk Logistic Regression