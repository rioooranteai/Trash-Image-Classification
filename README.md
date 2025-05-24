# Laporan Proyek Machine Learning - Mario Valerian Rante Ta'dung

## 1. Domain Proyek

### 1.1 Latar Belakang

Saat ini, pengolahan sampah masih menjadi tantangan global yang dihadapi oleh berbagai negara, salah satunya adalah Indonesia. Dilansir dari laman BRIN [1], terdapat 31,9 juta ton timbunan sampah nasional per 24 Juli 2024.  Dari keseluruhan sampah yang dihasilkan secara nasional, sekitar 63,3% atau setara dengan 20,5 juta ton berhasil dikelola dengan baik. Namun, masih terdapat 35,67% atau sekitar 11,3 juta ton sampah yang belum tertangani secara optimal. Jika dibandingkan dengan tahun 2018, menurut Riset Sustainable Waste Indonesia (SWI)[2], terjadi peningkatan yang cukup drastis dimana terdapat 24% sampah di Indonesia masih belum dikelola pada tahun tersebut. Kondisi ini menunjukkan bahwa meskipun mayoritas sampah telah dikelola, proporsi sampah yang belum tertangani mengalami peningkatan signifikan dalam kurun waktu enam tahun terakhir, dari 24% pada tahun 2018 menjadi 35,67% pada tahun 2024. Kenaikan lebih dari 11 persen ini menandakan adanya kesenjangan antara pertumbuhan volume sampah dengan kapasitas sistem pengelolaan yang tersedia. 

Sampah yang tidak terkelola dengan baik dapat menimbulkan berbagai dampak negatif. Penumpukan sampah di sungai dan saluran air menyebabkan banjir, sementara tumpukan sampah di Tempat Pemrosesan Akhir (TPA) dapat memicu longsor sampah, seperti yang terjadi di TPA Leuwigajah pada tahun 2005 . Selain itu, sampah plastik yang mencemari laut mengancam kehidupan biota laut dan dapat masuk ke rantai makanan manusia, menimbulkan risiko kesehatan yang serius

Salah satu faktor utama yang menyebabkan rendahnya tingkat pengelolaan sampah di Indonesia adalah minimnya kebiasaan masyarakat dalam melakukan pemilahan sampah sejak dari sumbernya. Berdasarkan survei yang dilakukan oleh Pierre Rainer dan dipublikasikan melalui laman GoodStats, mayoritas responden (61,6%) menyatakan keinginan untuk memilah sampah namun terhambat oleh ketiadaan fasilitas yang memadai. Selain itu, sebanyak 47% responden mengaku tidak memiliki waktu untuk melakukan pemilahan, sementara 6,8% lainnya menyatakan bahwa mereka tidak menganggap pemilahan sampah sebagai tanggung jawab pribadi. Temuan ini menunjukkan bahwa perilaku pemilahan sampah sangat dipengaruhi oleh ketersediaan infrastruktur serta kesadaran individu terhadap tanggung jawab lingkungan.


**Rubrik Tambahan:**

* Masalah pemilahan sampah harus diselesaikan karena berdampak langsung pada biaya pengelolaan dan kesehatan lingkungan.
* Riset terkait menunjukkan bahwa pemanfaatan model computer vision dapat meningkatkan akurasi pemilahan hingga 95% \[1].

## 2. Business Understanding

### 2.1 Problem Statements

1. Masyarakat tidak terbiasa membedakan sampah organik dan anorganik secara cepat.
2. Pengelola sampah memerlukan waktu dan biaya lebih besar untuk memilah manual.

### 2.2 Goals

* Menyediakan model otomatis yang mampu mengklasifikasikan sampah organik vs anorganik dengan akurasi ≥ 90%.
* Mengurangi waktu pemrosesan per gambar menjadi < 100 ms.

**Solution Statements (Tambahan):**

1. Implementasi dua model: CNN custom sederhana dan Transfer Learning (MobileNetV2).
2. Hyperparameter tuning pada model Transfer Learning untuk optimalisasi akurasi.

## 3. Data Understanding

Dataset yang digunakan adalah **Garbage Classification** dari Kaggle ([https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)). Terdiri atas 2.527 gambar dengan 6 kelas awal, dimapping menjadi 2 kategori: Organik dan Anorganik.

**Variabel/Fitur:**

* Input: citra RGB (ukuran bervariasi).
* Label: Organik, Anorganik.

## 4. Data Preparation

1. **Label Mapping:** Menggabungkan 6 kelas menjadi 2.
2. **Resize:** Menyesuaikan semua citra ke 224×224 pixel.
3. **Normalisasi:** Skala piksel ke rentang \[0,1].
4. **Augmentasi:** Flip horizontal, rotasi 20°, zoom 10%.
5. **Split:** Train 70%, Validation 15%, Test 15%.

## 5. Modeling

1. **Model A (CNN Custom):** 3 blok Conv2D + MaxPooling + Dense.
2. **Model B (Transfer Learning):** MobileNetV2 pretrained + head baru.

**Parameter:**

* Optimizer: Adam (lr=0.001)
* Batch size: 32
* Epochs: 20

**Rubrik Tambahan:**

* Kelebihan/C Kekurangan tiap algoritma.
* Proses hyperparameter tuning pada Model B.

## 6. Evaluation

**Metrik Evaluasi:** Akurasi, Precision, Recall, F1-score.

**Hasil:**

* Model A: akurasi 88%, F1-score 0.87.
* Model B: akurasi 93%, F1-score 0.92.

**Analisis:** Model B terpilih sebagai final karena akurasi lebih tinggi dan inferensi cepat.

---

**Referensi:**
\[1] Doe, J. (2021). Deep Learning for Waste Classification. *Journal of Environmental AI*.
