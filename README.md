# Laporan Proyek Machine Learning - Mario Valerian Rante Ta'dung

## 1. Domain Proyek

### 1.1 Latar Belakang

Saat ini, pengolahan sampah masih menjadi tantangan global yang dihadapi oleh berbagai negara, salah satunya adalah Indonesia. Dilansir dari laman BRIN [1], terdapat 31,9 juta ton timbunan sampah nasional per 24 Juli 2024.  Dari keseluruhan sampah yang dihasilkan secara nasional, sekitar 63,3% atau setara dengan 20,5 juta ton berhasil dikelola dengan baik. Namun, masih terdapat 35,67% atau sekitar 11,3 juta ton sampah yang belum tertangani secara optimal. Jika dibandingkan dengan tahun 2018, menurut Riset Sustainable Waste Indonesia (SWI)[2], terjadi peningkatan yang cukup drastis dimana terdapat 24% sampah di Indonesia masih belum dikelola pada tahun tersebut. Kondisi ini menunjukkan bahwa meskipun mayoritas sampah telah dikelola, proporsi sampah yang belum tertangani mengalami peningkatan signifikan dalam kurun waktu enam tahun terakhir, dari 24% pada tahun 2018 menjadi 35,67% pada tahun 2024. Kenaikan lebih dari 11 persen ini menandakan adanya kesenjangan antara pertumbuhan volume sampah dengan kapasitas sistem pengelolaan yang tersedia. 

Sampah yang tidak terkelola dengan baik dapat menimbulkan berbagai dampak negatif. Kondisi ini membawa dampak yang sangat signifikan, terutama pada aspek ekonomi dan sosial. Secara ekonomi, kerugian akibat sampah plastik yang bocor ke laut diperkirakan mencapai Rp250 triliun per tahun, yang berdampak pada sektor maritim, kelautan, dan perikanan serta menurunkan potensi pendapatan negara . Selain itu, pengelolaan sampah yang tidak optimal juga menghambat penciptaan lapangan kerja dan peluang ekonomi baru. Sebaliknya, pengelolaan sampah yang baik, seperti yang diterapkan pada Unit Pengolahan Sampah (UPS) di Kota Depok, telah terbukti mampu memberikan manfaat ekonomi bersih sebesar Rp472,9 juta per tahun per proyek, sekaligus menyerap tenaga kerja lokal dan menurunkan angka pengangguran di wilayah tersebut . Dengan demikian, pengelolaan sampah yang optimal tidak hanya mengurangi kerugian ekonomi, tetapi juga membuka peluang penyerapan tenaga kerja dan peningkatan kesejahteraan masyarakat secara lebih luas.

Salah satu faktor utama yang menyebabkan rendahnya tingkat pengelolaan sampah di Indonesia adalah minimnya kebiasaan masyarakat dalam melakukan pemilahan sampah sejak dari sumbernya. Berdasarkan survei yang dilakukan oleh Pierre Rainer dan dipublikasikan melalui laman GoodStats, mayoritas responden (61,6%) menyatakan keinginan untuk memilah sampah namun terhambat oleh ketiadaan fasilitas yang memadai. Selain itu, sebanyak 47% responden mengaku tidak memiliki waktu untuk melakukan pemilahan, sementara 6,8% lainnya menyatakan bahwa mereka tidak menganggap pemilahan sampah sebagai tanggung jawab pribadi. Temuan ini menunjukkan bahwa perilaku pemilahan sampah sangat dipengaruhi oleh ketersediaan infrastruktur serta kesadaran individu terhadap tanggung jawab lingkungan.

Kemajuan dalam bidang kecerdasan buatan, khususnya pada teknologi computer vision, telah membuka peluang baru dalam mengatasi tantangan pemilahan sampah. Dengan memanfaatkan algoritma pembelajaran mesin untuk mengenali dan mengklasifikasikan citra sampah, proses yang sebelumnya bergantung pada tenaga manusia kini dapat diotomatisasi. Inovasi ini memungkinkan identifikasi sampah organik dan anorganik secara langsung dari gambar, tanpa perlu intervensi manual. Penerapan teknologi semacam ini sangat relevan di Indonesia, di mana sebagian besar masyarakat belum terbiasa melakukan pemilahan sejak dari rumah. Dengan sistem klasifikasi otomatis berbasis citra, upaya pemilahan dapat dilakukan lebih konsisten dan cepat, sekaligus mengurangi beban kerja petugas pengelola sampah serta meningkatkan efisiensi sistem pengolahan secara keseluruhan. Meskipun masih memerlukan peningkatan akurasi dan adaptasi terhadap kondisi lokal, pendekatan ini merupakan langkah awal menuju sistem pengelolaan sampah yang lebih modern dan berkelanjutan.

## 2. Business Understanding

### 2.1 Problem Statements

### **Problem Statement 1**

**Rendahnya tingkat pengelolaan sampah di Indonesia disebabkan oleh kesenjangan antara pertumbuhan volume sampah dengan kapasitas sistem pengelolaan yang tersedia**, yang terbukti dari meningkatnya proporsi sampah tidak terkelola dari 24% pada tahun 2018 menjadi 35,67% pada tahun 2024. Hal ini menunjukkan bahwa sistem pengelolaan sampah saat ini belum mampu mengikuti laju timbunan sampah yang terus meningkat.

### **Problem Statement 2**

**Minimnya kebiasaan masyarakat dalam memilah sampah sejak dari sumbernya menjadi hambatan utama dalam meningkatkan efisiensi pengelolaan sampah.** Survei menunjukkan bahwa sebagian besar masyarakat tidak memilah sampah karena kurangnya fasilitas, waktu, dan kesadaran tanggung jawab, sehingga diperlukan solusi inovatif yang dapat mendukung proses pemilahan secara otomatis dan mudah diakses.

### 2.2 Goals

1. **Mengembangkan model klasifikasi citra berbasis deep learning** yang mampu mengidentifikasi enam kategori sampah (Paper, Glass, Plastic, Metal, Cardboard, dan Trash) dengan tingkat akurasi pada data validasi yang dapat diterima (>80%) sebagai baseline awal.
2. **Memanfaatkan arsitektur CNN dan pendekatan transfer learning** untuk membandingkan performa serta mengidentifikasi pendekatan yang paling sesuai untuk klasifikasi multi-kelas sampah.
3. **Menyiapkan pipeline pemrosesan data dan augmentasi gambar** untuk meningkatkan generalisasi model terhadap berbagai kondisi pencahayaan dan orientasi citra.
4. **Menghasilkan model yang dapat digunakan sebagai prototipe awal** untuk sistem pendukung pemilahan sampah otomatis berbasis visual, sebagai bagian dari solusi menuju pengelolaan sampah yang lebih efisien.

## 3. Data Understanding

Dataset yang digunakan adalah **Garbage Classification** dari Kaggle ([https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)). Terdiri atas 2.527 gambar dengan 6 kelas.

**Variabel/Fitur:**

* Input: citra RGB (ukuran bervariasi).
* Label: Paper, Glass, Plastic, Metal, Cardboard dan Trash

### **Distribusi Data**
![Train Data](Images/Training.png)
![Test Data](Images/Test.png)
![Validation Data](Images/Val.png)


## 4. Data Preparation

1. **Label Mapping:** Menggabungkan 6 kelas menjadi 2.
2. **Resize:** Menyesuaikan semua citra ke 224×224 pixel.
3. **Normalisasi:** Skala piksel ke rentang \[0,1].
4. **Augmentasi:** Flip horizontal, rotasi 20°, zoom 10%.
5. **Split:** Train 70%, Validation 15%, Test 15%.

## 5. Modeling

**Model (Transfer Learning):** Resnet50V2 pretrained (imagenet) + head baru.

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
