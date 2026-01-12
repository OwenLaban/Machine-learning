# EndToEnd-Fish Image Classification Pipeline

Pipeline end-to-end untuk **klasifikasi gambar ikan** menggunakan TensorFlow/Keras, mulai dari pemuatan dataset folder-based, augmentasi, penanganan class imbalance, training model (CNN custom + transfer learning MobileNetV2), evaluasi, hingga interpretability dengan **Grad-CAM**.

## Fitur Utama
- Load dataset dari struktur folder `train/val/test` dengan `image_dataset_from_directory`.
- Reproducibility dengan seed (random, NumPy, TensorFlow).
- Data augmentation di dalam model (mis. RandomFlip/Rotation/Zoom/Contrast).
- Penanganan class imbalance via **class weights**.
- Dua pendekatan model:
  - **Custom CNN** (SeparableConv2D + BatchNorm + ReLU + MaxPool + GlobalAveragePooling + Dropout + Softmax).
  - **Transfer Learning** dengan **MobileNetV2** sebagai feature extractor (dibekukan) + pooling + dropout + softmax.
- Evaluasi: kurva loss/accuracy, confusion matrix, classification report.
- Interpretability: **Grad-CAM** untuk melihat area gambar yang memengaruhi prediksi.

## Struktur Dataset
Pastikan dataset mengikuti format berikut (sesuai notebook).

    FishImgDataset/
    ├─ train/
    │  ├─ Bangus/
    │  ├─ Big Head Carp/
    │  ├─ ...
    ├─ val/
    │  ├─ Bangus/
    │  ├─ Big Head Carp/
    │  ├─ ...
    └─ test/
       ├─ Bangus/
       ├─ Big Head Carp/
       ├─ ...

## Environment & Dependensi
Notebook menggunakan Python dan library utama berikut.
- TensorFlow
- NumPy
- Matplotlib
- (Untuk evaluasi) scikit-learn, seaborn

Instalasi cepat (contoh):

    pip install tensorflow numpy matplotlib scikit-learn seaborn

## Cara Menjalankan
1. Siapkan dataset sesuai struktur folder `train/val/test`.
2. Buka dan jalankan notebook:
   - `Fish_Image_local.ipynb`
3. Pastikan path dataset di notebook sesuai lokasi dataset lokal (di notebook menggunakan `Path(...)` untuk `TRAINDIR`, `VALDIR`, `TESTDIR`).

## Alur Pipeline
- Setup & seed untuk hasil konsisten.
- Inspect distribusi kelas dan jumlah gambar per kelas.
- Buat `tf.data.Dataset` untuk train/val/test dan optimasi dengan shuffle + prefetch.
- Tambahkan augmentasi data di awal model agar hanya aktif saat training.
- Hitung class weights untuk mengatasi ketidakseimbangan kelas.
- Training:
  - Custom CNN.
  - Transfer learning MobileNetV2.
- Evaluasi model dengan confusion matrix dan classification report.
- Interpretability Grad-CAM untuk visualisasi kontribusi region gambar terhadap prediksi.

## Output / Artefak
- Model checkpoint tersimpan ke folder `checkpoints/` (melalui callback).
- Plot training (akurasi/loss), confusion matrix heatmap, dan visualisasi Grad-CAM.

## Catatan
- Performa model dipengaruhi kualitas dataset, ketidakseimbangan kelas, dan tuning hyperparameter (learning rate, jumlah epoch, augmentasi, dsb.).
- Transfer learning sering lebih stabil dibanding CNN dari nol untuk dataset gambar umum, namun fine-tuning sebagian layer bisa dipertimbangkan bila diperlukan.
