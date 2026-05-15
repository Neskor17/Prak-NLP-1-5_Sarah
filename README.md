# 🤖 Sentiment Analysis Web Application (NLP Project)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Web_Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)

Selamat datang di repository saya! Project ini adalah **Mini Project Sentiment Analysis** yang dibangun sebagai bagian dari praktikum *Natural Language Processing* (NLP). Project ini mengintegrasikan model klasifikasi teks ke dalam aplikasi web berbasis Flask.

## 🎯 Tujuan Project
*   Mengimplementasikan pipeline NLP secara *end-to-end*.
*   Melatih model klasifikasi sentimen menggunakan algoritma **Naive Bayes**.
*   Melakukan *deployment* model ke dalam antarmuka web yang interaktif.

## 🏗️ Arsitektur Sistem
Aplikasi mengikuti alur kerja berikut:
1.  **User Input**: User memasukkan teks melalui interface web.
2.  **Preprocessing**: Pembersihan teks dan normalisasi.
3.  **Vectorization**: Mengubah teks menjadi representasi angka menggunakan **TF-IDF**.
4.  **Inference**: Model Naive Bayes memprediksi kategori (Positif/Negatif).
5.  **Output**: Hasil prediksi ditampilkan kembali ke user secara *real-time*.

## 📂 Struktur Direktori
```text
sentiment_project/
├── app.py              # Entry point aplikasi Flask
├── preprocessing.py    # Modul pengolahan teks (cleaning/tokenizing)
├── train_model.py      # Script untuk training model
├── dataset.csv         # Dataset sentimen
├── model/
│   ├── model.pkl       # Serialized Model (Naive Bayes)
│   └── vectorizer.pkl  # Serialized Vectorizer (TF-IDF)
├── templates/
│   └── index.html      # Frontend (HTML structure)
└── static/             # File pendukung (CSS/JS)
