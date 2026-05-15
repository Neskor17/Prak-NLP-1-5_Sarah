from flask import Flask, render_template, request
import joblib
import os
# Memanggil fungsi preprocess dari file preprocessing.py kamu
from preprocessing import preprocess 

app = Flask(__name__)

# --- LOAD MODEL & VECTORIZER ---
# Pastikan kamu sudah menjalankan train_model.py sehingga folder 'model' sudah ada
try:
    model = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
except Exception as e:
    print(f"Error: Model tidak ditemukan! Jalankan train_model.py dulu ya. Detail: {e}")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    original_text = ""
    clean_text = ""
    
    if request.method == "POST":
        # Ambil teks dari form di index.html
        original_text = request.form["text"]
        
        # 1. Jalankan Preprocessing (Tugas Poin 12.2)
        # Ini akan membersihkan teks menggunakan fungsi yang sudah kamu buat
        clean_text = preprocess(original_text)
        
        # 2. Transform teks bersih menjadi angka pakai vectorizer
        vec = vectorizer.transform([clean_text])
        
        # 3. Prediksi menggunakan model Naive Bayes
        pred = model.predict(vec)
        prediction = pred[0] # Hasilnya: 'positif' atau 'negatif'
        
    # Kirim data ke tampilan (index.html)
    return render_template("index.html", 
                           prediction=prediction, 
                           text=original_text,
                           cleaned=clean_text)

if __name__ == "__main__":
    # Menjalankan server Flask dalam mode debug agar mudah cek error
    app.run(debug=True)