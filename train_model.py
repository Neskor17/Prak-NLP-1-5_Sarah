import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Import fungsi preprocessing dari minggu 2
from preprocessing import preprocess 

# 2. Membaca Dataset
data = pd.read_csv("dataset.csv")
print("Data Awal:")
print(data.head())

# 3. Preprocessing (Membersihkan teks)
print("\nSedang membersihkan teks...")
data["clean_text"] = data["text"].apply(preprocess)

# 4. Feature Extraction dengan TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_text"])

# 5. Membagi Dataset (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, data["label"], test_size=0.2, random_state=42
)

# 6. Training Model Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# 7. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {accuracy * 100:.2f}%")

# 8. Menyimpan Model (Disesuaikan agar masuk ke folder 'model')
import os

# Membuat folder 'model' jika belum ada
if not os.path.exists('model'):
    os.makedirs('model')

# Simpan ke dalam folder 'model'
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("\nModel dan Vectorizer berhasil disimpan di folder 'model'!")

print("\n--- Hasil Uji 10 Kalimat Baru ---")

kalimat_uji = [
    "Gokil sih ini mah, barangnya jauh lebih bagus dari ekspektasi gue!", # Positif
    "Jangan mau beli di sini, barangnya zonk banget sumpah nyesel parah", # Negatif
    "Worth it parah buat harga semurah ini, fungsional banget!",          # Positif
    "Udah nunggu lama-lama eh yang dateng malah salah warna, malesin bgt",# Negatif
    "Auto langganan ini mah, sellernya fast respon dan amanah abis",      # Positif
    "Bahannya kasar kayak saringan tahu, mending beli di pasar sekalian", # Negatif
    "Cuma sehari doang langsung nyampe, packingnya tebel aman jaya",      # Positif
    "Niatnya mau kasih kado tapi barangnya penyok semua, gak rekomen",    # Negatif
    "Sumpah ini enak banget, bumbunya gak pelit melimpah ruah!",          # Positif
    "Kecewa sih, dicharge dari tadi pagi tapi gak penuh-penuh baterainya" # Negatif
]

# Ubah kalimat uji jadi angka pakai vectorizer yang tadi
kalimat_uji_clean = [preprocess(k) for k in kalimat_uji]
test_vec = vectorizer.transform(kalimat_uji_clean)

# Prediksi pakai model
predictions = model.predict(test_vec)

# Tampilkan hasil
for teks, hasil in zip(kalimat_uji, predictions):
    print(f"Kalimat: {teks} --> Prediksi: {hasil}")