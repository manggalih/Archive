import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re
import os

# Konfigurasi Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def clean_keluhan(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_test():
    print("=== Memulai Testing Akurasi Integrasi Model ===\n")
    
    # 1. Load Model & Data
    try:
        model_rf = joblib.load(os.path.join(BASE_DIR, "model_rf.pkl"))
        tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
        dataset = joblib.load(os.path.join(BASE_DIR, "dataset.pkl"))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 2. Ambil sampel acak dari dataset untuk ditesting (10 sampel)
    test_samples = dataset.sample(10, random_state=42)
    
    correct_top1 = 0
    correct_top3 = 0
    total = len(test_samples)

    print(f"Menguji {total} sampel dari dataset...\n")
    print(f"{'No':<3} | {'Keluhan (Input)':<30} | {'Label Asli':<15} | {'Prediksi Top-1':<15} | {'Status'}")
    print("-" * 90)

    for i, (idx, row) in enumerate(test_samples.iterrows()):
        keluhan_input = row['pemeriksaan']
        label_asli = row['obat_utama']
        usia = row['usia']
        gender_val = 1 if str(row['jenis kelamin']).lower() == 'laki-laki' else 0
        
        # Preprocess sesuai logika app.py
        keluhan_bersih = clean_keluhan(keluhan_input)
        X_text = tfidf.transform([keluhan_bersih])
        fitur_tambahan = csr_matrix([[usia, gender_val]])
        X_input = hstack([X_text, fitur_tambahan])
        
        # Prediksi
        probabilitas = model_rf.predict_proba(X_input)[0]
        top_idx = np.argsort(probabilitas)[-3:][::-1]
        top_obat = label_encoder.inverse_transform(top_idx)
        
        # Evaluasi
        status = "❌"
        if label_asli.lower() == top_obat[0].lower():
            correct_top1 += 1
            correct_top3 += 1
            status = "✅ (Top-1)"
        elif label_asli.lower() in [o.lower() for o in top_obat]:
            correct_top3 += 1
            status = "🟡 (Top-3)"
            
        # Tampilkan hasil per baris
        display_keluhan = (keluhan_input[:27] + '..') if len(keluhan_input) > 27 else keluhan_input
        print(f"{i+1:<3} | {display_keluhan:<30} | {label_asli:<15} | {top_obat[0]:<15} | {status}")

    # 3. Ringkasan
    acc_top1 = (correct_top1 / total) * 100
    acc_top3 = (correct_top3 / total) * 100
    
    print("\n" + "="*40)
    print(f"HASIL AKURASI TESTING INTEGRASI")
    print("="*40)
    print(f"Akurasi Top-1 : {acc_top1:.2f}%")
    print(f"Akurasi Top-3 : {acc_top3:.2f}%")
    print(f"Interpretasi  : {'Sangat Bagus' if acc_top1 > 80 else 'Cukup Baik' if acc_top1 > 60 else 'Perlu Peningkatan Data'}")
    print("="*40)

if __name__ == "__main__":
    run_test()
