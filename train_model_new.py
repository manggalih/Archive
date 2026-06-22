import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

def clean_keluhan(text):
    text = str(text).lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ekstrak_obat_utama(resep):
    resep = str(resep).lower()
    obat = resep.split(',')[0]
    obat = re.sub(r'\d+x\d+.*', '', obat)
    obat = re.sub(r'\d+\s*mg.*', '', obat)
    obat = re.sub(r'\b(tab|tablet|sirup|syr|salep|zalf)\b', '', obat)
    obat = re.sub(r'\d+', '', obat)
    obat = re.sub(r'\s+', ' ', obat)
    return obat.strip()

def train_and_save_model(excel_path):
    print("Memuat dataset...")
    df = pd.read_excel(excel_path)
    
    # Seleksi atribut
    df = df[['jenis kelamin', 'tanggal lahir', 'pemeriksaan', 'pengobatan/resep']]
    
    # Data cleaning
    df = df.dropna(subset=['pemeriksaan', 'pengobatan/resep'])
    df = df.drop_duplicates()
    
    # Hitung usia (menggunakan pendekatan sederhana sesuai notebook)
    df['tanggal lahir'] = pd.to_datetime(df['tanggal lahir'], errors='coerce')
    today = pd.Timestamp.today()
    df['usia'] = ((today - df['tanggal lahir']).dt.days / 365.25).round()
    median_usia = round(df['usia'].median())
    df['usia'] = df['usia'].fillna(median_usia).astype(int)
    df = df[(df['usia'] >= 0) & (df['usia'] <= 120)]
    
    # Preprocessing Keluhan
    df['keluhan'] = df['pemeriksaan'].apply(clean_keluhan)
    
    # Preprocessing Resep/Obat
    df = df[df['pengobatan/resep'].astype(str).str.strip() != '0'].copy()
    df['obat_utama'] = df['pengobatan/resep'].apply(ekstrak_obat_utama)
    
    # Normalisasi Nama Obat
    normalisasi_obat = {
        'ceterizin': 'cetirizine',
        'cetirizinee': 'cetirizine',
        'ranitidinen': 'ranitidine',
        'captopill': 'captopril',
        'acylovir': 'acyclovir',
        'diclofana': 'diclofenac'
    }
    df['obat_utama'] = df['obat_utama'].replace(normalisasi_obat)
    
    # Seleksi Frekuensi Obat (>= 3 sesuai notebook)
    frekuensi = df['obat_utama'].value_counts()
    obat_valid = frekuensi[frekuensi >= 3].index
    df = df[df['obat_utama'].isin(obat_valid)]
    
    print(f"Jumlah data setelah cleaning: {len(df)}")
    print(f"Jumlah kelas obat: {df['obat_utama'].nunique()}")
    
    # Encoding Jenis Kelamin
    df['jenis_kelamin_encoded'] = df['jenis kelamin'].map({'laki-laki': 1, 'perempuan': 0})
    
    # TF-IDF
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(df['keluhan'])
    
    # Label Encoding Obat
    label_encoder = LabelEncoder()
    df['obat_encoded'] = label_encoder.fit_transform(df['obat_utama'])
    
    # Gabungkan Fitur
    X = hstack([X_tfidf, df[['usia', 'jenis_kelamin_encoded']].values])
    y = df['obat_encoded']
    
    # Latih Model
    print("Melatih model Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X, y)
    
    # Simpan file
    print("Menyimpan model dan pkl files...")
    joblib.dump(rf_model, 'model_rf.pkl')
    joblib.dump(tfidf, 'tfidf.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    # Dataset disimpan untuk kebutuhan pencarian histori mirip di app.py
    joblib.dump(df, 'dataset.pkl')
    
    print("Selesai! Model dan file pendukung telah diperbarui.")

if __name__ == "__main__":
    # Mencari file excel di direktori saat ini atau folder Downloads
    # Berdasarkan log notebook, nama file aslinya kemungkinan DATA_FINAL_TERBARU.xlsx
    excel_file = "DATA_FINAL_TERBARU.xlsx"
    if os.path.exists(excel_file):
        train_and_save_model(excel_file)
    else:
        print(f"File {excel_file} tidak ditemukan. Pastikan file excel ada di folder ini.")
