# ==============================
# IMPORT LIBRARY
# ==============================

import pandas as pd
import os
import re
import uuid
import joblib
import numpy as np
import requests

from flask import (
    Flask,
    render_template,
    request
)

from datetime import datetime

# ==============================
# FLASK
# ==============================

app = Flask(__name__)

# ==============================
# BASE DIRECTORY
# ==============================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FILE_HISTORI = os.path.join(
    BASE_DIR,
    "histori_pasien.csv"
)

# ==============================
# LOAD MODEL & DATASET
# ==============================

model_rf = joblib.load(
    os.path.join(BASE_DIR, "model_rf.pkl")
)

tfidf = joblib.load(
    os.path.join(BASE_DIR, "tfidf.pkl")
)

label_encoder = joblib.load(
    os.path.join(BASE_DIR, "label_encoder.pkl")
)

dataset_model = joblib.load(
    os.path.join(BASE_DIR, "dataset.pkl")
)
print(dataset_model.columns.tolist())
# ==============================
# CLEANING TEXT
# ==============================

def clean_keluhan(text):

    text = str(text)

    text = text.lower()

    text = re.sub(
        r'\d+',
        ' ',
        text
    )

    text = re.sub(
        r'[^a-zA-Z\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()
# ==============================
# RAPIIKAN TEXT CSV
# ==============================

def rapikan_text(text):

    text = str(text)

    text = text.replace("|", ",")

    text = text.replace(" . ", ", ")
    text = text.replace(" .", "")
    text = text.replace(". ", ", ")

    text = text.replace(",,", ",")

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    text = re.sub(
        r',\s*,',
        ', ',
        text
    )

    text = text.strip()

    return text

# ==============================
# CLEAN JOIN
# ==============================

def clean_join(items):

    hasil = []

    for item in items:

        item = rapikan_text(item)

        if item != "" and item not in hasil:

            hasil.append(item)

    return ", ".join(hasil)

# ==============================
# GENERATE MASTER KELUHAN
# ==============================

def generate_keluhan_master():

    kategori_keluhan = {
        "Kepala & Demam": [],
        "Mata": [],
        "Hidung": [],
        "Tenggorokan & Batuk": [],
        "Pencernaan (Perut)": [],
        "Kulit & Alergi": [],
        "Otot, Sendi & Kaki": [],
        "Lainnya": []
    }

    semua_keluhan = []

    if 'keluhan' in dataset_model.columns:
        keluhan_list = (
            dataset_model['keluhan']
            .dropna()
            .astype(str)
            .str.lower()
            .tolist()
        )
        semua_keluhan.extend(keluhan_list)

    semua_keluhan = sorted(list(set(semua_keluhan)))

    mapping = {
        "Mata": ["mata", "penglihatan", "merah pada mata", "belekan", "katarak"],
        "Hidung": ["pilek", "hidung", "bersin", "flu", "ingus", "sinus"],
        "Tenggorokan & Batuk": ["batuk", "tenggorokan", "serak", "dahak", "amandel", "menelan", "radang tenggorokan"],
        "Pencernaan (Perut)": ["mual", "muntah", "diare", "maag", "perut", "kembung", "lambung", "mencret", "ulu hati", "bab", "mencret"],
        "Kepala & Demam": ["kepala", "pusing", "migrain", "demam", "panas", "ubun", "vertigo"],
        "Kulit & Alergi": ["gatal", "ruam", "kulit", "koreng", "alergi", "biduran", "bintik", "panu", "kudis", "kurap", "eksim"],
        "Otot, Sendi & Kaki": ["kaki", "pegal", "kesemutan", "nyeri otot", "sendi", "tangan", "leher", "pinggang", "lutut", "bengkak", "nyeri sendi", "otot kaku"]
    }

    for keluhan in semua_keluhan:
        masuk = False
        keluhan_lower = keluhan.lower()
        
        # Cek berdasarkan keyword mapping
        for kategori, keywords in mapping.items():
            for key in keywords:
                if key in keluhan_lower:
                    kategori_keluhan[kategori].append(keluhan)
                    masuk = True
                    break
            if masuk: break

        if not masuk:
            kategori_keluhan["Lainnya"].append(keluhan)

    # Membersihkan duplikat dan sorting
    for kategori in kategori_keluhan:
        kategori_keluhan[kategori] = sorted(list(set(kategori_keluhan[kategori])))

    return kategori_keluhan

# ==============================
# LOAD HISTORI
# ==============================

def load_data():

    if os.path.exists(FILE_HISTORI):

        try:

            df = pd.read_csv(
                FILE_HISTORI
            )

            kolom_text = [
            'keluhan',
            'top1',
            'top2',
            'top3',
            'obat_pendamping'
        ]

            for kolom in kolom_text:

                if kolom in df.columns:

                    df[kolom] = df[kolom].astype(str).apply(
                        rapikan_text
                    )

            return df

        except:

            return pd.DataFrame()

    return pd.DataFrame()

# ==============================
# CHART DATA
# ==============================

def get_chart_data(df):

    if not df.empty:

        obat_counts = (
            df['obat']
            .value_counts()
        )

        obat_labels = list(
            obat_counts.index
        )

        obat_values = [

            int(x)
            for x in obat_counts.values
        ]

    else:

        obat_labels = []
        obat_values = []

    return obat_labels, obat_values

# ==============================
# AI EXPLANATION (HUGGING FACE)
# ==============================

def get_ai_explanation(keluhan, obat):
    """
    Menghasilkan keterangan obat menggunakan AI (Hugging Face API).
    Menggunakan model GPT-2 atau yang tersedia secara gratis.
    """
    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    # Anda bisa menambahkan API Key di sini jika punya untuk limit yang lebih besar
    # headers = {"Authorization": f"Bearer {API_TOKEN}"}
    headers = {} 

    prompt = (
        f"Berikan penjelasan singkat dan ramah dalam Bahasa Indonesia untuk pasien klinik. "
        f"Pasien mengeluh {keluhan}. Berikan alasan kenapa obat {obat} cocok untuk gejala tersebut. "
        f"Jangan terlalu teknis, gunakan bahasa yang menenangkan."
    )

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt, "parameters": {"max_length": 150}})
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', generate_keterangan(keluhan, obat))
    except Exception as e:
        print(f"AI Error: {e}")
    
    return generate_keterangan(keluhan, obat)

# ==============================
# KETERANGAN OBAT (FALLBACK)
# ==============================

def generate_keterangan(
    keluhan,
    obat
):

    knowledge_base = {
        "paracetamol": {
            "golongan": "Analgesik & Antipiretik",
            "fungsi": "menurunkan demam dan meredakan rasa nyeri"
        },
        "amoxicillin": {
            "golongan": "Antibiotik",
            "fungsi": "mengobati infeksi akibat bakteri"
        },
        "omeprazole": {
            "golongan": "Penghambat Pompa Proton",
            "fungsi": "mengurangi kadar asam berlebih di lambung"
        },
        "amlodipin": {
            "golongan": "Antihipertensi",
            "fungsi": "menurunkan tekanan darah tinggi agar tetap stabil"
        },
        "hydrocortisone": {
            "golongan": "Kortikosteroid",
            "fungsi": "meredakan peradangan, gatal-gatal, dan kemerahan pada kulit"
        },
        "ibuprofen": {
            "golongan": "NSAID (Anti-inflamasi)",
            "fungsi": "meredakan nyeri, peradangan, dan bengkak"
        },
        "cetirizine": {
            "golongan": "Antihistamin",
            "fungsi": "meredakan gejala alergi seperti gatal, bersin, dan pilek"
        },
        "oralit": {
            "golongan": "Rehidrasi",
            "fungsi": "menggantikan cairan tubuh dan elektrolit yang hilang akibat diare"
        }
    }

    obat_key = str(obat).lower().strip()

    if obat_key in knowledge_base:
        info = knowledge_base[obat_key]
        return (
            f"Berdasarkan keluhan yang Anda rasakan ({keluhan}), obat {obat} "
            f"dipilih karena kemampuannya sebagai {info['golongan']} yang efektif untuk {info['fungsi']}. "
            f"Rekomendasi ini didasarkan pada kecocokan gejala Anda dengan data penanganan pasien sebelumnya di klinik kami."
        )

    return (
        f"Obat {obat} direkomendasikan karena memiliki kecocokan pola yang tinggi dengan keluhan "
        f"pasien-pasien sebelumnya yang memiliki gejala serupa. Obat ini berfungsi untuk membantu "
        f"proses pemulihan kondisi kesehatan Anda sesuai dengan standar penanganan di klinik."
    )
    
# ==============================
# DASHBOARD
# ==============================

@app.route('/')
def dashboard():

    df = load_data()

    obat_labels, obat_values = (
        get_chart_data(df)
    )

    return render_template(
        "dashboard.html",

        hasil_list=None,

        total_data=len(df),

        jumlah_kelas_obat=len(
            label_encoder.classes_
        ),

        jumlah_obat=len(
            dataset_model[
                'obat_utama'
            ].unique()
        ),

        obat_labels=obat_labels,

        obat_values=obat_values,

        keluhan_master=
        generate_keluhan_master()
    )


@app.route('/pasien')

def pasien():

    df = load_data()

    data = (

        df.to_dict(
            orient='records'
        )

        if not df.empty

        else []
    )

    return render_template(

        "data_pasien.html",

        data=data
    )

# ==============================
# LAPORAN
# ==============================

@app.route('/laporan')
def laporan():

    df = load_data()

    if df.empty:

        return render_template(

            "laporan.html",

            gender_stat={},

            keluhan_stat={},

            obat_stat={},

            total=0
        )

    return render_template(

        "laporan.html",

        gender_stat=
        df['gender']
        .value_counts()
        .to_dict(),

        keluhan_stat=
        df['keluhan']
        .value_counts()
        .head(10)
        .to_dict(),

        obat_stat=
        df['obat']
        .value_counts()
        .to_dict(),

        total=len(df)
    )
# ==============================
# PRINT LAPORAN STATISTIK
# ==============================

@app.route('/print_laporan')
def print_laporan():

    df = load_data()

    if df.empty:

        return render_template(
            "laporan_print.html",
            gender_stat={},
            keluhan_stat={},
            obat_stat={},
            total=0,
            tanggal_cetak=datetime.now().strftime(
                "%d-%m-%Y"
            )
        )

    return render_template(
        "laporan_print.html",
        gender_stat=df['gender']
        .value_counts()
        .to_dict(),

        keluhan_stat=df['keluhan']
        .value_counts()
        .head(10)
        .to_dict(),

        obat_stat=df['obat']
        .value_counts()
        .to_dict(),

        total=len(df),

        tanggal_cetak=datetime.now().strftime(
            "%d-%m-%Y"
        )
    )

# ==============================
# PRINT LAPORAN PASIEN
# ==============================

@app.route('/print/<int:index>')
def print_data(index):

    import json
    df = load_data()

    if df.empty:
        return "Data pasien kosong"

    if index < 0 or index >= len(df):
        return "Data pasien tidak ditemukan"

    p = df.iloc[index]

    # ==========================
    # AMBIL DATA HISTORI
    # ==========================

    keluhan_text = str(p.get('keluhan', '-'))
    obat_utama = str(p.get('obat', '-'))
    pendamping_text = str(p.get('obat_pendamping', '-'))
    
    # Cek apakah ada data detail (JSON)
    hasil_detail_raw = p.get('hasil_detail')
    
    if pd.notna(hasil_detail_raw) and str(hasil_detail_raw).strip() != "":
        try:
            hasil_list = json.loads(hasil_detail_raw)
        except:
            # Fallback jika JSON korup
            hasil_list = [{
                "keluhan": keluhan_text,
                "top_3": [
                    {
                        "rank": 1,
                        "obat": obat_utama,
                        "obat_pendamping": pendamping_text,
                        "keterangan": generate_keterangan(keluhan_text, obat_utama)
                    }
                ]
            }]
    else:
        # Fallback untuk data lama yang belum punya hasil_detail
        hasil_list = [{
            "keluhan": keluhan_text,
            "top_3": [
                {
                    "rank": 1,
                    "obat": obat_utama,
                    "obat_pendamping": pendamping_text,
                    "keterangan": generate_keterangan(keluhan_text, obat_utama)
                }
            ]
        }]
        
        # Tambahkan top2 dan top3 jika ada di kolom lama
        if pd.notna(p.get('top2')) and str(p.get('top2')) != "-":
             hasil_list[0]["top_3"].append({
                 "rank": 2,
                 "obat": str(p.get('top2')),
                 "obat_pendamping": "-",
                 "keterangan": generate_keterangan(keluhan_text, str(p.get('top2')))
             })
        if pd.notna(p.get('top3')) and str(p.get('top3')) != "-":
             hasil_list[0]["top_3"].append({
                 "rank": 3,
                 "obat": str(p.get('top3')),
                 "obat_pendamping": "-",
                 "keterangan": generate_keterangan(keluhan_text, str(p.get('top3')))
             })

    return render_template(
        "laporan_pasien.html",
        p=p,
        hasil_list=hasil_list
    )

   
@app.route('/predict', methods=['POST'])
def predict():

    from scipy.sparse import hstack, csr_matrix
    import json

    nama = request.form['nama']
    gender = request.form['gender']
    usia = int(request.form['usia'])

    keluhan_list = request.form.getlist('keluhan')

    if len(keluhan_list) == 0:

        return render_template(
            "dashboard.html",
            hasil_list=[],
            error="Pilih minimal 1 keluhan",
            total_data=len(load_data()),
            jumlah_kelas_obat=len(
                label_encoder.classes_
            ),
            jumlah_obat=len(
                dataset_model['obat_utama'].unique()
            ),
            keluhan_master=
            generate_keluhan_master()
        )

    # ==================================
    # PROSES SETIAP KELUHAN SECARA TERPISAH (REQ 1)
    # ==================================
    
    hasil_rekomendasi_lengkap = []

    for keluhan_item in keluhan_list:
        keluhan_bersih = clean_keluhan(keluhan_item)
        gender_encode = 1 if gender == "laki-laki" else 0
        
        X_text = tfidf.transform([keluhan_bersih])
        fitur_tambahan = csr_matrix([[usia, gender_encode]])
        X_input = hstack([X_text, fitur_tambahan])

        # Prediksi Top 3
        probabilitas = model_rf.predict_proba(X_input)[0]
        top_idx = np.argsort(probabilitas)[-3:][::-1]
        top_obat = label_encoder.inverse_transform(top_idx)
        
        rekomendasi_per_keluhan = {
            "keluhan": keluhan_item,
            "top_3": []
        }

        # Cari histori keluhan yang mirip di dataset untuk mencari obat pendamping
        kata_kunci = "|".join(keluhan_bersih.split())
        histori_keluhan = dataset_model[
            dataset_model['keluhan']
            .astype(str)
            .str.lower()
            .str.contains(kata_kunci, regex=True, na=False)
        ]

        for i in range(len(top_obat)):
            obat = top_obat[i]
            prob = round(probabilitas[top_idx[i]] * 100, 2)
            
            # Cari obat pendamping dari histori (REQ 2)
            obat_pendamping = "-"
            histori_spesifik = histori_keluhan[
                histori_keluhan['obat_utama'].str.lower() == obat.lower()
            ]
            
            if len(histori_spesifik) > 0:
                resep_asli = str(histori_spesifik.iloc[0]['pengobatan/resep'])
                # Bersihkan string resep dari aturan minum (angka dan x)
                # Contoh: "amoxicillin 2x1" -> "amoxicillin"
                def clean_obat_name(name):
                    name = re.sub(r'\d+x\d+', '', name) # hapus 2x1, 3x1, dll
                    name = re.sub(r'\d+', '', name) # hapus angka sisa
                    name = name.replace('tablet', '').replace('kapsul', '').replace('syrup', '')
                    return name.strip().lower()

                daftar_resep = [x.strip() for x in resep_asli.split(',') if x.strip()]
                
                if len(daftar_resep) > 1:
                    top_obat_lower = [o.lower().strip() for o in top_obat]
                    # Ambil semua kecuali yang masuk dalam daftar Top 1-3
                    pendamping_list = []
                    for r in daftar_resep:
                        r_clean = clean_obat_name(r)
                        is_top_3 = False
                        for top_o in top_obat_lower:
                            if top_o in r_clean or r_clean in top_o:
                                is_top_3 = True
                                break
                        if not is_top_3 and r_clean != "":
                            pendamping_list.append(r)
                            
                    if pendamping_list:
                        obat_pendamping = ", ".join(pendamping_list)
            
            rekomendasi_per_keluhan["top_3"].append({
                "rank": i + 1,
                "obat": obat,
                "prob": prob,
                "obat_pendamping": obat_pendamping,
                "keterangan": get_ai_explanation(keluhan_item, obat) if i == 0 else generate_keterangan(keluhan_item, obat) # AI desc for Top 1 (REQ 3)
            })
            
        hasil_rekomendasi_lengkap.append(rekomendasi_per_keluhan)

    # ==================================
    # SIMPAN HISTORI
    # ==================================
    
    # Untuk kompatibilitas ke belakang dan ringkasan, simpan Top 1 dari keluhan pertama
    top1_all = hasil_rekomendasi_lengkap[0]["top_3"][0]["obat"]
    top2_all = hasil_rekomendasi_lengkap[0]["top_3"][1]["obat"] if len(hasil_rekomendasi_lengkap[0]["top_3"]) > 1 else "-"
    top3_all = hasil_rekomendasi_lengkap[0]["top_3"][2]["obat"] if len(hasil_rekomendasi_lengkap[0]["top_3"]) > 2 else "-"
    
    # Serialkan hasil lengkap ke JSON untuk fitur print detail
    hasil_json = json.dumps(hasil_rekomendasi_lengkap)

    histori_baru = pd.DataFrame([{
        "id": str(uuid.uuid4())[:8],
        "tanggal": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
        "nama": nama,
        "gender": gender,
        "usia": usia,
        "keluhan": ", ".join(keluhan_list),
        "top1": top1_all,
        "top2": top2_all,
        "top3": top3_all,
        "obat": top1_all, # Obat utama adalah Top 1
        "obat_pendamping": hasil_rekomendasi_lengkap[0]["top_3"][0]["obat_pendamping"],
        "hasil_detail": hasil_json # Kolom baru untuk menyimpan detail Top 1-3 per keluhan
    }])

    if os.path.exists(FILE_HISTORI):
        histori_lama = pd.read_csv(FILE_HISTORI)
        histori_baru = pd.concat([histori_lama, histori_baru], ignore_index=True)

    histori_baru.to_csv(FILE_HISTORI, index=False)

    return render_template(
        "dashboard.html",
        hasil_list=hasil_rekomendasi_lengkap,
        nama=nama,
        gender=gender,
        usia=usia,
        keluhan=", ".join(keluhan_list),
        total_data=len(load_data()),
        jumlah_kelas_obat=len(label_encoder.classes_),
        jumlah_obat=len(dataset_model['obat_utama'].unique()),
        keluhan_master=generate_keluhan_master()
    )

# ==============================
# MAIN
# ==============================

if __name__ == '__main__':

    app.run(
        debug=True
    )