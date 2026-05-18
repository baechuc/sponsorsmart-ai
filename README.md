# 🎯 SponsorSmart AI

> Sistem Pendukung Keputusan (SPK) otomatis berbasis Deep Learning  
> untuk klasifikasi kelayakan proposal sponsorship tidak terstruktur.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red)
![IndoBERT](https://img.shields.io/badge/Model-IndoBERT-orange)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)

---

## 📌 Latar Belakang

Perusahaan sering menerima proposal sponsorship dalam jumlah besar dalam format PDF
tidak terstruktur. Proses review yang selama ini manual bersifat subjektif, lambat,
dan tidak konsisten.

**SponsorSmart AI** hadir sebagai solusi yang:
- Mengekstrak teks dari PDF secara otomatis (pdfPlumber + OCR)
- Menilai kelayakan berdasarkan rubric 5 variabel
- Mengklasifikasikan proposal menggunakan IndoBERT fine-tuned
- Menyajikan reasoning yang transparan dan akuntabel

---

## 🔬 Metode

### Pipeline Sistem
```
PDF Upload
  → Ekstraksi Teks (pdfPlumber + Tesseract OCR)
  → Text Cleaning & Preprocessing
  → NLP Preprocessing (Tokenizer IndoBERT)
  → IndoBERT Fine-tuning / SVM + TF-IDF
  → Rubric Scoring (5 variabel)
  → Klasifikasi Output (Layak / Tidak Layak)
  → Dashboard Interaktif
```

### Rubric Penilaian

| Variabel | Definisi | Skor |
|---|---|---|
| 📡 **Exposure** | Kejelasan jangkauan audiens & media publikasi | 0 / 1 |
| 🎯 **Relevansi** | Kesesuaian acara dengan profil sponsor | 0 / 1 |
| 🎁 **Benefit** | Keuntungan konkret untuk sponsor | 0 / 1 |
| 💰 **Anggaran** | Kejelasan nominal & rincian biaya | 0 / 1 |
| 🏛️ **Kredibilitas** | Profesionalitas penyelenggara | 0 / 1 |

**Aturan Label:** Total Skor ≥ 3 → **Layak** | Total Skor < 3 → **Tidak Layak**

### Model

| Model | Deskripsi | Peran |
|---|---|---|
| **IndoBERT** | `indobenchmark/indobert-base-p1` fine-tuned | Model Utama |
| **SVM + TF-IDF** | Traditional ML dengan SMOTE balancing | Baseline |

---

## 📊 Hasil Evaluasi

| Metric | SVM | IndoBERT |
|---|---|---|
| Accuracy | - | - |
| Precision | - | - |
| Recall | - | - |
| F1-Score | - | - |

*(Diisi setelah training selesai)*

---

## 📁 Struktur Repository

```
sponsorsmart-ai/
├── capstone-ds-kel-2.ipynb   ← Notebook utama (training di Kaggle)
├── sponsorsmart_app.py        ← Streamlit web app
├── requirements.txt           ← Dependensi Python
├── .gitignore
└── README.md
```

---

## 🚀 Cara Menjalankan

### 1. Training Model (Kaggle)

1. Buka [Kaggle](https://kaggle.com) → buat Notebook baru
2. Upload dataset PDF sebagai **Kaggle Dataset**
3. Import notebook `capstone-ds-kel-2.ipynb`
4. Aktifkan **GPU T4** di Settings → Accelerator
5. Run All → model otomatis diupload ke HuggingFace

### 2. Jalankan Streamlit App (Lokal)

```bash
# Clone repo
git clone https://github.com/USERNAME/sponsorsmart-ai.git
cd sponsorsmart-ai

# Install dependencies
pip install -r requirements.txt

# Jalankan app
streamlit run sponsorsmart_app.py
```

> **Catatan:** Model dimuat otomatis dari HuggingFace Hub  
> (`calpycbara/sponsorsmart-indobert`). Tidak perlu download manual.

### 3. Jalankan di Google Colab

```python
!pip install streamlit pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("TOKEN_KAMU")
!nohup streamlit run sponsorsmart_app.py --server.port 8501 &
public_url = ngrok.connect(8501)
print(public_url)
```

---

## 🔌 Model di HuggingFace

Model tersimpan di: [`calpycbara/sponsorsmart-indobert`](https://huggingface.co/calpycbara/sponsorsmart-indobert)

File yang diupload:
- `config.json` — konfigurasi model IndoBERT
- `model.safetensors` — bobot model IndoBERT
- `tokenizer_config.json` — konfigurasi tokenizer
- `svm_model.pkl` — pipeline SVM + TF-IDF

> ⚠️ File model tidak disertakan di GitHub karena ukurannya besar.  
> Jalankan notebook di Kaggle untuk melatih ulang jika diperlukan.

---

## 👥 Tim Pengembang — Kelompok 2

| Nama | NIM | Peran |
|---|---|---|
| [Nama 1] | [NIM] | NLP Engineer |
| [Nama 2] | [NIM] | Data Scientist |
| [Nama 3] | [NIM] | ML Engineer |
| [Nama 4] | [NIM] | App Developer |

---

## 🏫 Informasi Capstone

- **Program:** Data Science
- **Institusi:** [Nama Institusi]
- **Tahun:** 2024/2025

---

## 📄 Lisensi

MIT License © 2024 Kelompok 2
