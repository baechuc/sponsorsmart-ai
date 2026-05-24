import streamlit as st
import re, os, torch
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle

# ── Konfigurasi Halaman ──────────────────────────────
st.set_page_config(
    page_title="SponsorSmart AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }

    /* ── LIGHT MODE ── */
    .stApp { background-color: #f0f2f5; }
    section[data-testid="stSidebar"] {
        background-color: #f8f9fb !important;
        border-right: 1px solid #e2e5ec;
    }

    /* ── DARK MODE ── */
    [data-theme="dark"] .stApp { background-color: #0f1117; }
    [data-theme="dark"] section[data-testid="stSidebar"] {
        background-color: #1a1d27 !important;
        border-right: 1px solid #2d3143;
    }
    [data-theme="dark"] .metric-card {
        background: #1e2130 !important;
        border-color: #2d3143 !important;
        border-top-color: #6b8cae !important;
    }
    [data-theme="dark"] .metric-card p { color: #9ca3af !important; }
    [data-theme="dark"] .stage-box { background: #1e2130 !important; border-color: #2d3143 !important; }
    [data-theme="dark"] .stage-pass { background: #0d2b1e !important; border-color: #34d399 !important; }
    [data-theme="dark"] .stage-fail { background: #2b0d0d !important; border-color: #f87171 !important; }
    [data-theme="dark"] .layak-badge      { background: #0d2b1e !important; color: #6ee7b7 !important; border-color: #34d399 !important; }
    [data-theme="dark"] .tidak-layak-badge { background: #2b0d0d !important; color: #fca5a5 !important; border-color: #f87171 !important; }
    [data-theme="dark"] .review-badge     { background: #2b1f0a !important; color: #fcd34d !important; border-color: #f59e0b !important; }
    [data-theme="dark"] [data-testid="stExpander"] {
        background: #1e2130 !important;
        border-color: #2d3143 !important;
    }
    [data-theme="dark"] hr { border-color: #2d3143 !important; }
    [data-theme="dark"] h1,
    [data-theme="dark"] h2,
    [data-theme="dark"] h3,
    [data-theme="dark"] h4 { color: #f3f4f6 !important; }
    [data-theme="dark"] p { color: #9ca3af !important; }
    [data-theme="dark"] .stMarkdown p { color: #9ca3af !important; }
    [data-theme="dark"] label { color: #9ca3af !important; }
    [data-theme="dark"] .subtitle { color: #6b7280 !important; }
    [data-theme="dark"] .stMetric [data-testid="stMetricValue"] { color: #f3f4f6 !important; }
    [data-theme="dark"] .stMetric label { color: #6b7280 !important; }

    /* ── SHARED (light + dark) ── */
    .main-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(135deg, #3d5a80, #6b8cae);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.4rem; letter-spacing: -0.02em;
    }
    .subtitle { text-align: center; color: #6b7280; font-size: 0.95rem; margin-bottom: 1rem; }

    .metric-card {
        background: #ffffff; border-radius: 14px;
        padding: 1.2rem 1rem; text-align: center;
        border: 1px solid #e2e5ec;
        border-top: 3px solid #3d5a80;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 6px rgba(61,90,128,0.07);
        transition: box-shadow 0.2s;
    }
    .metric-card p { color: #4b5563 !important; font-size: 0.88rem; }

    .layak-badge {
        background: #ecfdf5; color: #065f46;
        border: 1.5px solid #6ee7b7;
        padding: 0.7rem 2rem; border-radius: 50px;
        font-size: 1.3rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
    }
    .tidak-layak-badge {
        background: #fef2f2; color: #7f1d1d;
        border: 1.5px solid #fca5a5;
        padding: 0.7rem 2rem; border-radius: 50px;
        font-size: 1.3rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
    }
    .review-badge {
        background: #fffbeb; color: #78350f;
        border: 1.5px solid #fcd34d;
        padding: 0.7rem 2rem; border-radius: 50px;
        font-size: 1.3rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
    }

    .stage-box { border: 1.5px solid #e2e5ec; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1rem; background: #ffffff; }
    .stage-pass { border-color: #6ee7b7; background: #ecfdf5; }
    .stage-fail { border-color: #fca5a5; background: #fef2f2; }
    .stage-skip { border-color: #d1d5db; background: #f9fafb; opacity: 0.65; }

    .stButton > button {
        background: linear-gradient(135deg, #3d5a80, #6b8cae) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important;
        font-size: 1rem !important; letter-spacing: 0.01em !important;
        box-shadow: 0 4px 12px rgba(61,90,128,0.22) !important;
        transition: opacity 0.2s !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }

    .stMetric label { color: #6b7280 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1f2937 !important; font-weight: 700 !important; }

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e5ec !important;
        border-radius: 10px !important;
    }

    hr { border-color: #e2e5ec !important; }
    h1,h2,h3,h4 { color: #1f2937 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
    p { color: #4b5563 !important; }
    .stMarkdown p { color: #4b5563 !important; }
    label { color: #4b5563 !important; }
    .stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:var(--background-color, #ffffff);border:1px solid rgba(128,128,128,0.2);border-radius:16px;
            padding:2rem 1.5rem 1.5rem;margin-bottom:1.5rem;
            box-shadow:0 2px 12px rgba(61,90,128,0.07);">
    <p class="main-title">🎯 SponsorSmart AI</p>
    <p class="subtitle">Sistem Pendukung Keputusan — Penilaian Kelayakan Proposal Sponsorship</p>
    <div style="text-align:center;margin-top:0.8rem;">
        <span style="background:#dbeafe;color:#1e40af;border:1px solid #93c5fd;
                     border-radius:20px;font-size:0.72rem;font-weight:600;
                     padding:0.25rem 0.9rem;letter-spacing:0.05em;">
            MACHINE LEARNING · NLP · SVM + TF-IDF
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────
HF_REPO_ID    = "calpycbara/sponsorsmart-indobert"
OCR_THRESHOLD = 50

# Rubric gate threshold — proposal harus melewati ini sebelum ke AI
RUBRIC_PASS_THRESHOLD = 3   # dari 5

# ── HuggingFace Token ────────────────────────────────
HF_TOKEN = st.secrets.get("HUGGINGFACE_TOKEN", "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")
from huggingface_hub import login as hf_login
hf_login(token=HF_TOKEN)

# ── Rubric Keywords ───────────────────────────────────
RUBRIC_KEYWORDS = {
    "Exposure": [
        r"\b\d[\d.]*\s*(?:peserta|pengunjung|penonton|orang|hadirin)\b",
        r"\b(?:instagram|youtube|tiktok|facebook|twitter|linkedin)\b",
        r"\b(?:media sosial|live streaming|publikasi|promosi|liputan)\b",
        r"\b(?:followers|subscriber|views|reach|tayangan)\b",
        r"\b(?:poster|banner|flyer|spanduk|baliho|pamflet)\b",
    ],
    "Relevansi": [
        r"\b(?:teknologi|digital|startup|inovasi|bisnis|korporat)\b",
        r"\b(?:pendidikan|universitas|kampus|mahasiswa|akademis)\b",
        r"\b(?:olahraga|kesehatan|lifestyle|sport|atletik)\b",
        r"\b(?:seni|budaya|musik|festival|entertainment|hiburan)\b",
        r"\b(?:sesuai|relevan|sejalan|mendukung)\s+(?:dengan|visi|misi|brand)\b",
    ],
    "Benefit": [
        r"\b(?:logo|branding|brand awareness|visibilitas)\b",
        r"\b(?:logo placement|official sponsor|title sponsor|presenting sponsor)\b",
        r"\b(?:booth|stand|pameran|aktivasi brand|exhibition)\b",
        r"\b(?:mention|endorse|konten sponsor|publikasi sponsor)\b",
        r"\b(?:tiket gratis|vip|akses eksklusif|goodie bag|merchandise)\b",
    ],
    "Anggaran": [
        r"(?:rp|idr)\.?\s*[\d.,]{6,15}",
        r"(?:anggaran|biaya|dana|investasi|kontribusi|nominal)\s*[:=]?\s*(?:rp)?[\d.,]{6,15}",
        r"\b(?:paket)\s+(?:platinum|gold|silver|bronze|diamond)\b",
        r"\b(?:rab|rencana anggaran biaya|kebutuhan dana)\b",
    ],
    "Kredibilitas": [
        r"\b(?:himpunan|bem|osis|komunitas|lembaga|yayasan|pt\.|cv\.|organisasi)\b",
        r"\b(?:susunan panitia|struktur organisasi|divisi|sie)\b",
        r"\b(?:ketua|sekretaris|bendahara|koordinator|penanggung jawab)\b",
        r"\b(?:timeline|jadwal|rundown|susunan acara|agenda)\b",
        r"\b(?:contact person|cp|whatsapp|narahubung|email|telp)\b",
    ]
}

THRESHOLDS = {
    "Exposure": 2, "Relevansi": 1, "Benefit": 2,
    "Anggaran": 1, "Kredibilitas": 3
}

SARAN = {
    "Exposure"    : "Tambahkan data audiens (jumlah peserta, jangkauan media sosial, platform publikasi).",
    "Relevansi"   : "Jelaskan keterkaitan acara dengan industri/brand sponsor secara eksplisit.",
    "Benefit"     : "Cantumkan benefit konkret: logo placement, booth, mention medsos, goodie bag, dll.",
    "Anggaran"    : "Sertakan nominal jelas (Rp) dan paket sponsorship (Gold/Silver/Bronze).",
    "Kredibilitas": "Lengkapi identitas organisasi, susunan panitia, timeline, rundown, dan kontak PIC.",
}

VAR_ICONS = {
    "Exposure":"📡", "Relevansi":"🎯", "Benefit":"🎁",
    "Anggaran":"💰", "Kredibilitas":"🏛️"
}

# ── Custom SVM Pipeline ───────────────────────────────
class ThresholdSVMPipeline:
    def __init__(self, tfidf, svm, threshold=0.6):
        self.tfidf     = tfidf
        self.svm       = svm
        self.threshold = threshold
        self.classes_  = svm.classes_

    def predict(self, texts):
        probs     = self.predict_proba(texts)
        layak_idx = list(self.classes_).index("Layak")
        return ["Layak" if p[layak_idx] >= self.threshold
                else "Tidak Layak" for p in probs]

    def predict_proba(self, texts):
        return self.svm.predict_proba(self.tfidf.transform(texts))

# ── Fungsi Load Model ─────────────────────────────────
@st.cache_resource
def load_bert_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_REPO_ID)
        model     = AutoModelForSequenceClassification.from_pretrained(HF_REPO_ID)
        model.eval()
        return tokenizer, model
    except Exception:
        return None, None

def load_svm_model():
    try:
        from huggingface_hub import hf_hub_download
        svm_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="svm_model.pkl",
            token=HF_TOKEN
        )
        with open(svm_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[SVM Load Error] {e}")
        return None

# ── Fungsi Ekstraksi PDF ──────────────────────────────
def extract_text(uploaded_file) -> tuple:
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    full_text = ""
    method    = "pdfplumber"

    try:
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages[:10]:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except:
        pass

    if len(full_text.strip()) < OCR_THRESHOLD:
        method = "ocr"
        try:
            images = convert_from_path(tmp_path, first_page=1, last_page=3)
            for img in images:
                full_text += pytesseract.image_to_string(img, lang="ind") + "\n"
        except:
            full_text = ""
            method    = "failed"

    try:
        os.unlink(tmp_path)
    except:
        pass

    return full_text.strip(), method

# ── Fungsi Rubric Scoring ─────────────────────────────
def score_rubric(text: str) -> dict:
    """
    Tahap 1: Filter rubric berbasis keyword.
    Proposal yang tidak lolos (total < RUBRIC_PASS_THRESHOLD) langsung
    ditolak — AI tidak akan dipanggil.
    """
    text_lower = text.lower()
    scores, evidences = {}, {}

    for var, patterns in RUBRIC_KEYWORDS.items():
        matched = []
        for p in patterns:
            found = re.findall(p, text_lower, re.IGNORECASE)
            if found:
                matched.extend(found[:2])
        unique_matched = list(set(matched))
        scores[var]    = 1 if len(unique_matched) >= THRESHOLDS[var] else 0
        evidences[var] = unique_matched[:3] if unique_matched else []

    total    = sum(scores.values())
    passed   = total >= RUBRIC_PASS_THRESHOLD

    return {
        "scores"  : scores,
        "evidences": evidences,
        "total"   : total,
        "passed"  : passed,          # True = lolos filter, lanjut ke AI
    }

# ── Fungsi Prediksi IndoBERT ──────────────────────────
def predict_bert(text: str, tokenizer, model) -> dict:
    """Tahap 2: AI hanya dipanggil jika rubric sudah lolos."""
    if tokenizer is None or model is None:
        return {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
    inputs = tokenizer(
        text, max_length=256, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1).squeeze().tolist()

    label_idx  = int(torch.argmax(outputs.logits))
    labels_map = {0: "Tidak Layak", 1: "Layak"}
    return {
        "label"     : labels_map[label_idx],
        "confidence": max(probs),
        "prob_layak": probs[1],
        "prob_tidak": probs[0]
    }


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/target.png", width=80)
    st.title("ℹ️ Info Sistem")

    st.divider()
    st.markdown("### 🔌 Status Model")

    with st.spinner("Cek koneksi model..."):
        tokenizer_check, model_check = load_bert_model()
        svm_check                    = load_svm_model()

    st.markdown(f'<div style="background:rgba(128,128,128,0.08);border:1px solid rgba(128,128,128,0.18);border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;font-size:0.85rem;">{"🟢" if svm_check else "🔴"} <b>Model Utama</b> — {"Siap" if svm_check else "Tidak ditemukan"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:rgba(128,128,128,0.08);border:1px solid rgba(128,128,128,0.18);border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;font-size:0.85rem;">{"🟢" if tokenizer_check else "🔴"} <b>Model Pembanding</b> — {"Siap" if tokenizer_check else "Tidak ditemukan"}</div>', unsafe_allow_html=True)

    if tokenizer_check is None and svm_check is None:
        st.warning("⚠️ Semua model offline.\nHanya Penilaian Rubrikasi yang aktif.")

    st.divider()
    st.markdown("### 🔁 Alur Penilaian")
    st.markdown("""
    Sistem ini bekerja dalam **2 tahap berurutan**:

    **Tahap 1 — Penilaian Rubrikasi**
    Proposal dinilai dari 5 aspek keyword.
    Skor < 3 → ❌ **Langsung ditolak**, AI tidak dipanggil.

    **Tahap 2 — Penilaian AI**
    Dua model AI dijalankan otomatis.
    Model Utama menentukan keputusan final.

    **Hasil Akhir:**
    - ✅ Direkomendasikan = Rubrikasi ✓ + AI ✓
    - ⚠️ Perlu Ditinjau Ulang = Rubrikasi ✓, AI ragu
    - ❌ Tidak Direkomendasikan = Rubrikasi gagal
    """)


# ── Main Content ──────────────────────────────────────
st.subheader("📤 Upload Proposal Sponsorship")
uploaded_file = st.file_uploader(
    "Upload file PDF proposal sponsorship",
    type=["pdf"],
    help="Format PDF, maksimal 50MB"
)

if uploaded_file is not None:

    st.success(f"✅ File berhasil diupload: **{uploaded_file.name}**")

    with st.spinner("🔍 Mengekstrak teks dari PDF..."):
        full_text, method = extract_text(uploaded_file)

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.expander("👁️ Preview Teks Hasil Ekstraksi", expanded=False):
            if full_text:
                preview = full_text[:2000] + "..." if len(full_text) > 2000 else full_text
                st.text_area("Teks Proposal:", preview, height=250)
            else:
                st.error("Teks tidak dapat diekstrak dari file ini.")
    with col2:
        method_label = {"pdfplumber": "PDFPLUMBER", "ocr": "OCR", "failed": "GAGAL"}
        st.metric("Metode Ekstraksi", method_label.get(method, method.upper()))
        st.metric("Jumlah Kata",      len(full_text.split()))
        st.metric("Jumlah Karakter",  len(full_text))

    if not full_text or len(full_text.split()) < 30:
        st.error("⚠️ Teks proposal terlalu sedikit untuk dianalisis. "
                 "Coba upload file PDF yang lebih lengkap.")
        st.stop()

    st.divider()

    if st.button("🚀 Analisis Proposal", type="primary", use_container_width=True):
        st.divider()
        st.subheader("📊 Hasil Analisis")

        # ════════════════════════════════════════════
        # TAHAP 1: FILTER RUBRIC
        # ════════════════════════════════════════════
        st.markdown("### Tahap 1 — Penilaian Rubrikasi")

        with st.spinner("📋 Menghitung skor penilaian rubrikasi..."):
            rubric_result = score_rubric(full_text)

        # Tampilkan kartu skor per variabel
        cols = st.columns(5)
        for i, (var, score) in enumerate(rubric_result["scores"].items()):
            with cols[i]:
                color       = "#ecfdf5" if score == 1 else "#fef2f2"
                card_border = "#6ee7b7" if score == 1 else "#fca5a5"
                status      = "✓" if score == 1 else "✗"
                st.markdown(f"""
                <div style="background:{color};padding:12px;border-radius:12px;text-align:center;border:1.5px solid {card_border};">
                    <div style="font-size:1.5rem">{VAR_ICONS[var]}</div>
                    <div style="font-weight:bold;font-size:0.9rem">{var}</div>
                    <div style="font-size:1.8rem;font-weight:800">{status}</div>
                    <div style="font-size:0.8rem">Skor: {score}/1</div>
                </div>
                """, unsafe_allow_html=True)


        terpenuhi       = [v for v, s in rubric_result["scores"].items() if s == 1]
        tidak_terpenuhi = [v for v, s in rubric_result["scores"].items() if s == 0]

        rubric_color = "#ecfdf5" if rubric_result["passed"] else "#fef2f2"
        rubric_icon  = "✅" if rubric_result["passed"] else "❌"
        rubric_text  = (
            f"Penilaian Rubrikasi lolos ({rubric_result['total']}/5 variabel terpenuhi) "
            f"— lanjut ke verifikasi AI"
            if rubric_result["passed"] else
            f"Penilaian Rubrikasi gagal ({rubric_result['total']}/5 variabel terpenuhi) "
            f"— proposal ditolak tanpa perlu AI"
        )
        rubric_border = "#6ee7b7" if rubric_result["passed"] else "#fca5a5"
        st.markdown(
            f'<div style="background:{rubric_color};padding:0.8rem 1.2rem;border:1.5px solid {rubric_border};'
            f'border-radius:10px;font-weight:bold;font-size:1rem;">'
            f'{rubric_icon} {rubric_text}</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔍 Detail Bukti per Variabel (Penilaian Rubrikasi)"):
            for var, evid in rubric_result["evidences"].items():
                icon     = "✅" if rubric_result["scores"][var] == 1 else "❌"
                evid_str = str(evid) if evid else "Tidak ditemukan indikator"
                st.markdown(f"{icon} **{var}**: {evid_str}")

        # ════════════════════════════════════════════
        # TAHAP 2: VERIFIKASI AI  (hanya jika rubric lolos)
        # ════════════════════════════════════════════
        st.divider()
        st.markdown("### Tahap 2 — Verifikasi AI")

        # Inisialisasi default
        bert_result = {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
        svm_label   = None
        svm_conf    = 0
        ai_available = False

        if not rubric_result["passed"]:
            # Rubric gagal → tampilkan info, skip AI
            st.markdown(
                '<div class="stage-skip" style="border:1.5px solid rgba(128,128,128,0.3);border-radius:12px;'
                'padding:1rem;background:rgba(128,128,128,0.07);opacity:0.8;">'
                '⏭️ <b>AI tidak dijalankan</b> — proposal sudah gugur di Tahap 1 (Penilaian Rubrikasi gagal).'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            # Rubric lolos → jalankan kedua model otomatis
            svm_col, bert_col = st.columns(2)

            # Model Utama: SVM — penentu keputusan
            with svm_col:
                st.markdown("**⭐ Model Utama** — Penentu Keputusan")
                with st.spinner("⚙️ Menilai proposal..."):
                    svm_model = load_svm_model()
                    if svm_model:
                        ai_available = True
                        svm_label    = svm_model.predict([full_text])[0]
                        svm_prob     = svm_model.predict_proba([full_text])[0]
                        svm_conf     = int(max(svm_prob) * 100)
                        svm_display  = "Direkomendasikan" if svm_label == "Layak" else "Tidak Direkomendasikan"
                        badge        = "layak-badge" if svm_label == "Layak" else "tidak-layak-badge"
                        st.markdown(f'<span class="{badge}">{svm_display}</span>', unsafe_allow_html=True)
                        st.progress(svm_conf)
                        st.caption(f"Tingkat Keyakinan: {svm_conf}%")
                    else:
                        st.warning(f"⚠️ Model Utama tidak tersedia. Pastikan `svm_model.pkl` diupload ke `{HF_REPO_ID}`")

            # Model Pembanding: IndoBERT — hanya referensi
            with bert_col:
                st.markdown("**🔬 Model Pembanding** — Referensi")
                with st.spinner("🧠 Menilai proposal (pembanding)..."):
                    tokenizer, bert_model = load_bert_model()
                    bert_result           = predict_bert(full_text, tokenizer, bert_model)
                    if bert_result["label"]:
                        bert_display = "Direkomendasikan" if bert_result["label"] == "Layak" else "Tidak Direkomendasikan"
                        badge        = "layak-badge" if bert_result["label"] == "Layak" else "tidak-layak-badge"
                        st.markdown(f'<span class="{badge}">{bert_display}</span>', unsafe_allow_html=True)
                        conf = int(bert_result["confidence"] * 100)
                        st.progress(conf)
                        st.caption(f"Tingkat Keyakinan: {conf}%")
                    else:
                        st.info("Model pembanding tidak tersedia.")

            if ai_available and bert_result["label"] and svm_label != bert_result["label"]:
                st.info("ℹ️ Kedua model berbeda pendapat — keputusan final mengikuti Model Utama.")


        # KEPUTUSAN FINAL
        # ════════════════════════════════════════════
        st.divider()
        st.markdown("### ⚖️ Keputusan Final")

        # --- Logika keputusan 2 tahap ---
        # Rubric gagal → langsung Tidak Direkomendasikan
        if not rubric_result["passed"]:
            final_label   = "Tidak Direkomendasikan"
            final_reason  = "rubric_failed"

        # Rubric lolos, tapi tidak ada model AI → fallback ke rubric
        elif not ai_available:
            final_label   = "Direkomendasikan"
            final_reason  = "rubric_only"

        # Rubric lolos + ada hasil AI → SVM (Model Utama) yang menentukan
        else:
            if svm_label == "Layak":
                final_label  = "Direkomendasikan"
                final_reason = "rubric_pass_ai_agree"
            else:
                final_label  = "Perlu Ditinjau Ulang"
                final_reason = "rubric_pass_ai_disagree"

        # ── Tampilan badge final ──────────────────────
        if final_reason == "rubric_pass_ai_agree":
            badge_class = "layak-badge"
            label_text  = "✅ DIREKOMENDASIKAN"
        elif final_reason == "rubric_only":
            badge_class = "layak-badge"
            label_text  = "✅ DIREKOMENDASIKAN"
        elif final_reason == "rubric_pass_ai_disagree":
            badge_class = "review-badge"
            label_text  = "⚠️ PERLU DITINJAU ULANG"
        else:  # rubric_failed
            badge_class = "tidak-layak-badge"
            label_text  = "❌ TIDAK DIREKOMENDASIKAN"

        col_dec, col_reason = st.columns([1, 2])
        with col_dec:
            st.markdown(
                f'<div style="text-align:center"><span class="{badge_class}">'
                f'{label_text}</span></div>',
                unsafe_allow_html=True
            )

        with col_reason:
            st.markdown("**📋 Penjelasan Hasil:**")

            if final_reason == "rubric_failed":
                st.error(
                    f"❌ Penilaian Rubrikasi gagal: hanya {rubric_result['total']} dari 5 aspek terpenuhi "
                    f"(minimum {RUBRIC_PASS_THRESHOLD}).\n\n"
                    f"Penilaian AI tidak dijalankan karena proposal belum memenuhi syarat dasar."
                )

            elif final_reason == "rubric_only":
                st.success(f"✅ Penilaian Rubrikasi lolos ({rubric_result['total']}/5 aspek terpenuhi).")
                st.info("ℹ️ Model AI tidak tersedia — keputusan berdasarkan rubrikasi saja.")

            elif final_reason == "rubric_pass_ai_agree":
                st.success(f"✅ Penilaian Rubrikasi lolos ({rubric_result['total']}/5 aspek terpenuhi).")
                st.success("✅ Model Utama mengkonfirmasi proposal layak mendapatkan sponsorship.")
                st.info("💡 Proposal sudah lengkap dan dinilai baik. Silakan ajukan ke pihak sponsor.")

            elif final_reason == "rubric_pass_ai_disagree":
                st.success(f"✅ Penilaian Rubrikasi lolos ({rubric_result['total']}/5 aspek terpenuhi).")
                st.warning(
                    "⚠️ Model Utama menilai isi proposal masih perlu diperkuat.\n\n"
                    "Proposal sudah cukup lengkap secara struktur, namun isi perlu direvisi "
                    "sebelum diajukan ke pihak sponsor."
                )

            if terpenuhi:
                st.success(f"✅ Aspek terpenuhi: {', '.join(terpenuhi)}")
            if tidak_terpenuhi:
                st.error(f"❌ Aspek yang perlu dilengkapi: {', '.join(tidak_terpenuhi)}")

        if tidak_terpenuhi:
            with st.expander("💡 Saran Perbaikan Proposal"):
                for var in tidak_terpenuhi:
                    st.markdown(f"**{var}:** {SARAN.get(var, '')}")

else:
    # ── Landing Page ──────────────────────────────────
    st.info("👆 Upload file PDF proposal sponsorship untuk memulai analisis.")

    st.markdown("#### Alur Penilaian 2 Tahap")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>📋</h2><b>Tahap 1: Penilaian Rubrikasi</b>
            <p>5 aspek proposal diperiksa.<br>Skor &lt; 3 → langsung ditolak.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>🤖</h2><b>Tahap 2: Penilaian AI</b>
            <p>Dua model AI dijalankan otomatis.<br>Model Utama menentukan keputusan.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>⚖️</h2><b>Hasil Akhir</b>
            <p>Direkomendasikan jika rubrikasi ✅ <b>dan</b> AI ✅.<br>Konflik → perlu ditinjau ulang.</p>
        </div>""", unsafe_allow_html=True)
