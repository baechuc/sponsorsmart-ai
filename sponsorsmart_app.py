import streamlit as st
import re, os, torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── Custom CSS — Premium Dark Theme ──────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif !important;
    }
    .stApp {
        background: #0a0d14;
    }
    section[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid #1e2533;
    }

    /* ── Header ── */
    .hero-wrapper {
        background: linear-gradient(135deg, #0f1923 0%, #0a0d14 60%, #111827 100%);
        border: 1px solid #1e2a3a;
        border-radius: 20px;
        padding: 2.5rem 2rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: '';
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,179,237,0.12) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-wrapper::after {
        content: '';
        position: absolute;
        bottom: -40px; left: 30px;
        width: 150px; height: 150px;
        background: radial-gradient(circle, rgba(167,243,208,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        background: linear-gradient(90deg, #63b3ed 0%, #9ae6b4 60%, #fbd38d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.4rem;
        line-height: 1.1;
    }
    .subtitle {
        color: #718096;
        font-size: 0.95rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,179,237,0.12);
        border: 1px solid rgba(99,179,237,0.3);
        color: #63b3ed;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Stage Cards (landing) ── */
    .stage-card {
        background: #0f1923;
        border: 1px solid #1e2a3a;
        border-radius: 16px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        position: relative;
        transition: border-color 0.2s;
    }
    .stage-card:hover { border-color: #63b3ed55; }
    .stage-icon { font-size: 2rem; margin-bottom: 0.8rem; }
    .stage-title {
        font-size: 0.95rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 0.5rem;
    }
    .stage-desc { font-size: 0.82rem; color: #718096; line-height: 1.5; }
    .stage-num {
        position: absolute; top: 12px; right: 14px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem; font-weight: 600;
        color: #4a5568;
    }

    /* ── Upload Zone ── */
    .upload-wrapper {
        background: #0f1923;
        border: 1.5px dashed #2d3748;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .upload-wrapper:hover { border-color: #63b3ed55; }

    /* ── Info Pill ── */
    .info-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        background: rgba(99,179,237,0.08);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-size: 0.82rem; color: #a0aec0;
        margin: 0.2rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.7rem;
        margin: 1.5rem 0 1rem;
    }
    .section-badge {
        background: rgba(99,179,237,0.15);
        color: #63b3ed;
        border: 1px solid rgba(99,179,237,0.35);
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.7rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* ── Variabel Cards ── */
    .var-card {
        border-radius: 12px;
        padding: 1rem 0.8rem;
        text-align: center;
        border: 1px solid transparent;
        transition: transform 0.15s;
    }
    .var-card:hover { transform: translateY(-2px); }
    .var-card.pass {
        background: linear-gradient(160deg, #0f2319 0%, #0d1f17 100%);
        border-color: #276749;
    }
    .var-card.fail {
        background: linear-gradient(160deg, #2d1515 0%, #1f0e0e 100%);
        border-color: #9b2335;
    }
    .var-name { font-size: 0.8rem; font-weight: 600; color: #a0aec0; margin: 0.4rem 0; }
    .var-status { font-size: 1.4rem; font-weight: 800; }
    .var-status.pass { color: #48bb78; }
    .var-status.fail { color: #fc8181; }
    .var-score {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace !important;
        color: #718096;
        margin-top: 0.3rem;
    }

    /* ── Rubric Summary Bar ── */
    .rubric-summary {
        border-radius: 12px;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    .rubric-summary.pass {
        background: linear-gradient(90deg, #0f2319 0%, #0d1f17 100%);
        border: 1px solid #276749;
        color: #68d391;
    }
    .rubric-summary.fail {
        background: linear-gradient(90deg, #2d1515 0%, #1f0e0e 100%);
        border: 1px solid #9b2335;
        color: #fc8181;
    }

    /* ── AI Model Cards ── */
    .model-card {
        background: #0f1923;
        border: 1px solid #1e2a3a;
        border-radius: 14px;
        padding: 1.4rem;
        height: 100%;
    }
    .model-name {
        font-size: 0.85rem; font-weight: 700;
        color: #a0aec0; margin-bottom: 1rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Final Decision ── */
    .decision-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .decision-card.layak {
        background: linear-gradient(160deg, #0f2319 0%, #0d1a13 100%);
        border: 2px solid #276749;
    }
    .decision-card.tidak-layak {
        background: linear-gradient(160deg, #2d1515 0%, #1a0d0d 100%);
        border: 2px solid #9b2335;
    }
    .decision-card.review {
        background: linear-gradient(160deg, #2d2415 0%, #1a1709 100%);
        border: 2px solid #b7791f;
    }
    .decision-label {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.4rem;
    }
    .decision-label.layak { color: #68d391; }
    .decision-label.tidak-layak { color: #fc8181; }
    .decision-label.review { color: #f6ad55; }
    .decision-sub {
        font-size: 0.85rem;
        color: #718096;
    }

    /* ── Saran Box ── */
    .saran-box {
        background: #0f1923;
        border: 1px solid #1e2a3a;
        border-left: 3px solid #f6ad55;
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
        font-size: 0.87rem;
        color: #cbd5e0;
    }
    .saran-var {
        font-weight: 700;
        color: #f6ad55;
        font-size: 0.82rem;
        margin-bottom: 0.2rem;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Sidebar Styling ── */
    .sidebar-status-row {
        display: flex; align-items: center; gap: 0.6rem;
        padding: 0.6rem 0.8rem;
        background: #0f1923;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border: 1px solid #1e2a3a;
        font-size: 0.85rem;
        color: #cbd5e0;
    }
    .dot-green { color: #48bb78; }
    .dot-red   { color: #fc8181; }

    /* ── Override Streamlit defaults ── */
    .stButton > button {
        background: linear-gradient(90deg, #2b6cb0, #2c7a7b) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem 1.5rem !important;
        transition: opacity 0.2s !important;
        letter-spacing: 0.01em !important;
    }
    .stButton > button:hover { opacity: 0.88 !important; }
    .stRadio label { color: #a0aec0 !important; font-size: 0.88rem !important; }
    .stMetric label { color: #718096 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; font-family: 'JetBrains Mono', monospace !important; }
    [data-testid="stExpander"] { background: #0f1923 !important; border: 1px solid #1e2a3a !important; border-radius: 10px !important; }
    div[data-testid="stFileUploader"] { background: transparent; }
    .stAlert { border-radius: 10px !important; }
    hr { border-color: #1e2a3a !important; }
    h1,h2,h3,h4 { color: #e2e8f0 !important; font-family: 'Sora', sans-serif !important; }
    p, li { color: #a0aec0 !important; }
    .stMarkdown p { color: #a0aec0 !important; }
    label { color: #a0aec0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero Header ───────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">🤖 Machine Learning · NLP · SVM + TF-IDF</div>
    <div class="main-title">🎯 SponsorSmart AI</div>
    <div class="subtitle">Sistem Pendukung Keputusan — Penilaian Kelayakan Proposal Sponsorship secara Otomatis & Transparan</div>
</div>
""", unsafe_allow_html=True)

# ── Konstanta ─────────────────────────────────────────
HF_REPO_ID    = "calpycbara/sponsorsmart-indobert"
OCR_THRESHOLD = 50
RUBRIC_PASS_THRESHOLD = 3

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

# ── Load Model ────────────────────────────────────────
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
            repo_id=HF_REPO_ID, filename="svm_model.pkl", token=HF_TOKEN
        )
        with open(svm_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[SVM Load Error] {e}")
        return None

# ── Ekstraksi PDF ─────────────────────────────────────
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

# ── Rubric Scoring ────────────────────────────────────
def score_rubric(text: str) -> dict:
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

    total  = sum(scores.values())
    passed = total >= RUBRIC_PASS_THRESHOLD

    return {"scores": scores, "evidences": evidences, "total": total, "passed": passed}

# ── Prediksi IndoBERT ─────────────────────────────────
def predict_bert(text: str, tokenizer, model) -> dict:
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

# ── Chart Skor ────────────────────────────────────────
def render_score_chart(scores: dict):
    vars_  = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#48bb78" if v == 1 else "#fc8181" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor('#0f1923')
    ax.set_facecolor('#0f1923')

    bars = ax.barh(vars_, vals, color=colors, edgecolor='none', height=0.45)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Skor (0 = Tidak Terpenuhi, 1 = Terpenuhi)", color='#718096', fontsize=9)
    ax.set_title("Skor Tiap Variabel — Penilaian Rubrikasi", color='#e2e8f0', fontsize=11, fontweight='bold', pad=12)
    ax.tick_params(colors='#a0aec0')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e2a3a')

    for bar, val in zip(bars, vals):
        lbl = "✓ Terpenuhi" if val == 1 else "✗ Tidak"
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=9, fontweight="bold",
                color="#48bb78" if val == 1 else "#fc8181")

    green = mpatches.Patch(color="#48bb78", label="Terpenuhi (1)")
    red   = mpatches.Patch(color="#fc8181", label="Tidak Terpenuhi (0)")
    legend = ax.legend(handles=[green, red], loc="lower right",
                       facecolor='#0d1117', edgecolor='#1e2a3a', labelcolor='#a0aec0')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem">🎯</div>
        <div style="font-family:'Sora',sans-serif; font-weight:800; font-size:1.1rem;
                    background:linear-gradient(90deg,#63b3ed,#9ae6b4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            SponsorSmart
        </div>
        <div style="font-size:0.72rem; color:#4a5568; margin-top:0.2rem; letter-spacing:0.05em;">
            v2.0 · Kelompok 2 · Telkom University
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.divider()
    st.markdown("**🔌 Status Sistem AI**")

    with st.spinner("Memeriksa status..."):
        tokenizer_check, model_check = load_bert_model()
        svm_check                    = load_svm_model()

    svm_status  = ("🟢", "Siap") if svm_check is not None else ("🔴", "Tidak tersedia")
    bert_status = ("🟢", "Siap") if tokenizer_check is not None else ("🔴", "Tidak tersedia")

    st.markdown(f'''
    <div class="sidebar-status-row">
        <span>{svm_status[0]}</span>
        <span style="font-weight:600;">Model Utama</span>
        <span style="margin-left:auto; font-size:0.75rem; color:#718096;">{svm_status[1]}</span>
    </div>
    <div class="sidebar-status-row">
        <span>{bert_status[0]}</span>
        <span style="font-weight:600;">Model Pembanding</span>
        <span style="margin-left:auto; font-size:0.75rem; color:#718096;">{bert_status[1]}</span>
    </div>
    ''', unsafe_allow_html=True)

    if svm_check is None and tokenizer_check is None:
        st.warning("⚠️ Sistem AI tidak tersedia. Hanya Penilaian Rubrikasi yang aktif.")

    st.divider()
    st.markdown("**📖 Cara Kerja Penilaian**")
    st.markdown('''
    <div style="font-size:0.82rem; color:#718096; line-height:1.8;">
    <b style="color:#63b3ed;">Tahap 1 — Cek Kelengkapan Proposal</b><br>
    Sistem memeriksa 5 aspek penting secara otomatis.<br>
    Kurang dari 3 aspek → ❌ Proposal ditolak.<br><br>
    <b style="color:#63b3ed;">Tahap 2 — Penilaian AI</b><br>
    Proposal yang lolos dinilai lebih dalam oleh AI.<br><br>
    <b style="color:#e2e8f0;">Hasil Akhir:</b><br>
    ✅ <b style="color:#68d391;">Direkomendasikan</b> = Lolos semua penilaian<br>
    ⚠️ <b style="color:#f6ad55;">Perlu Ditinjau</b> = Ada ketidaksesuaian<br>
    ❌ <b style="color:#fc8181;">Tidak Direkomendasikan</b> = Tidak lolos
    </div>
    ''', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN — Upload Section
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <span class="section-badge">STEP 1</span>
    <span class="section-title">📤 Upload Proposal Sponsorship</span>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload file PDF proposal sponsorship",
    type=["pdf"],
    help="Format: PDF · Maks 50MB · Halaman dibaca otomatis (pdfPlumber + OCR fallback)"
)

if uploaded_file is None:
    # ── Landing Info ──────────────────────────────────
    st.markdown("""
    <div style="background:#0f1923; border:1px solid #1e2a3a; border-radius:14px;
                padding:1.4rem 1.6rem; margin-top:0.5rem; margin-bottom:1.5rem;">
        <div style="font-size:0.82rem; color:#718096; margin-bottom:0.8rem;">
            📎 Upload file PDF proposal Anda di atas untuk memulai analisis otomatis.
        </div>
        <div>
            <span class="info-pill">📄 Format: PDF</span>
            <span class="info-pill">🔍 pdfPlumber + OCR</span>
            <span class="info-pill">⭐ Model Utama + 🔬 Pembanding</span>
            <span class="info-pill">⚡ 2 Tahap Penilaian</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:1.5rem;">
        <span class="section-title">📋 Cara Kerja Sistem</span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("01", "📋", "Penilaian Rubrikasi",
         "5 variabel keyword diperiksa secara otomatis. Skor < 3 dari 5 → proposal langsung ditolak."),
        ("02", "🤖", "Verifikasi AI",
         "Proposal yang lolos kelengkapan dinilai oleh dua model AI. Keputusan final ditentukan oleh model terbaik."),
        ("03", "⚖️", "Keputusan Final",
         "Layak jika rubrik ✅ dan AI ✅. Konflik antara keduanya → rekomendasi review manual."),
    ]
    for col, (num, icon, title, desc) in zip([col1, col2, col3], cards):
        with col:
            st.markdown(f"""
            <div class="stage-card">
                <div class="stage-num">{num}</div>
                <div class="stage-icon">{icon}</div>
                <div class="stage-title">{title}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-header" style="margin-top:2rem;">
        <span class="section-title">📊 Variabel Penilaian Rubrikasi</span>
    </div>
    """, unsafe_allow_html=True)

    rubric_cols = st.columns(5)
    rubric_info = [
        ("📡", "Exposure", "Jangkauan audiens & media publikasi"),
        ("🎯", "Relevansi", "Kesesuaian acara dengan profil sponsor"),
        ("🎁", "Benefit", "Keuntungan konkret bagi sponsor"),
        ("💰", "Anggaran", "Kejelasan nominal & rincian biaya"),
        ("🏛️", "Kredibilitas", "Profesionalitas penyelenggara"),
    ]
    for col, (icon, name, desc) in zip(rubric_cols, rubric_info):
        with col:
            st.markdown(f"""
            <div style="background:#0f1923; border:1px solid #1e2a3a; border-radius:12px;
                        padding:1rem 0.8rem; text-align:center;">
                <div style="font-size:1.6rem;">{icon}</div>
                <div style="font-weight:700; font-size:0.85rem; color:#e2e8f0; margin:0.4rem 0 0.3rem;">{name}</div>
                <div style="font-size:0.75rem; color:#718096; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # ── File Uploaded ──────────────────────────────────
    st.success(f"✅ File berhasil diupload: **{uploaded_file.name}**")

    with st.spinner("🔍 Mengekstrak teks dari PDF..."):
        full_text, method = extract_text(uploaded_file)

    # File info row
    method_label = {"pdfplumber": "PDFPLUMBER", "ocr": "OCR (Tesseract)", "failed": "GAGAL"}
    col_m, col_w, col_c = st.columns(3)
    col_m.metric("🔧 Metode Ekstraksi", method_label.get(method, method.upper()))
    col_w.metric("📝 Jumlah Kata",      f"{len(full_text.split()):,}")
    col_c.metric("🔢 Jumlah Karakter",  f"{len(full_text):,}")

    with st.expander("👁️ Preview Teks Hasil Ekstraksi", expanded=False):
        if full_text:
            preview = full_text[:2000] + "\n\n…[teks dipotong]" if len(full_text) > 2000 else full_text
            st.text_area("Teks Proposal:", preview, height=220,
                         help="Hanya preview — teks lengkap digunakan untuk analisis")
        else:
            st.error("Teks tidak dapat diekstrak dari file ini.")

    if not full_text or len(full_text.split()) < 30:
        st.error("⚠️ Teks proposal terlalu sedikit untuk dianalisis. Coba upload file PDF yang lebih lengkap.")
        st.stop()

    st.divider()

    if st.button("🚀 Mulai Analisis Proposal", type="primary", use_container_width=True):

        st.markdown("""
        <div class="section-header" style="margin-top:1rem;">
            <span class="section-badge">TAHAP 1</span>
            <span class="section-title">📋 Penilaian Rubrikasi</span>
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("📋 Menghitung skor rubrikasi..."):
            rubric_result = score_rubric(full_text)

        # ── Variabel Cards ──
        cols = st.columns(5)
        for i, (var, score) in enumerate(rubric_result["scores"].items()):
            with cols[i]:
                css_class = "pass" if score == 1 else "fail"
                status    = "✓" if score == 1 else "✗"
                st.markdown(f"""
                <div class="var-card {css_class}">
                    <div style="font-size:1.5rem">{VAR_ICONS[var]}</div>
                    <div class="var-name">{var}</div>
                    <div class="var-status {css_class}">{status}</div>
                </div>
                """, unsafe_allow_html=True)

        terpenuhi       = [v for v, s in rubric_result["scores"].items() if s == 1]
        tidak_terpenuhi = [v for v, s in rubric_result["scores"].items() if s == 0]

        rubric_class = "pass" if rubric_result["passed"] else "fail"
        rubric_icon  = "✅" if rubric_result["passed"] else "❌"
        rubric_text  = (
            f"Kelengkapan Proposal: Lolos ({rubric_result['total']} dari 5 aspek terpenuhi) — lanjut ke Penilaian AI"
            if rubric_result["passed"] else
            f"Kelengkapan Proposal: Tidak Memenuhi Syarat ({rubric_result['total']} dari 5 aspek terpenuhi) — proposal tidak dapat dilanjutkan"
        )
        st.markdown(f"""
        <div class="rubric-summary {rubric_class}">
            <span style="font-size:1.3rem">{rubric_icon}</span>
            <span>{rubric_text}</span>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 Detail Bukti per Variabel (Penilaian Rubrikasi)"):
            for var, evid in rubric_result["evidences"].items():
                icon     = "✅" if rubric_result["scores"][var] == 1 else "❌"
                evid_str = str(evid) if evid else "Tidak ditemukan indikator"
                st.markdown(f"{icon} **{var}**: `{evid_str}`")

        # ════════════════════════════════════════════
        # TAHAP 2: VERIFIKASI AI
        # ════════════════════════════════════════════
        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;">
            <span class="section-badge">TAHAP 2</span>
            <span class="section-title">🤖 Penilaian AI</span>
        </div>
        """, unsafe_allow_html=True)

        svm_label    = None
        svm_conf     = 0
        bert_result  = {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
        ai_available = False

        if not rubric_result["passed"]:
            st.markdown("""
            <div style="background:#0f1923; border:1px solid #2d3748; border-radius:12px;
                        padding:1.2rem 1.4rem; color:#718096; font-size:0.9rem;">
                ⏭️ <b style="color:#a0aec0;">Penilaian AI dilewati</b> — Proposal tidak memenuhi 
                syarat kelengkapan dasar. Penilaian AI tidak perlu dijalankan.
            </div>
            """, unsafe_allow_html=True)
        else:
            col_svm, col_bert = st.columns(2)

            # ── Model Utama: SVM (penentu keputusan) ──
            with col_svm:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown('''<div class="model-name">⭐ Model Utama <span style="font-size:0.65rem;color:#f6ad55;margin-left:4px;">PENENTU KEPUTUSAN</span></div>''', unsafe_allow_html=True)
                with st.spinner("Menilai proposal..."):
                    svm_model = load_svm_model()
                    if svm_model:
                        ai_available = True
                        svm_label    = svm_model.predict([full_text])[0]
                        svm_prob     = svm_model.predict_proba([full_text])[0]
                        svm_conf     = int(max(svm_prob) * 100)
                        is_layak     = svm_label == "Layak"
                        color        = "#48bb78" if is_layak else "#fc8181"
                        verdict_text = "Direkomendasikan" if is_layak else "Tidak Direkomendasikan"
                        verdict_icon = "✅" if is_layak else "❌"
                        st.markdown(f"""
                        <div style="font-size:1.5rem; font-weight:800; color:{color};
                                    margin:0.5rem 0 0.6rem; letter-spacing:-0.02em;">
                            {verdict_icon} {verdict_text}
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(svm_conf)
                        st.caption(f"Tingkat Keyakinan: **{svm_conf}%**")
                    else:
                        st.warning(f"⚠️ Model utama tidak tersedia.")
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Model Pembanding: IndoBERT (hanya informasi) ──
            with col_bert:
                st.markdown('<div class="model-card">', unsafe_allow_html=True)
                st.markdown('''<div class="model-name">🔬 Model Pembanding <span style="font-size:0.65rem;color:#718096;margin-left:4px;">REFERENSI</span></div>''', unsafe_allow_html=True)
                with st.spinner("Menilai proposal (pembanding)..."):
                    tokenizer, bert_model = load_bert_model()
                    bert_result           = predict_bert(full_text, tokenizer, bert_model)
                    if bert_result["label"]:
                        is_layak_bert = bert_result["label"] == "Layak"
                        color_bert    = "#48bb78" if is_layak_bert else "#fc8181"
                        verdict_bert  = "Direkomendasikan" if is_layak_bert else "Tidak Direkomendasikan"
                        icon_bert     = "✅" if is_layak_bert else "❌"
                        conf_bert     = int(bert_result["confidence"] * 100)
                        st.markdown(f"""
                        <div style="font-size:1.5rem; font-weight:800; color:{color_bert};
                                    margin:0.5rem 0 0.6rem; letter-spacing:-0.02em;">
                            {icon_bert} {verdict_bert}
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(conf_bert)
                        st.caption(f"Tingkat Keyakinan: **{conf_bert}%**")
                    else:
                        st.info("Model pembanding tidak tersedia.")
                st.markdown('</div>', unsafe_allow_html=True)

            if ai_available and bert_result["label"] and svm_label != bert_result["label"]:
                st.markdown("""
                <div style="background:#1a1a0d; border:1px solid #b7791f33; border-radius:10px;
                            padding:0.8rem 1.1rem; font-size:0.82rem; color:#f6ad55; margin-top:0.5rem;">
                    ℹ️ Kedua model memberikan hasil berbeda. <b>Keputusan final mengikuti Model Utama</b> 
                    karena terbukti lebih akurat berdasarkan evaluasi.
                </div>
                """, unsafe_allow_html=True)

        # ════════════════════════════════════════════
        # KEPUTUSAN FINAL
        # ════════════════════════════════════════════
        st.markdown("""
        <div class="section-header" style="margin-top:1.5rem;">
            <span class="section-badge">HASIL</span>
            <span class="section-title">⚖️ Hasil Penilaian Akhir</span>
        </div>
        """, unsafe_allow_html=True)

        # Logika keputusan
        if not rubric_result["passed"]:
            final_label  = "Tidak Direkomendasikan"
            final_reason = "rubric_failed"
        elif not ai_available:
            final_label  = "Direkomendasikan"
            final_reason = "rubric_only"
        else:
            # Keputusan final selalu mengikuti Model Utama (SVM)
            if svm_label == "Layak":
                final_label  = "Direkomendasikan"
                final_reason = "rubric_pass_ai_agree"
            else:
                final_label  = "Perlu Ditinjau Ulang"
                final_reason = "rubric_pass_ai_disagree"

        # Decision display config
        if final_reason == "rubric_pass_ai_agree":
            d_class, d_lclass, d_icon, d_label = "layak", "layak", "✅", "DIREKOMENDASIKAN"
            d_sub = "Proposal Anda memenuhi semua kriteria dan telah diverifikasi oleh sistem AI."
        elif final_reason == "rubric_only":
            d_class, d_lclass, d_icon, d_label = "layak", "layak", "✅", "DIREKOMENDASIKAN"
            d_sub = "Proposal Anda lolos penilaian kelengkapan. Verifikasi AI tidak tersedia saat ini."
        elif final_reason == "rubric_pass_ai_disagree":
            d_class, d_lclass, d_icon, d_label = "review", "review", "⚠️", "PERLU DITINJAU ULANG"
            d_sub = "Proposal lolos kelengkapan, namun AI menemukan kekurangan pada isi proposal. Disarankan untuk diperbaiki."
        else:
            d_class, d_lclass, d_icon, d_label = "tidak-layak", "tidak-layak", "❌", "TIDAK DIREKOMENDASIKAN"
            d_sub = f"Hanya {rubric_result['total']}/5 variabel rubrikasi terpenuhi (minimum: {RUBRIC_PASS_THRESHOLD})."

        col_dec, col_detail = st.columns([1, 2])

        with col_dec:
            st.markdown(f"""
            <div class="decision-card {d_class}">
                <div style="font-size:2.5rem; margin-bottom:0.4rem;">{d_icon}</div>
                <div class="decision-label {d_lclass}">{d_label}</div>
                <div class="decision-sub">{d_sub}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown("**📋 Penjelasan Hasil**")

            if final_reason == "rubric_failed":
                st.error(f"❌ Proposal tidak memenuhi syarat kelengkapan: hanya {rubric_result['total']} dari 5 aspek terpenuhi (minimal 3 aspek).")
                st.info("ℹ️ Penilaian AI tidak dijalankan karena proposal belum memenuhi syarat kelengkapan dasar.")
            elif final_reason == "rubric_only":
                st.success(f"✅ Proposal memenuhi {rubric_result['total']} dari 5 aspek kelengkapan.")
                st.info("ℹ️ Sistem AI tidak tersedia — keputusan berdasarkan kelengkapan proposal saja.")
            elif final_reason == "rubric_pass_ai_agree":
                st.success(f"✅ Proposal memenuhi {rubric_result['total']} dari 5 aspek kelengkapan.")
                st.success("✅ Sistem AI mengkonfirmasi proposal ini layak mendapatkan sponsorship.")
                st.info("💡 Proposal Anda sudah lengkap dan dinilai baik oleh sistem kami. Silakan ajukan ke pihak sponsor.")
            elif final_reason == "rubric_pass_ai_disagree":
                st.success(f"✅ Proposal memenuhi {rubric_result['total']} dari 5 aspek kelengkapan.")
                st.warning("⚠️ Sistem AI menilai isi proposal masih kurang meyakinkan.\n\nProposal sudah cukup lengkap secara struktur, namun perlu diperkuat pada bagian isi dan penjelasan. Kami sarankan untuk merevisi sebelum diajukan.")

            if terpenuhi:
                st.success(f"✅ Aspek yang sudah terpenuhi: {', '.join(terpenuhi)}")
            if tidak_terpenuhi:
                st.error(f"❌ Aspek yang perlu dilengkapi: {', '.join(tidak_terpenuhi)}")

        # ── Saran Perbaikan ──
        if tidak_terpenuhi:
            st.markdown("""
            <div class="section-header" style="margin-top:1rem;">
                <span class="section-title">💡 Saran Perbaikan Proposal</span>
            </div>
            """, unsafe_allow_html=True)
            for var in tidak_terpenuhi:
                st.markdown(f"""
                <div class="saran-box">
                    <div class="saran-var">{VAR_ICONS.get(var,'')} {var}</div>
                    {SARAN.get(var, '')}
                </div>
                """, unsafe_allow_html=True)
