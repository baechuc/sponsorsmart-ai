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
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    /* ═══ LIGHT MODE ═══ */
    .stApp {
        background-color: #f4f3ff;
        background-image: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(99,83,243,0.12) 0%, transparent 70%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #1e1b4b 0%, #2d2870 100%) !important;
        border-right: none !important;
    }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] b  { color: #ffffff !important; }
    section[data-testid="stSidebar"] .stDivider { border-color: rgba(255,255,255,0.12) !important; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }
    section[data-testid="stSidebar"] p  { color: #c7d2fe !important; }

    /* ═══ DARK MODE ═══ */
    [data-theme="dark"] .stApp {
        background-color: #0c0b1a;
        background-image: radial-gradient(ellipse 80% 50% at 50% -5%, rgba(99,83,243,0.18) 0%, transparent 65%);
    }
    [data-theme="dark"] section[data-testid="stSidebar"] {
        background: linear-gradient(175deg, #0f0e1f 0%, #1a1842 100%) !important;
    }
    [data-theme="dark"] .metric-card  { background: #16132e !important; border-color: rgba(99,83,243,0.25) !important; border-top-color: #7c6ff7 !important; }
    [data-theme="dark"] .metric-card p { color: #a5b4fc !important; }
    [data-theme="dark"] .stage-box    { background: #16132e !important; border-color: rgba(99,83,243,0.25) !important; }
    [data-theme="dark"] .stage-pass   { background: #0a2618 !important; border-color: #4ade80 !important; }
    [data-theme="dark"] .stage-fail   { background: #200a0a !important; border-color: #f87171 !important; }
    [data-theme="dark"] .layak-badge       { background: linear-gradient(135deg,#052e16,#064e2a) !important; color: #86efac !important; border-color: #4ade80 !important; }
    [data-theme="dark"] .tidak-layak-badge { background: linear-gradient(135deg,#1c0505,#3b0a0a) !important; color: #fca5a5 !important; border-color: #f87171 !important; }
    [data-theme="dark"] .review-badge      { background: linear-gradient(135deg,#1c1205,#3b2200) !important; color: #fde68a !important; border-color: #fbbf24 !important; }
    [data-theme="dark"] [data-testid="stExpander"] { background: #16132e !important; border-color: rgba(99,83,243,0.25) !important; }
    [data-theme="dark"] hr { border-color: rgba(99,83,243,0.2) !important; }
    [data-theme="dark"] h1,[data-theme="dark"] h2,[data-theme="dark"] h3,[data-theme="dark"] h4 { color: #ede9fe !important; }
    [data-theme="dark"] p  { color: #a5b4fc !important; }
    [data-theme="dark"] .stMarkdown p  { color: #a5b4fc !important; }
    [data-theme="dark"] label { color: #a5b4fc !important; }
    [data-theme="dark"] .subtitle { color: #7c6ff7 !important; }
    [data-theme="dark"] .stMetric [data-testid="stMetricValue"] { color: #ede9fe !important; }
    [data-theme="dark"] .stMetric label { color: #7c6ff7 !important; }

    /* ═══ SHARED ═══ */

    /* ── Header ── */
    .hero-wrap {
        background: linear-gradient(135deg, #1e1b4b 0%, #3b30b8 60%, #5b21b6 100%);
        border-radius: 20px;
        padding: 2.2rem 2rem 1.8rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 40px rgba(99,83,243,0.35);
    }
    .hero-wrap::before {
        content: "";
        position: absolute; inset: 0;
        background: radial-gradient(ellipse 70% 80% at 80% 50%, rgba(251,191,36,0.18) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-wrap::after {
        content: "";
        position: absolute; bottom: -30px; right: -30px;
        width: 200px; height: 200px;
        background: radial-gradient(circle, rgba(251,191,36,0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .main-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 2.6rem; font-weight: 800;
        color: #ffffff !important;
        text-align: center; margin-bottom: 0.3rem;
        letter-spacing: -0.03em;
        text-shadow: 0 2px 20px rgba(251,191,36,0.3);
        -webkit-text-fill-color: unset !important;
        background: none !important;
    }
    .subtitle {
        text-align: center;
        color: #c7d2fe !important;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    /* ── Score badge chips ── */
    .chip-tag {
        background: rgba(251,191,36,0.18);
        color: #fef3c7;
        border: 1px solid rgba(251,191,36,0.4);
        border-radius: 20px;
        font-size: 0.72rem; font-weight: 600;
        padding: 0.25rem 0.9rem;
        letter-spacing: 0.06em;
        display: inline-block;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.3rem 1rem;
        text-align: center;
        border: 1px solid #e0d9ff;
        border-top: 3px solid #6355f3;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 18px rgba(99,83,243,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 28px rgba(99,83,243,0.18); }
    .metric-card p { color: #5b5480 !important; font-size: 0.88rem; }

    /* ── Status badges ── */
    .layak-badge {
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        color: #14532d;
        border: 1.5px solid #4ade80;
        padding: 0.75rem 2.2rem; border-radius: 50px;
        font-size: 1.25rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(74,222,128,0.25);
        font-family: 'Syne', sans-serif !important;
    }
    .tidak-layak-badge {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #7f1d1d;
        border: 1.5px solid #f87171;
        padding: 0.75rem 2.2rem; border-radius: 50px;
        font-size: 1.25rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(248,113,113,0.25);
        font-family: 'Syne', sans-serif !important;
    }
    .review-badge {
        background: linear-gradient(135deg, #fef9c3, #fef08a);
        color: #713f12;
        border: 1.5px solid #fbbf24;
        padding: 0.75rem 2.2rem; border-radius: 50px;
        font-size: 1.25rem; font-weight: 700;
        display: inline-block; margin: 0.8rem 0;
        box-shadow: 0 4px 16px rgba(251,191,36,0.3);
        font-family: 'Syne', sans-serif !important;
    }

    /* ── Stage boxes ── */
    .stage-box  { border: 1.5px solid #e0d9ff; border-radius: 14px; padding: 1rem 1.2rem; margin-bottom: 1rem; background: #ffffff; }
    .stage-pass { border-color: #4ade80; background: #f0fdf4; }
    .stage-fail { border-color: #f87171; background: #fef2f2; }
    .stage-skip { border-color: #d1d5db; background: #f9fafb; opacity: 0.65; }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #f59e0b, #fb923c) !important;
        color: #1c0f00 !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 6px 20px rgba(245,158,11,0.35) !important;
        transition: opacity 0.2s, transform 0.15s !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e0d9ff !important;
        border-radius: 12px !important;
    }

    /* ── Misc ── */
    hr { border-color: rgba(99,83,243,0.15) !important; }
    h1,h2,h3,h4 { color: #1e1b4b !important; font-family: 'Syne', sans-serif !important; }
    p  { color: #5b5480 !important; }
    .stMarkdown p { color: #5b5480 !important; }
    label { color: #5b5480 !important; }
    .stAlert { border-radius: 12px !important; }
    .stMetric label { color: #7c6ff7 !important; font-size: 0.8rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #1e1b4b !important; font-weight: 700 !important; }
    .stFileUploader { border-color: rgba(99,83,243,0.3) !important; border-radius: 12px !important; }

    /* ── Sidebar model status cards ── */
    .sidebar-status {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        color: #e0e7ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-wrap">
    <p class="main-title">🎯 SponsorSmart AI</p>
    <p class="subtitle">Sistem Pendukung Keputusan — Penilaian Kelayakan Proposal Sponsorship</p>
    <div style="text-align:center;margin-top:0.8rem;">
        <span class="chip-tag">MACHINE LEARNING · NLP · SVM + TF-IDF</span>
    </div>
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
        return ["Layak" if p[layak_idx] >= self.threshold else "Tidak Layak" for p in probs]

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
        svm_path = hf_hub_download(repo_id=HF_REPO_ID, filename="svm_model.pkl", token=HF_TOKEN)
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

# ── Fungsi Prediksi IndoBERT ──────────────────────────
def predict_bert(text: str, tokenizer, model) -> dict:
    if tokenizer is None or model is None:
        return {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
    inputs = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
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
    st.image("https://img.icons8.com/color/96/000000/target.png", width=72)
    st.title("ℹ️ Info Sistem")
    st.divider()
    st.markdown("### 🔌 Status Model")

    with st.spinner("Cek koneksi model..."):
        tokenizer_check, model_check = load_bert_model()
        svm_check                    = load_svm_model()

    st.markdown(f'<div class="sidebar-status">{"🟢" if svm_check else "🔴"} <b>Model Utama</b> — {"Siap" if svm_check else "Tidak ditemukan"}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-status">{"🟢" if tokenizer_check else "🔴"} <b>Model Pembanding</b> — {"Siap" if tokenizer_check else "Tidak ditemukan"}</div>', unsafe_allow_html=True)

    if tokenizer_check is None and svm_check is None:
        st.warning("⚠️ Semua model offline.\nHanya Penilaian Rubrikasi yang aktif.")

    st.divider()
    st.markdown("### 🔁 Alur Penilaian")
    st.markdown("""
    Sistem ini bekerja dalam **2 tahap berurutan**:

    **Tahap 1 — Penilaian Rubrikasi**
    5 aspek proposal diperiksa.
    Skor < 3 → ❌ Langsung ditolak.

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
        st.error("⚠️ Teks proposal terlalu sedikit untuk dianalisis.")
        st.stop()

    st.divider()

    if st.button("🚀 Analisis Proposal", type="primary", use_container_width=True):
        st.divider()
        st.subheader("📊 Hasil Analisis")

        # ════════════════ TAHAP 1: FILTER RUBRIC ════════════════
        st.markdown("### Tahap 1 — Penilaian Rubrikasi")

        with st.spinner("📋 Menghitung skor penilaian rubrikasi..."):
            rubric_result = score_rubric(full_text)

        cols = st.columns(5)
        for i, (var, score) in enumerate(rubric_result["scores"].items()):
            with cols[i]:
                if score == 1:
                    bg, border, status = "#f0fdf4", "#4ade80", "✓"
                else:
                    bg, border, status = "#fef2f2", "#f87171", "✗"
                st.markdown(f"""
                <div style="background:{bg};padding:14px 10px;border-radius:14px;text-align:center;
                            border:2px solid {border};box-shadow:0 2px 10px {border}33;">
                    <div style="font-size:1.6rem">{VAR_ICONS[var]}</div>
                    <div style="font-weight:700;font-size:0.88rem;color:#1e1b4b;">{var}</div>
                    <div style="font-size:2rem;font-weight:900;color:{"#16a34a" if score==1 else "#dc2626"};">{status}</div>
                    <div style="font-size:0.78rem;color:#6b7280;">Skor: {score}/1</div>
                </div>
                """, unsafe_allow_html=True)

        terpenuhi       = [v for v, s in rubric_result["scores"].items() if s == 1]
        tidak_terpenuhi = [v for v, s in rubric_result["scores"].items() if s == 0]

        rubric_color  = "#f0fdf4" if rubric_result["passed"] else "#fef2f2"
        rubric_border = "#4ade80" if rubric_result["passed"] else "#f87171"
        rubric_icon   = "✅" if rubric_result["passed"] else "❌"
        rubric_text   = (
            f"Penilaian Rubrikasi lolos ({rubric_result['total']}/5 variabel terpenuhi) — lanjut ke verifikasi AI"
            if rubric_result["passed"] else
            f"Penilaian Rubrikasi gagal ({rubric_result['total']}/5 variabel terpenuhi) — proposal ditolak tanpa perlu AI"
        )
        st.markdown(
            f'<div style="background:{rubric_color};padding:0.9rem 1.3rem;border:2px solid {rubric_border};'
            f'border-radius:12px;font-weight:700;font-size:1rem;margin-top:1rem;'
            f'box-shadow:0 2px 12px {rubric_border}33;">'
            f'{rubric_icon} {rubric_text}</div>',
            unsafe_allow_html=True
        )

        with st.expander("🔍 Detail Bukti per Variabel (Penilaian Rubrikasi)"):
            for var, evid in rubric_result["evidences"].items():
                icon     = "✅" if rubric_result["scores"][var] == 1 else "❌"
                evid_str = str(evid) if evid else "Tidak ditemukan indikator"
                st.markdown(f"{icon} **{var}**: {evid_str}")

        # ════════════════ TAHAP 2: VERIFIKASI AI ════════════════
        st.divider()
        st.markdown("### Tahap 2 — Verifikasi AI")

        bert_result  = {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
        svm_label    = None
        svm_conf     = 0
        ai_available = False

        if not rubric_result["passed"]:
            st.markdown(
                '<div class="stage-skip" style="border:1.5px solid rgba(128,128,128,0.3);border-radius:14px;'
                'padding:1rem 1.2rem;background:rgba(128,128,128,0.06);">'
                '⏭️ <b>AI tidak dijalankan</b> — proposal sudah gugur di Tahap 1.</div>',
                unsafe_allow_html=True
            )
        else:
            svm_col, bert_col = st.columns(2)

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
                        st.warning(f"⚠️ Model Utama tidak tersedia.")

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

        # ════════════════ KEPUTUSAN FINAL ════════════════
        st.divider()
        st.markdown("### ⚖️ Keputusan Final")

        if not rubric_result["passed"]:
            final_reason = "rubric_failed"
        elif not ai_available:
            final_reason = "rubric_only"
        elif svm_label == "Layak":
            final_reason = "rubric_pass_ai_agree"
        else:
            final_reason = "rubric_pass_ai_disagree"

        badge_map = {
            "rubric_pass_ai_agree"   : ("layak-badge",      "✅ DIREKOMENDASIKAN"),
            "rubric_only"            : ("layak-badge",      "✅ DIREKOMENDASIKAN"),
            "rubric_pass_ai_disagree": ("review-badge",     "⚠️ PERLU DITINJAU ULANG"),
            "rubric_failed"          : ("tidak-layak-badge","❌ TIDAK DIREKOMENDASIKAN"),
        }
        badge_class, label_text = badge_map[final_reason]

        col_dec, col_reason = st.columns([1, 2])
        with col_dec:
            st.markdown(f'<div style="text-align:center"><span class="{badge_class}">{label_text}</span></div>', unsafe_allow_html=True)

        with col_reason:
            st.markdown("**📋 Penjelasan Hasil:**")
            if final_reason == "rubric_failed":
                st.error(f"❌ Rubrikasi gagal: hanya {rubric_result['total']}/5 aspek terpenuhi (min {RUBRIC_PASS_THRESHOLD}). AI tidak dijalankan.")
            elif final_reason == "rubric_only":
                st.success(f"✅ Rubrikasi lolos ({rubric_result['total']}/5 aspek).")
                st.info("ℹ️ Model AI tidak tersedia — keputusan berdasarkan rubrikasi saja.")
            elif final_reason == "rubric_pass_ai_agree":
                st.success(f"✅ Rubrikasi lolos ({rubric_result['total']}/5 aspek).")
                st.success("✅ Model Utama mengkonfirmasi proposal layak mendapatkan sponsorship.")
                st.info("💡 Proposal sudah lengkap dan dinilai baik. Silakan ajukan ke pihak sponsor.")
            elif final_reason == "rubric_pass_ai_disagree":
                st.success(f"✅ Rubrikasi lolos ({rubric_result['total']}/5 aspek).")
                st.warning("⚠️ Model Utama menilai isi proposal masih perlu diperkuat sebelum diajukan.")

            if terpenuhi:
                st.success(f"✅ Aspek terpenuhi: {', '.join(terpenuhi)}")
            if tidak_terpenuhi:
                st.error(f"❌ Perlu dilengkapi: {', '.join(tidak_terpenuhi)}")

        if tidak_terpenuhi:
            with st.expander("💡 Saran Perbaikan Proposal"):
                for var in tidak_terpenuhi:
                    st.markdown(f"**{var}:** {SARAN.get(var, '')}")

else:
    # ── Landing Page ──────────────────────────────────────
    st.info("👆 Upload file PDF proposal sponsorship untuk memulai analisis.")

    st.markdown("#### Alur Penilaian 2 Tahap")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2 style="font-size:2rem">📋</h2>
            <b>Tahap 1: Penilaian Rubrikasi</b>
            <p>5 aspek proposal diperiksa.<br>Skor &lt; 3 → langsung ditolak.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2 style="font-size:2rem">🤖</h2>
            <b>Tahap 2: Penilaian AI</b>
            <p>Dua model AI dijalankan otomatis.<br>Model Utama menentukan keputusan.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2 style="font-size:2rem">⚖️</h2>
            <b>Hasil Akhir</b>
            <p>Direkomendasikan jika rubrikasi ✅ <b>dan</b> AI ✅.<br>Konflik → perlu ditinjau ulang.</p>
        </div>""", unsafe_allow_html=True)
