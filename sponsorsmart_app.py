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

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .subtitle { text-align: center; color: #666; margin-bottom: 2rem; }
    .metric-card {
        background: #f8f9fa; border-radius: 12px;
        padding: 1rem; text-align: center;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
    }
    .layak-badge {
        background: #d4edda; color: #155724;
        padding: 0.8rem 2rem; border-radius: 25px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 1rem 0;
    }
    .tidak-layak-badge {
        background: #f8d7da; color: #721c24;
        padding: 0.8rem 2rem; border-radius: 25px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 1rem 0;
    }
    .review-badge {
        background: #fff3cd; color: #856404;
        padding: 0.8rem 2rem; border-radius: 25px;
        font-size: 1.4rem; font-weight: 700;
        display: inline-block; margin: 1rem 0;
    }
    .stage-box {
        border: 2px solid #dee2e6; border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 1rem;
    }
    .stage-pass  { border-color: #2ecc71; background: #f0fff4; }
    .stage-fail  { border-color: #e74c3c; background: #fff5f5; }
    .stage-skip  { border-color: #adb5bd; background: #f8f9fa; opacity: 0.6; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🎯 SponsorSmart AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistem Pendukung Keputusan Penilaian Kelayakan Proposal Sponsorship</p>',
            unsafe_allow_html=True)
st.divider()

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

# ── Fungsi Chart Skor ─────────────────────────────────
def render_score_chart(scores: dict):
    vars_  = list(scores.keys())
    vals   = list(scores.values())
    colors = ["#2ecc71" if v == 1 else "#e74c3c" for v in vals]

    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(vars_, vals, color=colors, edgecolor="white", height=0.5)
    ax.set_xlim(0, 1.5)
    ax.set_xlabel("Skor (0 = Tidak Terpenuhi, 1 = Terpenuhi)")
    ax.set_title("Skor Tiap Variabel — Penilaian Rubrikasi")

    for bar, val in zip(bars, vals):
        lbl = "✓ Terpenuhi" if val == 1 else "✗ Tidak"
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=10, fontweight="bold")

    green = mpatches.Patch(color="#2ecc71", label="Terpenuhi (1)")
    red   = mpatches.Patch(color="#e74c3c", label="Tidak Terpenuhi (0)")
    ax.legend(handles=[green, red], loc="lower right")
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/target.png", width=80)
    st.title("⚙️ Pengaturan")

    model_choice = st.radio(
        "Pilih Model AI (Tahap 2):",
        ["IndoBERT (Rekomendasi)", "SVM + TF-IDF", "Keduanya (Bandingkan)"]
    )

    st.divider()
    st.markdown("### 🔌 Status Model")

    with st.spinner("Cek koneksi model..."):
        tokenizer_check, model_check = load_bert_model()
        svm_check                    = load_svm_model()

    if tokenizer_check is not None:
        st.markdown("🟢 **IndoBERT** — Terhubung")
    else:
        st.markdown("🔴 **IndoBERT** — Tidak ditemukan")
        st.caption(f"Repo: `{HF_REPO_ID}`")

    if svm_check is not None:
        st.markdown("🟢 **SVM** — Terhubung")
    else:
        st.markdown("🔴 **SVM** — Tidak ditemukan")
        st.caption("File `svm_model.pkl` belum ada di HuggingFace repo.")

    if tokenizer_check is None and svm_check is None:
        st.warning("⚠️ Semua model offline.\nHanya Rubric Scoring yang aktif.")

    st.divider()
    st.markdown("### 🔁 Alur Penilaian")
    st.markdown("""
    Sistem ini bekerja dalam **2 tahap berurutan**:

    **Tahap 1 — Penilaian Rubrikasi**
    Proposal dinilai dari 5 variabel keyword.
    Skor < 3 → ❌ **Langsung ditolak**, AI tidak dipanggil.

    **Tahap 2 — Verifikasi AI**
    Hanya proposal yang lolos rubric (skor ≥ 3)
    yang dikirim ke IndoBERT / SVM untuk keputusan final.

    **Label Final:**
    - ✅ Layak = Rubric lolos **+** AI setuju
    - ⚠️ Perlu Review = Rubric lolos, tapi AI ragu
    - ❌ Tidak Layak = Rubric gagal (AI tidak dijalankan)
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
                color  = "#d4edda" if score == 1 else "#f8d7da"
                status = "✓" if score == 1 else "✗"
                st.markdown(f"""
                <div style="background:{color};padding:12px;border-radius:10px;text-align:center;">
                    <div style="font-size:1.5rem">{VAR_ICONS[var]}</div>
                    <div style="font-weight:bold;font-size:0.9rem">{var}</div>
                    <div style="font-size:1.8rem;font-weight:800">{status}</div>
                    <div style="font-size:0.8rem">Skor: {score}/1</div>
                </div>
                """, unsafe_allow_html=True)

        st.pyplot(render_score_chart(rubric_result["scores"]))

        terpenuhi       = [v for v, s in rubric_result["scores"].items() if s == 1]
        tidak_terpenuhi = [v for v, s in rubric_result["scores"].items() if s == 0]

        rubric_color = "#d4edda" if rubric_result["passed"] else "#f8d7da"
        rubric_icon  = "✅" if rubric_result["passed"] else "❌"
        rubric_text  = (
            f"Penilaian Rubrikasi lolos ({rubric_result['total']}/5 variabel terpenuhi) "
            f"— lanjut ke verifikasi AI"
            if rubric_result["passed"] else
            f"Penilaian Rubrikasi gagal ({rubric_result['total']}/5 variabel terpenuhi) "
            f"— proposal ditolak tanpa perlu AI"
        )
        st.markdown(
            f'<div style="background:{rubric_color};padding:0.8rem 1.2rem;'
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
        bert_result  = {"label": None, "confidence": 0, "prob_layak": 0, "prob_tidak": 0}
        svm_label    = None
        svm_conf     = 0
        ai_available = False

        if not rubric_result["passed"]:
            st.markdown(
                '<div class="stage-skip" style="border:2px solid #adb5bd;border-radius:12px;'
                'padding:1rem;background:#f8f9fa;opacity:0.7;">'
                '⏭️ <b>Penilaian AI tidak dijalankan</b> — proposal sudah gugur di Tahap 1 (Penilaian Rubrikasi gagal).'
                '</div>',
                unsafe_allow_html=True
            )
        else:
            svm_col, bert_col = st.columns(2)

            # ── Model Utama: SVM (penentu keputusan) ──
            with svm_col:
                st.markdown("**⭐ Model Utama** <span style='font-size:0.72rem;color:#6d28d9;background:#ede9fe;padding:1px 6px;border-radius:4px;margin-left:4px;'>PENENTU KEPUTUSAN</span>", unsafe_allow_html=True)
                with st.spinner("⚙️ Menilai proposal..."):
                    svm_model = load_svm_model()
                    if svm_model:
                        ai_available = True
                        svm_label    = svm_model.predict([full_text])[0]
                        svm_prob     = svm_model.predict_proba([full_text])[0]
                        svm_conf     = int(max(svm_prob) * 100)
                        svm_verdict  = "Direkomendasikan" if svm_label == "Layak" else "Tidak Direkomendasikan"
                        badge        = "layak-badge" if svm_label == "Layak" else "tidak-layak-badge"
                        st.markdown(f'<span class="{badge}">{svm_verdict}</span>', unsafe_allow_html=True)
                        st.progress(svm_conf)
                        st.caption(f"Tingkat Keyakinan: {svm_conf}%")
                    else:
                        st.warning(f"⚠️ Model Utama tidak tersedia.\nPastikan `svm_model.pkl` diupload ke `{HF_REPO_ID}`")

            # ── Model Pembanding: IndoBERT (referensi) ──
            with bert_col:
                st.markdown("**🔬 Model Pembanding** <span style='font-size:0.72rem;color:#6c757d;background:#f8f9fa;border:1px solid #dee2e6;padding:1px 6px;border-radius:4px;margin-left:4px;'>REFERENSI</span>", unsafe_allow_html=True)
                with st.spinner("🧠 Menilai proposal (pembanding)..."):
                    tokenizer, bert_model = load_bert_model()
                    bert_result           = predict_bert(full_text, tokenizer, bert_model)
                    if bert_result["label"]:
                        bert_verdict = "Direkomendasikan" if bert_result["label"] == "Layak" else "Tidak Direkomendasikan"
                        badge        = "layak-badge" if bert_result["label"] == "Layak" else "tidak-layak-badge"
                        st.markdown(f'<span class="{badge}">{bert_verdict}</span>', unsafe_allow_html=True)
                        conf = int(bert_result["confidence"] * 100)
                        st.progress(conf)
                        st.caption(f"Tingkat Keyakinan: {conf}%")
                    else:
                        st.info("Model pembanding tidak tersedia.")

            if ai_available and bert_result["label"] and svm_label != bert_result["label"]:
                st.markdown(
                    '<div class="review-badge" style="display:block;text-align:center;padding:0.6rem 1rem;font-size:0.88rem;margin-top:0.5rem;">'
                    'ℹ️ Kedua model berbeda pendapat — Keputusan final mengikuti Model Utama'
                    '</div>',
                    unsafe_allow_html=True
                )

        # ════════════════════════════════════════════
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
                st.success("✅ Model Utama mengkonfirmasi proposal ini layak mendapatkan sponsorship.")
                st.info("💡 Proposal sudah lengkap dan dinilai baik. Silakan ajukan ke pihak sponsor.")

            elif final_reason == "rubric_pass_ai_disagree":
                st.success(f"✅ Penilaian Rubrikasi lolos ({rubric_result['total']}/5 aspek terpenuhi).")
                st.warning(
                    "⚠️ Model Utama menilai isi proposal masih kurang meyakinkan.\n\n"
                    "Proposal sudah cukup lengkap secara struktur, namun perlu diperkuat pada bagian isi. "
                    "Kami sarankan untuk merevisi sebelum diajukan."
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

    st.markdown("#### Cara Kerja Penilaian 2 Tahap")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>📋</h2><b>Tahap 1: Penilaian Rubrikasi</b>
            <p>5 variabel diperiksa otomatis.<br>Skor &lt; 3 → langsung ditolak.</p>
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
