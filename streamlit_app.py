import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import joblib, numpy as np, pandas as pd, streamlit as st
from src.preprocessing import clean_text

st.set_page_config(page_title="CineScore · Sentiment AI", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif;
  background: #070a10 !important;
  color: #e2e8f0;
}

/* ─── animated mesh bg ─── */
.stApp::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 10%, rgba(99,102,241,.12) 0%, transparent 70%),
    radial-gradient(ellipse 60% 50% at 80% 80%, rgba(244,63,94,.10) 0%, transparent 70%),
    radial-gradient(ellipse 50% 40% at 60% 30%, rgba(16,185,129,.07) 0%, transparent 70%);
  pointer-events: none;
  z-index: 0;
}

/* ─── sidebar ─── */
[data-testid="stSidebar"] {
  background: #0d1117 !important;
  border-right: 1px solid rgba(255,255,255,.06) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1; }

/* ─── hide chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ─── hero ─── */
.hero {
  text-align: center;
  padding: 3.5rem 1rem 2rem;
  position: relative;
}
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(99,102,241,.12);
  border: 1px solid rgba(99,102,241,.3);
  border-radius: 999px;
  padding: 4px 16px;
  font-size: .75rem;
  font-weight: 600;
  letter-spacing: .08em;
  color: #a5b4fc;
  text-transform: uppercase;
  margin-bottom: 1.2rem;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 5vw, 4rem);
  font-weight: 800;
  line-height: 1.05;
  background: linear-gradient(135deg, #f1f5f9 0%, #a5b4fc 45%, #f43f5e 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: .8rem;
}
.hero-sub {
  color: #64748b;
  font-size: 1.05rem;
  max-width: 540px;
  margin: 0 auto;
  line-height: 1.6;
}

/* ─── divider ─── */
.glow-divider {
  height: 1px;
  border: none;
  background: linear-gradient(90deg, transparent, rgba(99,102,241,.5), rgba(244,63,94,.4), transparent);
  margin: 2rem 0;
}

/* ─── glass card ─── */
.card {
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 16px;
  padding: 1.6rem;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color .25s, transform .2s;
}
.card:hover { border-color: rgba(99,102,241,.3); transform: translateY(-1px); }

/* ─── badge row ─── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.09);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: .78rem;
  color: #94a3b8;
  margin: 3px;
  transition: background .2s;
}
.badge:hover { background: rgba(99,102,241,.12); color: #a5b4fc; }

/* ─── example buttons ─── */
.stButton > button {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  color: #94a3b8 !important;
  border-radius: 10px !important;
  font-size: .83rem !important;
  font-weight: 500 !important;
  transition: all .2s !important;
}
.stButton > button:hover {
  background: rgba(99,102,241,.14) !important;
  border-color: rgba(99,102,241,.4) !important;
  color: #a5b4fc !important;
  transform: translateY(-1px) !important;
}

/* ─── primary button ─── */
button[kind="primary"] {
  background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: .95rem !important;
  letter-spacing: .02em !important;
  border-radius: 12px !important;
  padding: .75rem 2rem !important;
  box-shadow: 0 0 24px rgba(99,102,241,.4) !important;
  transition: all .25s !important;
}
button[kind="primary"]:hover {
  box-shadow: 0 0 40px rgba(99,102,241,.65) !important;
  transform: translateY(-2px) !important;
}

/* ─── textarea ─── */
.stTextArea textarea {
  background: rgba(255,255,255,.04) !important;
  border: 1px solid rgba(255,255,255,.1) !important;
  color: #e2e8f0 !important;
  border-radius: 12px !important;
  font-size: .93rem !important;
  line-height: 1.65 !important;
  transition: border .25s, box-shadow .25s !important;
}
.stTextArea textarea:focus {
  border-color: rgba(99,102,241,.55) !important;
  box-shadow: 0 0 0 3px rgba(99,102,241,.18) !important;
}

/* ─── result section ─── */
.result-wrap {
  background: linear-gradient(135deg, rgba(15,23,42,.9) 0%, rgba(15,20,40,.95) 100%);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 20px;
  padding: 2rem;
  position: relative;
  overflow: hidden;
}
.result-wrap::before {
  content: '';
  position: absolute;
  top: -1px; left: 0; right: 0;
  height: 2px;
}
.result-wrap.positive::before { background: linear-gradient(90deg, #10b981, #34d399); }
.result-wrap.negative::before { background: linear-gradient(90deg, #f43f5e, #fb7185); }

.verdict {
  font-family: 'Syne', sans-serif;
  font-size: 2.4rem;
  font-weight: 800;
  line-height: 1;
}
.verdict.positive { color: #34d399; }
.verdict.negative { color: #fb7185; }

/* ─── confidence ring (SVG wrapper) ─── */
.ring-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}
.ring-label { font-size: .8rem; color: #64748b; letter-spacing: .06em; text-transform: uppercase; }
.ring-pct { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; }

/* ─── keyword chips ─── */
.chip-pos {
  display: inline-block;
  background: rgba(16,185,129,.1);
  border: 1px solid rgba(16,185,129,.3);
  color: #6ee7b7;
  border-radius: 8px;
  padding: 3px 12px;
  margin: 3px;
  font-size: .78rem;
  font-family: monospace;
}
.chip-neg {
  display: inline-block;
  background: rgba(244,63,94,.1);
  border: 1px solid rgba(244,63,94,.3);
  color: #fda4af;
  border-radius: 8px;
  padding: 3px 12px;
  margin: 3px;
  font-size: .78rem;
  font-family: monospace;
}

/* ─── stat mini ─── */
.mini-stat {
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 12px;
  padding: .9rem 1.1rem;
  text-align: center;
}
.mini-stat-val { font-family: 'Syne', sans-serif; font-size: 1.5rem; font-weight: 700; color: #a5b4fc; }
.mini-stat-lbl { font-size: .72rem; color: #475569; text-transform: uppercase; letter-spacing: .08em; margin-top: 2px; }

/* ─── pipeline steps ─── */
.step-card {
  background: rgba(255,255,255,.025);
  border: 1px solid rgba(255,255,255,.06);
  border-radius: 14px;
  padding: 1.2rem;
  transition: border-color .25s;
}
.step-card:hover { border-color: rgba(99,102,241,.3); }
.step-icon { font-size: 2rem; margin-bottom: .5rem; }
.step-title { font-weight: 700; font-size: .95rem; color: #e2e8f0; margin-bottom: .35rem; }
.step-desc { font-size: .8rem; color: #64748b; line-height: 1.55; }

/* ─── cleaned text box ─── */
.cleaned-box {
  background: #0d1117;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 10px;
  padding: 1rem;
  font-family: monospace;
  font-size: .82rem;
  color: #64748b;
  max-height: 180px;
  overflow-y: auto;
  line-height: 1.7;
}

/* ─── expander ─── */
details { border: 1px solid rgba(255,255,255,.06) !important; border-radius: 12px !important; }
summary { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    base = os.path.dirname(__file__)
    mp = os.path.join(base, "models", "sentiment_model.pkl")
    ep = os.path.join(base, "models", "label_encoder.pkl")
    if not os.path.exists(mp):
        return None, None
    return joblib.load(mp), joblib.load(ep)

model, le = load_model()


def get_signal_words(cleaned, top_n=8):
    try:
        tfidf = model.named_steps["tfidf"]
        clf   = model.named_steps["classifier"]
        vec   = tfidf.transform([cleaned])
        names = np.array(tfidf.get_feature_names_out())
        coefs = clf.coef_[0]
        nz    = vec.nonzero()[1]
        if not len(nz): return [], []
        w     = np.array(vec[0, nz].toarray()).flatten() * coefs[nz]
        pos   = [n for n in names[nz[np.argsort(w)[::-1][:top_n]]] if " " not in n][:top_n]
        neg   = [n for n in names[nz[np.argsort(w)[:top_n]]] if " " not in n][:top_n]
        return pos, neg
    except Exception:
        return [], []


def confidence_ring_svg(pct: float, color: str, size=130) -> str:
    r = 46; cx = size // 2; cy = size // 2
    circ = 2 * 3.14159 * r
    dash = circ * pct / 100
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,.06)" stroke-width="8"/>
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="8"
        stroke-dasharray="{dash:.1f} {circ:.1f}" stroke-dashoffset="{circ/4:.1f}"
        stroke-linecap="round"/>
      <text x="{cx}" y="{cy+6}" text-anchor="middle" font-family="Syne,sans-serif"
        font-size="18" font-weight="700" fill="{color}">{pct:.0f}%</text>
    </svg>"""


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineScore")
    st.markdown('<hr style="border-color:rgba(255,255,255,.07)">', unsafe_allow_html=True)

    if model:
        st.markdown('<div style="display:flex;align-items:center;gap:8px;background:rgba(16,185,129,.1);border:1px solid rgba(16,185,129,.25);border-radius:10px;padding:10px 14px;font-size:.85rem;color:#34d399">✅ Model ready</div>', unsafe_allow_html=True)
        try:
            vp = os.path.join(os.path.dirname(__file__), "models", "sklearn_version.txt")
            with open(vp) as f: skv = f.read().strip()
            st.caption(f"scikit-learn {skv}")
        except: pass
    else:
        st.error("Model not trained. Run `python src/train_model.py`")

    st.markdown('<hr style="border-color:rgba(255,255,255,.07)">', unsafe_allow_html=True)
    st.markdown("**Dataset stats**")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="mini-stat"><div class="mini-stat-val">50K</div><div class="mini-stat-lbl">Reviews</div></div>', unsafe_allow_html=True)
    c2.markdown('<div class="mini-stat"><div class="mini-stat-val">89%</div><div class="mini-stat-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    st.caption("Balanced · Binary · IMDB Kaggle dataset")

    st.markdown('<hr style="border-color:rgba(255,255,255,.07)">', unsafe_allow_html=True)
    st.markdown("**📂 Batch prediction**")
    st.caption("Upload a CSV with a `review` column.")
    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    if uploaded and model:
        with st.spinner("Processing…"):
            bdf = pd.read_csv(uploaded)
            if "review" not in bdf.columns:
                st.error("CSV needs a `review` column.")
            else:
                bdf["clean"]      = bdf["review"].apply(clean_text)
                preds             = model.predict(bdf["clean"])
                bdf["sentiment"]  = le.inverse_transform(preds)
                bdf["confidence"] = model.predict_proba(bdf["clean"]).max(axis=1).round(3)
                pos_pct           = (bdf["sentiment"] == "positive").mean() * 100
                st.success(f"{len(bdf):,} reviews · {pos_pct:.1f}% positive")
                st.dataframe(bdf[["review","sentiment","confidence"]].head(20), use_container_width=True)
                st.download_button("⬇️ Download results",
                    bdf[["review","sentiment","confidence"]].to_csv(index=False),
                    "predictions.csv", "text/csv")

    st.markdown('<hr style="border-color:rgba(255,255,255,.07)">', unsafe_allow_html=True)
    st.caption("TF-IDF + Logistic Regression · ngram(1,2) · 10K features")


# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">🎬 &nbsp; Sentiment Intelligence</div>
  <div class="hero-title">Know How Any Review Feels</div>
  <div class="hero-sub">
    Machine learning trained on 50,000 IMDB reviews.<br>
    Paste any movie review — get instant sentiment + confidence analysis.
  </div>
</div>
""", unsafe_allow_html=True)

# pipeline badges
badges = [("📓","Jupyter Notebook"),("🔤","TF-IDF"),("🤖","Logistic Regression"),("⚡","Streamlit")]
st.markdown(
    '<div style="text-align:center;margin-bottom:2rem">' +
    "".join(f'<span class="badge">{i} {l}</span>' for i, l in badges) +
    "</div>",
    unsafe_allow_html=True
)

if model is None:
    st.warning("⚠️ Model not found. Run `python src/train_model.py` first.")
    st.stop()

st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)

# ── INPUT ─────────────────────────────────────────────────────────────────────
left, right = st.columns([3, 1], gap="large")

EXAMPLES = {
    "😍 Loved it":  "This film was an absolute masterpiece. Nuanced storytelling, outstanding performances, and breathtaking cinematography. A must-watch that will stay with you long after the credits roll.",
    "😡 Hated it":  "What a colossal waste of time. The plot was incoherent, the acting wooden and unconvincing. I checked my watch four times in the first act alone.",
    "😐 Mixed":     "The visuals are genuinely stunning and the score is magnificent, but the second act drags badly and the ending feels unearned and rushed.",
}

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Enter a movie review")
    e1, e2, e3 = st.columns(3)
    for col, (label, text) in zip([e1, e2, e3], EXAMPLES.items()):
        if col.button(label, key=label):
            st.session_state["_review"] = text

    user_text = st.text_area(
        "Review",
        value=st.session_state.get("_review", ""),
        height=180,
        placeholder="e.g. 'The cinematography was breathtaking but the story felt hollow…'",
        label_visibility="collapsed",
        key="_review_box",
    )
    analyse = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("""
<div class="card">
  <div style="font-size:1.05rem;font-weight:700;margin-bottom:.8rem">💡 Tips</div>
  <ul style="color:#64748b;font-size:.83rem;line-height:1.9;padding-left:1.1rem">
    <li>Write <strong style="color:#94a3b8">2–3+ sentences</strong> for best results</li>
    <li>HTML tags like <code>&lt;br&gt;</code> are auto-stripped</li>
    <li>Works with formal <em>and</em> casual language</li>
    <li>No neutral class — binary only</li>
  </ul>
  <hr style="border-color:rgba(255,255,255,.06);margin:1rem 0">
  <div style="font-size:.78rem;color:#334155;line-height:1.8">
    <code style="color:#818cf8">TF-IDF</code> · 10K features<br>
    <code style="color:#818cf8">ngram_range=(1,2)</code><br>
    <code style="color:#818cf8">LogisticRegression</code><br>
    80/20 stratified split
  </div>
</div>
""", unsafe_allow_html=True)


# ── RESULT ────────────────────────────────────────────────────────────────────
if analyse:
    raw = st.session_state.get("_review_box") or user_text
    if not raw or not raw.strip():
        st.warning("Please enter some review text first.")
    else:
        with st.spinner(""):
            time.sleep(0.25)
            cleaned   = clean_text(raw)
            pred_int  = model.predict([cleaned])[0]
            conf      = float(model.predict_proba([cleaned]).max())
            label     = le.inverse_transform([pred_int])[0]
            pos_words, neg_words = get_signal_words(cleaned)

        is_pos = label == "positive"
        accent = "#34d399" if is_pos else "#fb7185"
        cls    = "positive" if is_pos else "negative"
        icon   = "✅" if is_pos else "❌"
        pct    = conf * 100

        st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Result")

        # result card
        st.markdown(f'<div class="result-wrap {cls}">', unsafe_allow_html=True)

        ra, rb, rc = st.columns([1.4, 1, 1.8], gap="large")
        with ra:
            st.markdown(f"""
<div style="padding:.5rem 0">
  <div style="font-size:.72rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.6rem">Verdict</div>
  <div class="verdict {cls}">{icon} {label.capitalize()}</div>
  <div style="margin-top:.9rem;font-size:.85rem;color:#475569;line-height:1.6">
    The model is <strong style="color:{accent}">{pct:.1f}% confident</strong><br>this review is <strong style="color:{accent}">{label}</strong>.
  </div>
</div>""", unsafe_allow_html=True)

        with rb:
            st.markdown('<div class="ring-wrap">', unsafe_allow_html=True)
            st.markdown('<div class="ring-label">Confidence</div>', unsafe_allow_html=True)
            st.markdown(confidence_ring_svg(pct, accent), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with rc:
            st.markdown('<div style="font-size:.72rem;color:#475569;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.7rem">Key Signal Words</div>', unsafe_allow_html=True)
            if pos_words:
                st.markdown('<div style="font-size:.75rem;color:#34d399;margin-bottom:.3rem">▲ Positive drivers</div>', unsafe_allow_html=True)
                st.markdown(" ".join(f'<span class="chip-pos">{w}</span>' for w in pos_words), unsafe_allow_html=True)
            if neg_words:
                st.markdown('<div style="font-size:.75rem;color:#fb7185;margin-top:.6rem;margin-bottom:.3rem">▼ Negative drivers</div>', unsafe_allow_html=True)
                st.markdown(" ".join(f'<span class="chip-neg">{w}</span>' for w in neg_words), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # result-wrap

        with st.expander("🔬 Inspect cleaned text"):
            st.markdown(
                f'<div class="cleaned-box">{cleaned[:900]}{"…" if len(cleaned)>900 else ""}</div>',
                unsafe_allow_html=True)
            st.caption(f"Original: {len(raw):,} chars → Cleaned: {len(cleaned):,} chars")


# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown('<hr class="glow-divider">', unsafe_allow_html=True)
st.markdown("### 🏗️ How It Works")

steps = [
    ("📓","1 · Notebook","<code>sentiment_analysis.ipynb</code> — 12-phase pipeline: load → EDA → clean → encode → split → train → evaluate → save artifacts."),
    ("🧹","2 · Clean","<code>clean_text()</code> strips HTML tags, entities, URLs, and non-alpha chars before any vectorisation."),
    ("🔤","3 · Vectorise","TF-IDF with 10K features, bigrams <code>(1,2)</code>, log-normalised TF — captures phrases like <em>not good</em> or <em>highly recommend</em>."),
    ("🤖","4 · Classify","Logistic Regression (<code>C=1.0, balanced</code>) outputs a probability per class — fast, interpretable, ~89% accurate."),
]

cols = st.columns(4, gap="medium")
for col, (icon, title, desc) in zip(cols, steps):
    col.markdown(f"""
<div class="step-card">
  <div class="step-icon">{icon}</div>
  <div class="step-title">{title}</div>
  <div class="step-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
