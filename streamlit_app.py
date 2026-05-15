import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import joblib, numpy as np, pandas as pd, streamlit as st
from src.preprocessing import clean_text

st.set_page_config(page_title="CineScore · Sentiment AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: #f5f5f7 !important;
  color: #1d1d1f;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── NAV ── */
.nav-bar {
  background: rgba(255,255,255,0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid #d6d6d6;
  padding: 0 40px;
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
  margin-bottom: 0;
}
.nav-logo {
  font-size: 20px;
  font-weight: 600;
  color: #1d1d1f;
  letter-spacing: -0.22px;
}
.nav-links {
  display: flex;
  gap: 32px;
}
.nav-link {
  font-size: 14px;
  font-weight: 400;
  color: #1d1d1f;
  letter-spacing: -0.18px;
  text-decoration: none;
}

/* ── HERO ── */
.hero-section {
  background: #f5f5f7;
  text-align: center;
  padding: 80px 40px 60px;
  border-bottom: 1px solid #d6d6d6;
}
.hero-eyebrow {
  font-size: 18px;
  font-weight: 600;
  color: #1d1d1f;
  letter-spacing: -0.22px;
  margin-bottom: 8px;
}
.hero-title {
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
  font-size: 56px;
  font-weight: 700;
  line-height: 1.07;
  letter-spacing: -0.28px;
  color: #1d1d1f;
  margin-bottom: 16px;
}
.hero-sub {
  font-size: 21px;
  font-weight: 400;
  color: #707070;
  line-height: 1.19;
  letter-spacing: -0.28px;
  max-width: 600px;
  margin: 0 auto 32px;
}
.hero-divider {
  height: 1px;
  background: #d6d6d6;
  border: none;
  margin: 0;
}

/* ── BUTTONS ── */
.btn-primary {
  display: inline-block;
  background: #0071e3;
  color: #fff !important;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.19px;
  padding: 11px 21px;
  border-radius: 980px;
  border: none;
  cursor: pointer;
  text-decoration: none;
  transition: background 0.2s;
}
.btn-primary:hover { background: #0077ed; }

.btn-outline {
  display: inline-block;
  background: transparent;
  color: #0066cc !important;
  font-size: 17px;
  font-weight: 400;
  letter-spacing: -0.19px;
  padding: 11px 21px;
  border-radius: 980px;
  border: 1px solid #0066cc;
  cursor: pointer;
  text-decoration: none;
  transition: all 0.2s;
}
.btn-outline:hover { background: rgba(0,102,204,0.06); }

/* Override Streamlit buttons */
.stButton > button {
  background: transparent !important;
  color: #0066cc !important;
  border: 1px solid #0066cc !important;
  border-radius: 980px !important;
  font-size: 14px !important;
  font-weight: 400 !important;
  padding: 8px 20px !important;
  letter-spacing: -0.18px !important;
  transition: all 0.2s !important;
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
}
.stButton > button:hover {
  background: rgba(0,102,204,0.06) !important;
}

button[kind="primary"] {
  background: #0071e3 !important;
  color: #fff !important;
  border: none !important;
  border-radius: 980px !important;
  font-size: 17px !important;
  font-weight: 400 !important;
  letter-spacing: -0.19px !important;
  padding: 11px 21px !important;
  box-shadow: none !important;
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
}
button[kind="primary"]:hover {
  background: #0077ed !important;
  transform: none !important;
}

/* ── CONTENT SECTION ── */
.content-section {
  background: #ffffff;
  padding: 70px 40px;
  border-bottom: 1px solid #d6d6d6;
}
.section-title {
  font-size: 40px;
  font-weight: 700;
  color: #1d1d1f;
  letter-spacing: 0.44px;
  text-align: center;
  margin-bottom: 8px;
}
.section-sub {
  font-size: 21px;
  font-weight: 400;
  color: #707070;
  letter-spacing: -0.28px;
  text-align: center;
  margin-bottom: 48px;
}

/* ── INPUT CARD ── */
.input-section {
  background: #f5f5f7;
  padding: 70px 40px;
  border-bottom: 1px solid #d6d6d6;
}

.stTextArea textarea {
  background: #ffffff !important;
  border: 1px solid #d6d6d6 !important;
  border-radius: 8px !important;
  color: #1d1d1f !important;
  font-size: 17px !important;
  line-height: 1.47 !important;
  letter-spacing: -0.18px !important;
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
  padding: 16px !important;
  box-shadow: rgba(0,0,0,0.22) 3px 5px 30px 0px;
  transition: border 0.2s, box-shadow 0.2s !important;
}
.stTextArea textarea:focus {
  border-color: #2997ff !important;
  box-shadow: 0 0 0 3px rgba(41,151,255,0.18), rgba(0,0,0,0.22) 3px 5px 30px 0px !important;
  outline: none !important;
}

/* ── STAT TILES ── */
.stat-tile {
  background: #ffffff;
  border: 1px solid #d6d6d6;
  border-radius: 8px;
  padding: 24px;
  text-align: center;
  box-shadow: rgba(0,0,0,0.22) 3px 5px 30px 0px;
}
.stat-val {
  font-size: 34px;
  font-weight: 700;
  color: #0071e3;
  letter-spacing: -0.1px;
  line-height: 1;
}
.stat-lbl {
  font-size: 14px;
  font-weight: 400;
  color: #707070;
  letter-spacing: -0.18px;
  margin-top: 8px;
}

/* ── RESULT CARD ── */
.result-card {
  background: #ffffff;
  border: 1px solid #d6d6d6;
  border-radius: 8px;
  padding: 40px;
  box-shadow: rgba(0,0,0,0.22) 3px 5px 30px 0px;
  margin-top: 40px;
}
.result-label {
  font-size: 12px;
  font-weight: 600;
  color: #707070;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.verdict {
  font-size: 40px;
  font-weight: 700;
  line-height: 1.1;
  letter-spacing: 0.44px;
}
.verdict-positive { color: #1d1d1f; }
.verdict-negative { color: #1d1d1f; }
.verdict-accent-positive { color: #0071e3; }
.verdict-accent-negative { color: #1d1d1f; }

.confidence-bar-bg {
  background: #e2e2e5;
  border-radius: 999px;
  height: 8px;
  margin-top: 12px;
}
.confidence-bar-fill {
  height: 8px;
  border-radius: 999px;
  transition: width 0.5s ease;
}

/* ── KEYWORD CHIPS ── */
.chip-pos {
  display: inline-block;
  background: #f5f5f7;
  border: 1px solid #d6d6d6;
  color: #0071e3;
  border-radius: 999px;
  padding: 4px 14px;
  margin: 3px;
  font-size: 12px;
  letter-spacing: -0.15px;
}
.chip-neg {
  display: inline-block;
  background: #f5f5f7;
  border: 1px solid #d6d6d6;
  color: #474747;
  border-radius: 999px;
  padding: 4px 14px;
  margin: 3px;
  font-size: 12px;
  letter-spacing: -0.15px;
}

/* ── PIPELINE STEPS ── */
.step-card {
  background: #ffffff;
  border: 1px solid #d6d6d6;
  border-radius: 8px;
  padding: 24px;
  box-shadow: rgba(0,0,0,0.22) 3px 5px 30px 0px;
  height: 100%;
}
.step-num {
  font-size: 12px;
  font-weight: 600;
  color: #0071e3;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.step-title {
  font-size: 21px;
  font-weight: 600;
  color: #1d1d1f;
  letter-spacing: -0.28px;
  margin-bottom: 10px;
}
.step-desc {
  font-size: 14px;
  font-weight: 400;
  color: #707070;
  letter-spacing: -0.18px;
  line-height: 1.47;
}

/* ── BATCH SECTION ── */
.pale-section {
  background: #9fc6f4;
  padding: 70px 40px;
  border-bottom: 1px solid #d6d6d6;
}
.pale-section .section-title { color: #1d1d1f; }
.pale-section .section-sub { color: #333333; }

/* ── FOOTER ── */
.footer {
  background: #f5f5f7;
  border-top: 1px solid #d6d6d6;
  padding: 24px 40px;
  text-align: center;
}
.footer-text {
  font-size: 12px;
  color: #707070;
  letter-spacing: -0.15px;
  line-height: 1.5;
}

/* Cleaned text expander */
details {
  border: 1px solid #d6d6d6 !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  padding: 4px !important;
}
summary { color: #474747 !important; font-size: 14px !important; padding: 8px !important; }
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


def get_signal_words(cleaned, top_n=6):
    try:
        tfidf = model.named_steps["tfidf"]
        clf   = model.named_steps["classifier"]
        vec   = tfidf.transform([cleaned])
        names = np.array(tfidf.get_feature_names_out())
        coefs = clf.coef_[0]
        nz    = vec.nonzero()[1]
        if not len(nz): return [], []
        w   = np.array(vec[0, nz].toarray()).flatten() * coefs[nz]
        pos = [n for n in names[nz[np.argsort(w)[::-1][:top_n]]] if " " not in n][:top_n]
        neg = [n for n in names[nz[np.argsort(w)[:top_n]]] if " " not in n][:top_n]
        return pos, neg
    except Exception:
        return [], []


# ── NAV ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">🎬 CineScore</div>
  <div class="nav-links">
    <span class="nav-link">Analyser</span>
    <span class="nav-link">Batch</span>
    <span class="nav-link">How It Works</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ──────────────────────────────────────────────────────────────────────
model_status = "✓ Model ready" if model else "⚠ Model not trained"
st.markdown(f"""
<div class="hero-section">
  <div class="hero-eyebrow">Sentiment AI</div>
  <div class="hero-title">Know How Any Review Feels.</div>
  <div class="hero-sub">Machine learning trained on 50,000 IMDB reviews.<br>
  Instant sentiment analysis with confidence scoring.</div>
  <div style="display:flex;gap:10px;justify-content:center;margin-bottom:24px;">
    <a class="btn-primary" href="#analyser">Analyse a Review</a>
    <a class="btn-outline" href="#how-it-works">Learn More</a>
  </div>
  <div style="font-size:12px;color:#707070;letter-spacing:-0.15px;">{model_status}</div>
</div>
<hr class="hero-divider">
""", unsafe_allow_html=True)

if model is None:
    st.error("Model not found. Run `python src/train_model.py` first.")
    st.stop()


# ── STATS ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="content-section">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4, gap="large")
stats = [("50K", "Training Reviews"), ("89%", "Test Accuracy"), ("10K", "TF-IDF Features"), ("2", "Classes")]
for col, (val, lbl) in zip([c1, c2, c3, c4], stats):
    col.markdown(f'<div class="stat-tile"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── INPUT ─────────────────────────────────────────────────────────────────────
st.markdown('<div id="analyser" class="input-section">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;margin-bottom:40px;">
  <div class="section-title">Analyse a Review</div>
  <div class="section-sub">Paste any movie review and get instant sentiment classification.</div>
</div>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Loved it":  "This film was an absolute masterpiece. Nuanced storytelling, outstanding performances, and breathtaking cinematography. A must-watch.",
    "Hated it":  "What a colossal waste of time. The plot was incoherent, the acting wooden. I checked my watch four times in the first act alone.",
    "Mixed":     "The visuals are genuinely stunning and the score is magnificent, but the second act drags badly and the ending feels unearned.",
}

left, right = st.columns([2, 1], gap="large")

with left:
    st.markdown('<div style="margin-bottom:12px;display:flex;gap:10px;">', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    for col, label in zip([e1, e2, e3], EXAMPLES):
        if col.button(label, key=label):
            st.session_state["_review"] = EXAMPLES[label]
    st.markdown('</div>', unsafe_allow_html=True)

    user_text = st.text_area(
        "Review",
        value=st.session_state.get("_review", ""),
        height=200,
        placeholder="e.g. 'The cinematography was breathtaking but the story felt hollow…'",
        label_visibility="collapsed",
        key="_review_box",
    )
    analyse = st.button("Analyse Sentiment", type="primary", use_container_width=True)

with right:
    st.markdown("""
<div class="step-card">
  <div class="step-num">Tips</div>
  <div class="step-title">Get better results</div>
  <div class="step-desc">
    Write 2–3+ sentences for best accuracy.<br><br>
    HTML tags like &lt;br&gt; are auto-stripped before processing.<br><br>
    Works with formal and casual language.<br><br>
    Binary classification only — Positive or Negative.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# ── RESULT ────────────────────────────────────────────────────────────────────
if analyse:
    raw = st.session_state.get("_review_box") or user_text
    if not raw or not raw.strip():
        st.warning("Please enter some review text first.")
    else:
        with st.spinner(""):
            time.sleep(0.2)
            cleaned   = clean_text(raw)
            pred_int  = model.predict([cleaned])[0]
            conf      = float(model.predict_proba([cleaned]).max())
            label     = le.inverse_transform([pred_int])[0]
            pos_words, neg_words = get_signal_words(cleaned)

        is_pos = label == "positive"
        pct    = conf * 100
        icon   = "✓" if is_pos else "✕"
        bar_color = "#0071e3" if is_pos else "#474747"

        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        ra, rb, rc = st.columns([1.5, 1, 1.5], gap="large")

        with ra:
            st.markdown(f"""
<div class="result-label">Verdict</div>
<div class="verdict">{icon} {label.capitalize()}</div>
<div style="margin-top:16px;">
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>
  </div>
  <div style="font-size:14px;color:#707070;letter-spacing:-0.18px;margin-top:8px;">
    {pct:.1f}% confidence
  </div>
</div>
""", unsafe_allow_html=True)

        with rb:
            st.markdown(f"""
<div class="result-label">Model Confidence</div>
<div style="font-size:56px;font-weight:700;color:{bar_color};line-height:1.07;letter-spacing:-0.28px;">{pct:.0f}%</div>
<div style="font-size:14px;color:#707070;margin-top:8px;letter-spacing:-0.18px;">
  {'Strong signal' if pct > 80 else 'Moderate signal' if pct > 65 else 'Weak signal'}
</div>
""", unsafe_allow_html=True)

        with rc:
            st.markdown('<div class="result-label">Key Signal Words</div>', unsafe_allow_html=True)
            if pos_words:
                st.markdown('<div style="font-size:12px;color:#0071e3;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">Positive drivers</div>', unsafe_allow_html=True)
                st.markdown(" ".join(f'<span class="chip-pos">{w}</span>' for w in pos_words), unsafe_allow_html=True)
            if neg_words:
                st.markdown('<div style="font-size:12px;color:#474747;letter-spacing:0.08em;text-transform:uppercase;margin:10px 0 6px;">Negative drivers</div>', unsafe_allow_html=True)
                st.markdown(" ".join(f'<span class="chip-neg">{w}</span>' for w in neg_words), unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Inspect cleaned text"):
            st.markdown(
                f'<div style="font-family:monospace;font-size:13px;color:#474747;line-height:1.7;padding:16px;background:#f5f5f7;border-radius:8px;">{cleaned[:900]}{"…" if len(cleaned)>900 else ""}</div>',
                unsafe_allow_html=True)
            st.caption(f"Original: {len(raw):,} chars → Cleaned: {len(cleaned):,} chars")

        st.markdown('</div>', unsafe_allow_html=True)


# ── HOW IT WORKS ──────────────────────────────────────────────────────────────
st.markdown('<div id="how-it-works" class="content-section">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;margin-bottom:48px;">
  <div class="section-title">How It Works.</div>
  <div class="section-sub">A four-phase ML pipeline, built end to end.</div>
</div>
""", unsafe_allow_html=True)

steps = [
    ("01", "Data & EDA", "50,000 IMDB reviews. Class-balanced, HTML-aware cleaning strips tags, entities, and noise before any vectorisation."),
    ("02", "TF-IDF", "10K features, bigrams (1,2), log-normalised TF. Captures phrases like 'not good' and 'highly recommend'."),
    ("03", "Logistic Regression", "C=1.0, class_weight=balanced. Fast, interpretable, and achieves ~89% accuracy on the held-out test set."),
    ("04", "Live Inference", "Raw text → clean → vectorise → predict → decode. Full pipeline runs in milliseconds on every request."),
]
c1, c2, c3, c4 = st.columns(4, gap="large")
for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
    col.markdown(f"""
<div class="step-card">
  <div class="step-num">{num}</div>
  <div class="step-title">{title}</div>
  <div class="step-desc">{desc}</div>
</div>""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# ── BATCH ─────────────────────────────────────────────────────────────────────
st.markdown('<div id="batch" class="pale-section">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;margin-bottom:40px;">
  <div class="section-title">Batch Prediction.</div>
  <div class="section-sub">Upload a CSV with a <code style="font-size:18px;background:rgba(0,0,0,0.08);padding:2px 8px;border-radius:4px;">review</code> column and classify thousands of reviews instantly.</div>
</div>
""", unsafe_allow_html=True)

_, center, _ = st.columns([1, 2, 1])
with center:
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded and model:
        with st.spinner("Processing…"):
            bdf = pd.read_csv(uploaded)
            if "review" not in bdf.columns:
                st.error("CSV must have a 'review' column.")
            else:
                bdf["clean"]      = bdf["review"].apply(clean_text)
                preds             = model.predict(bdf["clean"])
                bdf["sentiment"]  = le.inverse_transform(preds)
                bdf["confidence"] = model.predict_proba(bdf["clean"]).max(axis=1).round(3)
                pos_pct           = (bdf["sentiment"] == "positive").mean() * 100
                st.success(f"{len(bdf):,} reviews classified — {pos_pct:.1f}% positive")
                st.dataframe(bdf[["review","sentiment","confidence"]].head(20), use_container_width=True)
                st.download_button("Download Results",
                    bdf[["review","sentiment","confidence"]].to_csv(index=False),
                    "predictions.csv", "text/csv")

st.markdown('</div>', unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-text">
    CineScore · Sentiment AI &nbsp;|&nbsp; TF-IDF + Logistic Regression &nbsp;|&nbsp; Trained on IMDB 50K Dataset<br>
    <span style="color:#858585;">Copyright © 2025 CineScore. Built with Streamlit.</span>
  </div>
</div>
""", unsafe_allow_html=True)
