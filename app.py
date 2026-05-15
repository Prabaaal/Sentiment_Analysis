"""
app.py — IMDB Sentiment Analyser — Streamlit Web Application

Builds on sentiment_analysis.ipynb pipeline (Phases 1–12):
  - Loads trained model + encoder from models/
  - Applies HTML-aware clean_text() before prediction (Phase 3/11)
  - Shows sentiment label, confidence score, confidence gauge
  - Highlights top positive/negative signal words (TF-IDF feature names)
  - Sidebar: batch CSV upload with downloadable results
  - Sidebar: live dataset stats

Run:
    streamlit run app.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.preprocessing import clean_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IMDB Sentiment Analyser",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "IMDB Sentiment Analyser — TF-IDF + Logistic Regression on 50K reviews.",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@700;800&display=swap');

/* ── Root variables ── */
:root {
    --bg-main:        #0d1117;
    --bg-card:        #161b22;
    --bg-card-hover:  #1c2330;
    --border:         #30363d;
    --accent-blue:    #2196F3;
    --accent-orange:  #FF5722;
    --accent-green:   #4CAF50;
    --text-primary:   #e6edf3;
    --text-muted:     #8b949e;
    --radius:         12px;
    --shadow:         0 4px 24px rgba(0,0,0,0.4);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-primary);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card);
    border-right: 1px solid var(--border);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headline ── */
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #2196F3 0%, #9c27b0 50%, #FF5722 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    color: var(--text-muted);
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 1.5rem;
}

/* ── Card ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: border-color 0.2s ease;
}
.card:hover { border-color: #444d56; }

/* ── Result pill ── */
.sentiment-pill {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 999px;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    margin-bottom: 0.5rem;
}
.positive-pill { background: rgba(76,175,80,0.15); color: #4CAF50; border: 1px solid #4CAF50; }
.negative-pill { background: rgba(244,67,54,0.15); color: #f44336; border: 1px solid #f44336; }

/* ── Gauge bar ── */
.gauge-wrap { margin-top: 0.5rem; }
.gauge-bar {
    height: 10px;
    border-radius: 999px;
    overflow: hidden;
    background: #21262d;
    margin-top: 0.4rem;
}
.gauge-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Keyword badge ── */
.kw-pos {
    display: inline-block;
    background: rgba(33,150,243,0.12);
    color: #64b5f6;
    border: 1px solid rgba(33,150,243,0.35);
    border-radius: 6px;
    padding: 2px 10px;
    margin: 3px;
    font-size: 0.8rem;
    font-family: monospace;
}
.kw-neg {
    display: inline-block;
    background: rgba(255,87,34,0.12);
    color: #ff8a65;
    border: 1px solid rgba(255,87,34,0.35);
    border-radius: 6px;
    padding: 2px 10px;
    margin: 3px;
    font-size: 0.8rem;
    font-family: monospace;
}

/* ── Stat box ── */
.stat-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.25rem;
    text-align: center;
}
.stat-val { font-size: 1.6rem; font-weight: 700; color: var(--accent-blue); }
.stat-lbl { font-size: 0.78rem; color: var(--text-muted); margin-top: 2px; }

/* ── Step badge ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(33,150,243,0.1);
    border: 1px solid rgba(33,150,243,0.25);
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #90caf9;
    margin-bottom: 0.75rem;
}

/* ── Divider ── */
.fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.5rem 0;
}

/* ── Streamlit overrides ── */
.stTextArea textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
}
.stTextArea textarea:focus {
    border-color: #2196F3 !important;
    box-shadow: 0 0 0 2px rgba(33,150,243,0.2) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1565C0, #2196F3) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:active { transform: scale(0.97) !important; }

/* ── Example button ── */
.stButton[data-testid] > button {
    background: #21262d !important;
    color: #90caf9 !important;
    border: 1px solid #30363d !important;
    font-size: 0.82rem !important;
    padding: 0.35rem 1rem !important;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model + encoder ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained pipeline + label encoder. Cached across sessions."""
    model_path   = os.path.join(os.path.dirname(__file__), "models", "sentiment_model.pkl")
    encoder_path = os.path.join(os.path.dirname(__file__), "models", "label_encoder.pkl")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    le    = joblib.load(encoder_path)
    return model, le


model, le = load_model()


# ── Helper: extract top signal words ─────────────────────────────────────────
def get_signal_words(cleaned_text: str, sentiment_class: int, top_n: int = 8):
    """
    Return top TF-IDF feature words from the review that most influenced
    the predicted class (positive-leaning vs negative-leaning).
    """
    if model is None:
        return [], []
    try:
        tfidf = model.named_steps["tfidf"]
        clf   = model.named_steps["classifier"]
        vec   = tfidf.transform([cleaned_text])
        feature_names = np.array(tfidf.get_feature_names_out())
        coefs         = clf.coef_[0]  # positive class coefficients

        # Non-zero indices in this review's TF-IDF vector
        nonzero = vec.nonzero()[1]
        if len(nonzero) == 0:
            return [], []

        # Weight = TF-IDF value × logistic coef
        weights  = np.array(vec[0, nonzero].toarray()).flatten() * coefs[nonzero]
        pos_idx  = nonzero[np.argsort(weights)[::-1][:top_n]]
        neg_idx  = nonzero[np.argsort(weights)[:top_n]]

        pos_words = [w for w in feature_names[pos_idx] if " " not in w][:top_n]
        neg_words = [w for w in feature_names[neg_idx] if " " not in w][:top_n]
        return pos_words, neg_words
    except Exception:
        return [], []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 IMDB Sentiment")
    st.markdown("---")

    # Model status
    if model is not None:
        st.markdown("**Model Status**")
        st.success("✅ Model loaded", icon="✅")
        try:
            version_path = os.path.join(os.path.dirname(__file__), "models", "sklearn_version.txt")
            with open(version_path) as f:
                skv = f.read().strip()
            st.caption(f"scikit-learn {skv}")
        except Exception:
            pass
    else:
        st.error("⚠️ Model not trained yet")
        st.info("Run the notebook or:\n```bash\npython src/train_model.py\n```")

    st.markdown("---")

    # Dataset stats
    st.markdown("**Dataset**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="stat-box"><div class="stat-val">50K</div><div class="stat-lbl">Reviews</div></div>', unsafe_allow_html=True)
    with col_b:
        st.markdown('<div class="stat-box"><div class="stat-val">89–92%</div><div class="stat-lbl">Accuracy</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.caption("IMDB Dataset · TF-IDF + Logistic Regression · `ngram_range=(1,2)`")

    st.markdown("---")

    # ── Batch CSV upload ──────────────────────────────────────────────────
    st.markdown("**📂 Batch Prediction**")
    st.caption("Upload a CSV with a `review` column to analyse multiple reviews at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file and model is not None:
        with st.spinner("Analysing …"):
            batch_df = pd.read_csv(uploaded_file)
            if "review" not in batch_df.columns:
                st.error("CSV must have a `review` column.")
            else:
                batch_df["clean"] = batch_df["review"].apply(clean_text)
                preds = model.predict(batch_df["clean"])
                batch_df["sentiment"]  = le.inverse_transform(preds)
                batch_df["confidence"] = model.predict_proba(batch_df["clean"]).max(axis=1).round(3)

                pos_pct = (batch_df["sentiment"] == "positive").mean() * 100
                st.success(f"✅ {len(batch_df):,} reviews analysed · {pos_pct:.1f}% positive")
                st.dataframe(
                    batch_df[["review", "sentiment", "confidence"]].head(20),
                    use_container_width=True,
                )
                csv_out = batch_df[["review", "sentiment", "confidence"]].to_csv(index=False)
                st.download_button(
                    "⬇️ Download results",
                    data=csv_out,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

    st.markdown("---")
    st.markdown("**🔗 Pipeline**")
    st.markdown("""
- `sentiment_analysis.ipynb` — core ML pipeline  
- `src/preprocessing.py` — `clean_text()`  
- `src/train_model.py` — training script  
- `src/predict.py` — prediction helper  
- `app.py` — this Streamlit app  
""")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 IMDB Sentiment Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Classify movie reviews as <strong>Positive</strong> or <strong>Negative</strong> using TF-IDF + Logistic Regression trained on 50K IMDB reviews.</div>', unsafe_allow_html=True)

# ── Pipeline badge row ─────────────────────────────────────────────────────
badges = [
    ("📓", "Jupyter Notebook"),
    ("🔤", "TF-IDF Vectoriser"),
    ("🤖", "Logistic Regression"),
    ("⚡", "Streamlit App"),
]
badge_html = "".join(
    f'<span class="step-badge"><span>{icon}</span><span>{label}</span></span>&nbsp;'
    for icon, label in badges
)
st.markdown(badge_html, unsafe_allow_html=True)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

if model is None:
    st.warning(
        "⚠️ Model not found. Run the **sentiment_analysis.ipynb** notebook "
        "(Phases 1–10) or execute `python src/train_model.py` first.",
        icon="⚠️",
    )
    st.stop()

# ── Input area ────────────────────────────────────────────────────────────────
col_main, col_tips = st.columns([3, 1], gap="large")

with col_main:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### ✍️ Enter a movie review")

    # Example buttons
    examples = {
        "😍 Loved it": "This film was an absolute masterpiece. The storytelling was nuanced, the performances were outstanding, and the cinematography breathtaking. A must-watch!",
        "😡 Hated it": "What a waste of time. The plot was incoherent, the acting wooden, and the special effects looked cheap. I walked out halfway through.",
        "😐 Mixed": "The cinematography was stunning and some scenes were genuinely moving, but the second act dragged terribly and the ending felt rushed and unsatisfying.",
    }

    ex_col1, ex_col2, ex_col3 = st.columns(3)
    ex_cols = [ex_col1, ex_col2, ex_col3]
    for i, (label, text) in enumerate(examples.items()):
        if ex_cols[i].button(label, key=f"ex_{i}"):
            st.session_state["review_text"] = text

    user_input = st.text_area(
        "Review text",
        value=st.session_state.get("review_text", ""),
        height=170,
        placeholder="e.g. 'The cinematography was breathtaking but the story felt hollow...'",
        label_visibility="collapsed",
        key="review_text_input",
    )

    analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_tips:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 💡 Tips")
    st.markdown("""
**For best results:**

- Write at least **2–3 sentences**
- Use natural language
- HTML tags like `<br>` are auto-stripped
- The model handles both formal and casual language
""")
    st.markdown("---")
    st.markdown("**Model details**")
    st.caption("""
`TF-IDF` with 10K features  
`ngram_range=(1,2)`  
`min_df=3`, `max_df=0.90`  
`LogisticRegression(C=1.0)`  
Trained on 80% of 50K reviews  
""")
    st.markdown('</div>', unsafe_allow_html=True)


# ── Result ────────────────────────────────────────────────────────────────────
if analyse_btn:
    review_text = st.session_state.get("review_text_input") or user_input
    if not review_text or not review_text.strip():
        st.warning("⚠️ Please enter some review text before analysing.")
    else:
        with st.spinner("Analysing …"):
            time.sleep(0.3)  # micro-delay for UX feel
            cleaned_input = clean_text(review_text)
            pred_int      = model.predict([cleaned_input])[0]
            confidence    = float(model.predict_proba([cleaned_input]).max())
            sentiment_label = le.inverse_transform([pred_int])[0]

        st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
        st.markdown("### 📊 Analysis Result")

        # ── Three-column result layout ─────────────────────────────────────
        r1, r2, r3 = st.columns([1, 1, 2], gap="large")

        with r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Sentiment**")
            is_positive = sentiment_label == "positive"
            pill_class  = "positive-pill" if is_positive else "negative-pill"
            pill_icon   = "✅" if is_positive else "❌"
            st.markdown(
                f'<div class="sentiment-pill {pill_class}">{pill_icon} {sentiment_label.capitalize()}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Confidence**")
            pct = confidence * 100
            bar_color = "#4CAF50" if is_positive else "#f44336"
            st.markdown(f'<div style="font-size:2rem;font-weight:700;color:{bar_color}">{pct:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="gauge-bar"><div class="gauge-fill" style="width:{pct}%;background:{bar_color}"></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Key signal words**")
            pos_words, neg_words = get_signal_words(cleaned_input, pred_int)

            if pos_words:
                st.caption("🔵 Positive signals")
                st.markdown(
                    " ".join(f'<span class="kw-pos">{w}</span>' for w in pos_words),
                    unsafe_allow_html=True,
                )
            if neg_words:
                st.caption("🔴 Negative signals")
                st.markdown(
                    " ".join(f'<span class="kw-neg">{w}</span>' for w in neg_words),
                    unsafe_allow_html=True,
                )
            if not pos_words and not neg_words:
                st.caption("No strong signal words found.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Cleaned text expander ──────────────────────────────────────────
        with st.expander("🔍 Inspect cleaned text (what the model sees)"):
            st.markdown(
                f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;'
                f'padding:1rem;font-family:monospace;font-size:0.85rem;color:#8b949e;'
                f'max-height:200px;overflow-y:auto">{cleaned_input[:800]}'
                f'{"…" if len(cleaned_input) > 800 else ""}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"Original length: {len(review_text):,} chars → "
                f"Cleaned length: {len(cleaned_input):,} chars"
            )


# ── How it works ──────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown("### 🏗️ How the Pipeline Works")

ph1, ph2, ph3, ph4 = st.columns(4, gap="medium")

phases = [
    ("📓", "1. Notebook", "**sentiment_analysis.ipynb** runs 12 phases: load → clean → EDA → encode → split → train → evaluate → save"),
    ("🔤", "2. Vectorise", "**TF-IDF** converts cleaned text to a 10K-dim sparse vector. Bigrams capture phrases like *not good* or *highly recommend*."),
    ("🤖", "3. Classify", "**Logistic Regression** (C=1.0, balanced) predicts a probability for each class — fast, interpretable, ~90% accurate."),
    ("⚡", "4. App", "**Streamlit** calls `clean_text()` → `model.predict()` → `le.inverse_transform()` to display human-readable results."),
]

for col, (icon, title, body) in zip([ph1, ph2, ph3, ph4], phases):
    with col:
        st.markdown(
            f'<div class="card" style="min-height:160px">'
            f'<div style="font-size:2rem;margin-bottom:0.5rem">{icon}</div>'
            f'<strong>{title}</strong><br><span style="color:#8b949e;font-size:0.85rem">{body}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
