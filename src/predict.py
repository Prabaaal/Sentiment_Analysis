"""
predict.py — Prediction helper for IMDB Sentiment Analysis.

Mirrors Phase 11 of sentiment_analysis.ipynb.

Usage:
    from src.predict import load_model, predict_sentiment

    model, le = load_model()
    label, confidence = predict_sentiment("This film was absolutely brilliant!", model, le)
    print(label, confidence)   # → 'positive', 0.97
"""

import os
import joblib

# Allow running from any directory
BASE_DIR     = os.path.join(os.path.dirname(__file__), "..")
MODEL_PATH   = os.path.join(BASE_DIR, "models", "sentiment_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

# Lazy-cached references (populated by load_model / module-level predict_sentiment)
_model = None
_le    = None


def load_model(model_path: str = MODEL_PATH, encoder_path: str = ENCODER_PATH):
    """
    Load and return (model_pipeline, label_encoder) from disk.

    Raises
    ------
    FileNotFoundError
        If model files are missing — prompt user to run train_model.py first.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at '{model_path}'. "
            "Run `python src/train_model.py` to train the model first."
        )
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(
            f"Label encoder not found at '{encoder_path}'. "
            "Run `python src/train_model.py` to train the model first."
        )

    model = joblib.load(model_path)
    le    = joblib.load(encoder_path)
    return model, le


def predict_sentiment(raw_text: str, model=None, le=None) -> tuple:
    """
    Predict sentiment from raw (uncleaned) input text.

    Mirrors Phase 11 of sentiment_analysis.ipynb — applies clean_text()
    before calling model.predict() to match training-time preprocessing.

    Parameters
    ----------
    raw_text : str
        Raw review text (may contain HTML tags, punctuation, etc.)
    model    : sklearn Pipeline, optional
        Loaded model pipeline. If None, loads from disk.
    le       : LabelEncoder, optional
        Loaded label encoder. If None, loads from disk.

    Returns
    -------
    tuple[str, float]
        (sentiment_label, confidence)  e.g. ("positive", 0.97)

    Raises
    ------
    ValueError
        If raw_text is empty after stripping.
    """
    # Import here to avoid circular import issues at module level
    from src.preprocessing import clean_text

    global _model, _le
    if model is None or le is None:
        if _model is None or _le is None:
            _model, _le = load_model()
        model, le = _model, _le

    if not raw_text or not raw_text.strip():
        raise ValueError("Input text cannot be empty.")

    cleaned      = clean_text(raw_text)
    pred_int     = model.predict([cleaned])[0]
    confidence   = float(model.predict_proba([cleaned]).max())
    label        = le.inverse_transform([pred_int])[0]

    return label, confidence


def batch_predict(texts: list, model=None, le=None) -> list:
    """
    Predict sentiment for a list of raw texts.

    Parameters
    ----------
    texts : list[str]
        List of raw review strings.
    model : sklearn Pipeline, optional
    le    : LabelEncoder, optional

    Returns
    -------
    list[dict]
        Each dict has keys: 'label', 'confidence'
    """
    from src.preprocessing import clean_text

    global _model, _le
    if model is None or le is None:
        if _model is None or _le is None:
            _model, _le = load_model()
        model, le = _model, _le

    cleaned_texts = [clean_text(t) for t in texts]
    preds         = model.predict(cleaned_texts)
    probas        = model.predict_proba(cleaned_texts).max(axis=1)
    labels        = le.inverse_transform(preds)

    return [
        {"label": lbl, "confidence": float(conf)}
        for lbl, conf in zip(labels, probas)
    ]


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_inputs = [
        "This movie was absolutely brilliant. The acting was superb!",
        "Terrible film. Waste of two hours. The plot made no sense at all.",
        "The cinematography was stunning but the story felt hollow.<br />",
        "An average film, nothing special but not unwatchable either.",
    ]

    print(f"{'Review':<65} {'Sentiment':<12} {'Confidence'}")
    print("-" * 90)
    for text in test_inputs:
        label, conf = predict_sentiment(text)
        print(f"{text[:63]:<65} {label:<12} {conf:.2%}")
