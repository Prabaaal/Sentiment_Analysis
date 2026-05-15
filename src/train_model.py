"""
train_model.py — Standalone training script for IMDB Sentiment Analysis.

Mirrors the pipeline defined in sentiment_analysis.ipynb (Phases 1–10):
  Phase 1  : Imports & Setup
  Phase 2  : Load Dataset
  Phase 3  : Data Cleaning & Preprocessing
  Phase 4  : (EDA skipped in training — see eda.py)
  Phase 5  : Label Encoding
  Phase 6  : Train / Test Split
  Phase 7  : Build ML Pipeline (TF-IDF + Logistic Regression)
  Phase 8  : Train the Model
  Phase 9  : Evaluate
  Phase 10 : Save Model

Usage:
    python src/train_model.py
    # or from project root:
    python -m src.train_model
"""

import os
import sys
import warnings
import joblib
import sklearn

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# Allow running from project root or from inside src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import clean_text

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH  = os.path.join(BASE_DIR, "data", "IMDB Dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "sentiment_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
VERSION_PATH = os.path.join(MODEL_DIR, "sklearn_version.txt")


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Phase 2 & 3: Load CSV, drop nulls/duplicates, apply clean_text."""
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"  Raw shape       : {df.shape}")
    print(f"  Columns         : {df.columns.tolist()}")

    # Phase 3 — clean
    df = df.drop_duplicates().dropna(subset=["review", "sentiment"])
    df = df[df["review"].str.strip() != ""].reset_index(drop=True)
    print(f"  After cleaning  : {df.shape}")

    print("  Applying clean_text() … (this may take ~30 s for 50K rows)")
    df["clean_text"] = df["review"].apply(clean_text)

    # Sanity check — no HTML tags should survive
    assert df["clean_text"].str.contains("<br").sum() == 0, "HTML tags still present!"
    print("  HTML tag check  : PASSED")
    return df


def encode_labels(df: pd.DataFrame) -> tuple:
    """Phase 5: Encode string labels to integers, return df and encoder."""
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["sentiment"])
    print(f"\nLabel mapping   : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Label dist      :\n{df['label'].value_counts()}")
    return df, le


def split_data(df: pd.DataFrame):
    """Phase 6: Stratified 80/20 train-test split."""
    X = df["clean_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size      : {len(X_train):,}")
    print(f"Test size       : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


def build_pipeline() -> Pipeline:
    """
    Phase 7: Build TF-IDF + Logistic Regression pipeline.

    TF-IDF parameters tuned for IMDB:
      - min_df=3       : drops rare noise tokens
      - max_df=0.90    : drops near-universal terms (film, movie …)
      - ngram_range=(1,2): captures "not good", "highly recommend"
      - sublinear_tf=True: log-normalises TF (handles long reviews)
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=10_000,
            min_df=3,
            max_df=0.90,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])


def evaluate(model: Pipeline, le: LabelEncoder, X_test, y_test) -> None:
    """Phase 9: Print accuracy + classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy        : {acc:.4f}  ({acc * 100:.2f}%)\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(cm)


def save_artifacts(model: Pipeline, le: LabelEncoder) -> None:
    """Phase 10: Persist model + encoder + sklearn version."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    with open(VERSION_PATH, "w") as f:
        f.write(sklearn.__version__)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {ENCODER_PATH}")
    print(f"Saved: {VERSION_PATH}  (sklearn {sklearn.__version__})")


def main() -> None:
    print("=" * 60)
    print("  IMDB Sentiment Analysis — Training Pipeline")
    print("=" * 60)

    # Phases 2 & 3
    df = load_and_clean()

    # Phase 5
    df, le = encode_labels(df)

    # Phase 6
    X_train, X_test, y_train, y_test = split_data(df)

    # Phase 7 & 8
    print("\nBuilding pipeline …")
    model_pipeline = build_pipeline()
    print("Training … (may take ~10–30 s)")
    model_pipeline.fit(X_train, y_train)
    print("Training DONE.")

    # Phase 9
    evaluate(model_pipeline, le, X_test, y_test)

    # Phase 10
    save_artifacts(model_pipeline, le)

    print("\n" + "=" * 60)
    print("  Training complete. Run:  streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
