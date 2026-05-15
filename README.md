# 🎬 IMDB Sentiment Analysis — Streamlit App

A machine learning web app that classifies IMDB movie reviews as **positive** or **negative**.

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 reviews, balanced (25k positive / 25k negative), binary classification only.

> **Note:** This dataset does NOT have a Neutral class.

## Tech Stack

| Layer | Library |
|---|---|
| Data | pandas 2.2.2 |
| ML | scikit-learn 1.5.0 (TF-IDF + Logistic Regression) |
| App | Streamlit 1.35.0 |
| Serialisation | joblib 1.4.2 |

## Project Structure

```
Sentiment Analysis/
├── sentiment_analysis.ipynb      # PRIMARY: 12-phase ML pipeline
├── data/
│   └── IMDB Dataset.csv          # Download from Kaggle
├── models/
│   ├── sentiment_model.pkl       # Trained pipeline
│   ├── label_encoder.pkl         # LabelEncoder (negative=0, positive=1)
│   └── sklearn_version.txt       # Version lock for pickle compatibility
├── src/
│   ├── __init__.py               # Package init
│   ├── preprocessing.py          # clean_text() — HTML-aware
│   ├── train_model.py            # Standalone training script (Phases 1-10)
│   ├── predict.py                # Prediction helper (Phase 11)
│   └── eda.py                    # EDA + per-class word frequency
├── notebooks/                    # EDA output images saved here
├── streamlit_app.py              # Streamlit app
├── requirements.txt              # Pinned versions
└── README.md
```

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset from Kaggle and place it at:
#    data/IMDB Dataset.csv

# 3. Train the model (Choose one method)
# Method A: Run the jupyter notebook
jupyter notebook sentiment_analysis.ipynb # Run all cells
# Method B: Run the standalone script
python src/train_model.py

# 4. Launch the app
streamlit run streamlit_app.py
```

## Key Features & Design Decisions

- **Premium UI**: Dark-themed Streamlit app with glassmorphism cards, animated confidence gauge, and top signal word extraction.
- **HTML tag stripping**: IMDB reviews contain `<br />` and other HTML tags. The `clean_text()` function removes these before vectorisation.
- **Label encoding**: String labels (`positive`/`negative`) are encoded to integers via `LabelEncoder` and saved alongside the model for consistent decoding.
- **TF-IDF tuning**: `min_df=3`, `max_df=0.90`, `ngram_range=(1,2)`, `sublinear_tf=True` — reduces noise from very rare and very common terms, adds bigram features.
- **Version pinning**: `requirements.txt` pins all library versions to prevent pickle incompatibility on Streamlit Cloud.
- **Preprocessing at prediction time**: `app.py` and `predict.py` apply `clean_text()` to user input before calling `model.predict()`.
- **Batch Processing**: The Streamlit sidebar allows uploading a CSV of reviews to process them in batch and download the predictions.

## Deployment

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Set main file path: `streamlit_app.py`
5. Deploy
