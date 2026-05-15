"""
eda.py — Exploratory Data Analysis for the IMDB Sentiment dataset.

Generates and saves:
  - Class distribution bar chart           → notebooks/class_distribution.png
  - Review length distribution             → (displayed inline)
  - Per-class top-word frequency charts    → notebooks/top_words_per_class.png
  - Word clouds (if wordcloud installed)   → notebooks/wordcloud_positive.png
                                             notebooks/wordcloud_negative.png
"""

import os
import sys
import re

# Allow running from project root or from inside src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.preprocessing import clean_text

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH      = os.path.join(os.path.dirname(__file__), "..", "data", "IMDB Dataset.csv")
NOTEBOOKS_DIR  = os.path.join(os.path.dirname(__file__), "..", "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

# ── Stopwords for top-word charts ─────────────────────────────────────────────
STOPWORDS = {
    "film", "movie", "the", "a", "an", "is", "it", "in", "of", "and", "to",
    "was", "that", "this", "i", "for", "with", "he", "she", "they", "on",
    "at", "be", "as", "but", "not", "have", "had", "his", "her", "one",
    "its", "are", "were", "by", "from", "so", "there", "br", "also", "just",
    "do", "if", "my", "me", "we", "or", "all", "would", "been", "which",
    "their", "about", "who", "out", "up", "can", "more", "no", "what", "when",
}


def load_and_clean(path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, drop nulls/duplicates, and apply clean_text."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().dropna(subset=["review", "sentiment"])
    df = df[df["review"].str.strip() != ""].reset_index(drop=True)
    df["clean_text"] = df["review"].apply(clean_text)
    df["text_length"] = df["clean_text"].apply(len)
    return df


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of positive vs negative review counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["sentiment"].value_counts()
    colors = ["#2196F3", "#FF5722"]
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Class Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Review Count")
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(counts):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=11)
    plt.tight_layout()
    out = os.path.join(NOTEBOOKS_DIR, "class_distribution.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved → {out}")


def plot_review_length(df: pd.DataFrame) -> None:
    """Overlaid histogram of review character lengths by sentiment class."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for sentiment, color in [("positive", "#2196F3"), ("negative", "#FF5722")]:
        subset = df[df["sentiment"] == sentiment]["text_length"]
        ax.hist(subset, bins=60, alpha=0.65, color=color, label=sentiment)
    ax.set_title("Review Length Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Character count (after cleaning)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.show()
    print("\nMean review length by sentiment:")
    print(df.groupby("sentiment")["text_length"].mean().round(1))


def plot_top_words_per_class(df: pd.DataFrame, top_n: int = 20) -> None:
    """Horizontal bar charts of top N words for each sentiment class."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, sentiment, color in zip(axes, ["positive", "negative"], ["#2196F3", "#FF5722"]):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_text"])
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
        common = Counter(words).most_common(top_n)
        labels, counts = zip(*common)
        ax.barh(labels[::-1], counts[::-1], color=color, edgecolor="white", alpha=0.9)
        ax.set_title(f"Top {top_n} words — {sentiment} reviews", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frequency")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    plt.tight_layout()
    out = os.path.join(NOTEBOOKS_DIR, "top_words_per_class.png")
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"Saved → {out}")


def plot_word_clouds(df: pd.DataFrame) -> None:
    """Word cloud for each sentiment class (requires wordcloud package)."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed. Run: pip install wordcloud")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sentiment, cmap in zip(
        axes,
        ["positive", "negative"],
        ["RdYlGn", "RdYlBu_r"],
    ):
        text = " ".join(df[df["sentiment"] == sentiment]["clean_text"])
        wc = WordCloud(
            width=700, height=350,
            background_color="white",
            colormap=cmap,
            max_words=150,
        ).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word cloud — {sentiment}", fontsize=13, fontweight="bold")

        out = os.path.join(NOTEBOOKS_DIR, f"wordcloud_{sentiment}.png")
        wc.to_file(out)
        print(f"Saved → {out}")

    plt.tight_layout()
    plt.show()


def run_full_eda() -> pd.DataFrame:
    """Run all EDA steps and return the cleaned dataframe."""
    print("Loading dataset …")
    df = load_and_clean()
    print(f"Shape after cleaning: {df.shape}")
    print(f"\nClass distribution:\n{df['sentiment'].value_counts()}")

    print("\n[1/4] Plotting class distribution …")
    plot_class_distribution(df)

    print("\n[2/4] Plotting review length distribution …")
    plot_review_length(df)

    print("\n[3/4] Plotting top words per class …")
    plot_top_words_per_class(df)

    print("\n[4/4] Plotting word clouds …")
    plot_word_clouds(df)

    return df


if __name__ == "__main__":
    run_full_eda()
