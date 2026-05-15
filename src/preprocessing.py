"""
preprocessing.py — HTML-aware text cleaning for IMDB movie reviews.

Fixes vs a naive approach:
  - HTML tags are stripped FIRST (critical for IMDB which contains <br />, <p>, etc.)
  - HTML entities decoded (&amp;, &#39;, &lt;, &gt;, &quot;)
  - URLs, @mentions, and hashtag symbols removed
  - Non-alphabetic characters removed (keeps bigram-friendly spacing)
  - Extra whitespace collapsed
"""

import re


def clean_text(text: str) -> str:
    """
    Clean raw IMDB review text.

    Parameters
    ----------
    text : str
        Raw review string (may contain HTML tags and entities).

    Returns
    -------
    str
        Cleaned, lowercased, alphabetic-only text suitable for TF-IDF.
    """
    text = str(text)

    # 1. Strip HTML tags — CRITICAL for IMDB (contains <br />, <p>, <i>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )

    # 3. Lowercase
    text = text.lower()

    # 4. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # 5. Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # 6. Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # 7. Keep only alphabetic characters (removes digits, punctuation)
    text = re.sub(r"[^a-z\s]", "", text)

    # 8. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
