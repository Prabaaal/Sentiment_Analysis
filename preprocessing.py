import re


def clean_text(text):
    """
    Clean raw IMDB review text.
    Fixes vs original plan:
    - Strip HTML tags (e.g. <br />, <p>) — critical for IMDB dataset
    - Handle HTML entities (&amp;, &lt;, etc.)
    - Remove URLs, mentions, hashtag symbols
    - Lowercase, strip extra whitespace
    """
    text = str(text)
    # 1. Remove HTML tags (NOT in original plan — critical for IMDB)
    text = re.sub(r"<[^>]+>", " ", text)
    # 2. Decode common HTML entities
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'")
    # 3. Lowercase
    text = text.lower()
    # 4. Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    # 5. Remove @mentions
    text = re.sub(r"@\w+", "", text)
    # 6. Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)
    # 7. Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", "", text)
    # 8. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
