"""
sentiment.py
--------------
Real NLP-based sentiment analysis using VADER (Valence Aware Dictionary and
sEntiment Reasoner) instead of a hardcoded keyword list.

VADER is a well-known, published sentiment analysis model tuned for
short text like headlines and social media - much easier to defend in a
viva than "I checked if the word 'surge' was in the title."
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def analyze_headlines(news_list, max_articles=5):
    """
    Takes a list of news articles (from yfinance) and returns:
    - overall sentiment label (Bullish / Bearish / Neutral)
    - average compound score (-1 to +1)
    - per-headline breakdown (for display)
    """
    if not news_list:
        return {
            "label": "No Data ⚪",
            "avg_score": 0.0,
            "details": [],
        }

    details = []
    scores = []

    for item in news_list[:max_articles]:
        title = item.get("title", "") if isinstance(item, dict) else str(item)
        publisher = item.get("publisher", "Financial Source") if isinstance(item, dict) else "News Feed"

        if not title:
            continue

        vs = analyzer.polarity_scores(title)
        compound = vs["compound"]
        scores.append(compound)

        if compound >= 0.05:
            headline_label = "Positive 🟢"
        elif compound <= -0.05:
            headline_label = "Negative 🔴"
        else:
            headline_label = "Neutral ⚪"

        details.append({
            "title": title,
            "publisher": publisher,
            "score": compound,
            "label": headline_label,
        })

    if not scores:
        return {"label": "No Data ⚪", "avg_score": 0.0, "details": []}

    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.05:
        overall_label = "Bullish / Positive 🟢"
    elif avg_score <= -0.05:
        overall_label = "Bearish / Negative 🔴"
    else:
        overall_label = "Neutral ⚪"

    return {
        "label": overall_label,
        "avg_score": avg_score,
        "details": details,
    }
