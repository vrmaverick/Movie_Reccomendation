"""
content_retriever.py — Pure BM25 keyword-based retrieval.

Loads pre-built indexes from data/processed/ (never reads the CSV).
Exposes a single async function: retrieve(query, top_k) -> list[dict]

Each result dict contains:
  title, genre, reason  (reason = matched keywords from query)
"""

import pathlib
import pickle
import re
import string

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"


def _load_indexes():
    """Load BM25 index and movies list from data/processed/."""
    bm25_path = PROCESSED_DIR / "bm25.pkl"
    movies_path = PROCESSED_DIR / "movies.pkl"

    if not bm25_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {bm25_path}. Run ingest.py first.")
    if not movies_path.exists():
        raise FileNotFoundError(f"Movies list not found at {movies_path}. Run ingest.py first.")

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(movies_path, "rb") as f:
        movies = pickle.load(f)

    print('Works _load_indexes')
    return bm25, movies


# Module-level cache — loaded once on first call
_bm25 = None
_movies = None


def _get_indexes():
    """Return cached indexes, loading them on first access."""
    global _bm25, _movies
    if _bm25 is None or _movies is None:
        _bm25, _movies = _load_indexes()
        print('works _get_indexes')
    return _bm25, _movies


def _tokenise(text: str) -> list[str]:
    """Lowercase and split query into tokens, stripping punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _matched_keywords(query_tokens: list[str], movie: dict) -> list[str]:
    """
    Return which query tokens appear in the movie's combined text fields.
    Used to build the human-readable reason string.
    """
    combined = " ".join([
        str(movie.get("title", "") or ""),
        str(movie.get("description", "") or ""),
        str(movie.get("listed_in", "") or ""),
        str(movie.get("cast", "") or ""),
    ]).lower()

    return [t for t in query_tokens if t in combined]


async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Run BM25 keyword search against the pre-built index.

    Args:
        query:  Free-text search string from the user.
        top_k:  Number of results to return (default 5).

    Returns:
        List of dicts with keys: title, genre, reason.
        Ordered by descending BM25 score.

    Raises:
        ValueError: If query is empty.
        FileNotFoundError: If indexes have not been built yet.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    bm25, movies = _get_indexes()
    tokens = _tokenise(query)

    scores = bm25.get_scores(tokens)

    # argsort descending, take top_k
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked:
        movie = movies[idx]
        matched = _matched_keywords(tokens, movie)
        reason = (
            f"matched keywords: {', '.join(matched)}"
            if matched
            else "matched by term frequency across description and metadata"
        )
        results.append({
            "title": movie.get("title", "Unknown"),
            "genre": movie.get("listed_in", "Unknown"),
            "reason": reason,
        })

    return results


if __name__ == '__main__': 
    _load_indexes()
    _get_indexes()
    x = _tokenise('Vedant is a good boy')
    print(x)
    y = retrieve("A film with a thriller and horror combination", top_k = 5)
    print(y)