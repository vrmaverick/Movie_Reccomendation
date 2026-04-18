"""
hybrid_retriever.py — BM25 + FAISS semantic search fused via Reciprocal Rank Fusion.

Pipeline (per CLAUDE.md):
  1. Use recommender.extract_intent (claude-sonnet-4-6) to extract structured intent
     from the query (mood, genre, themes, keywords)
  2. Run BM25 on extracted keywords  -> ranked list A
  3. Embed original query with sentence-transformers -> FAISS cosine search -> ranked list B
  4. Fuse A and B with Reciprocal Rank Fusion (RRF)
  5. Return top_k results with reason = semantic score + matched themes

Loads pre-built indexes from data/processed/ (never reads the CSV).
Exposes a single async function: retrieve(query, top_k) -> list[dict]
"""

import asyncio
import pathlib
import pickle
import string

import faiss
from sentence_transformers import SentenceTransformer

from src.recommender import extract_intent

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# RRF constant — standard value; dampens the impact of very high ranks
RRF_K = 60

# Module-level cache
_bm25 = None
_movies: list[dict] | None = None
_faiss_index = None
_encoder: SentenceTransformer | None = None


# ---------------------------------------------------------------------------
# Index loading
# ---------------------------------------------------------------------------

def _load_indexes() -> None:
    """Load FAISS, BM25, and movies artifacts from data/processed/ into cache."""
    global _bm25, _movies, _faiss_index, _encoder

    bm25_path = PROCESSED_DIR / "bm25.pkl"
    movies_path = PROCESSED_DIR / "movies.pkl"
    faiss_path = PROCESSED_DIR / "faiss.index"

    for p in (bm25_path, movies_path, faiss_path):
        if not p.exists():
            raise FileNotFoundError(f"Index not found at {p}. Run ingest.py first.")

    with open(bm25_path, "rb") as f:
        _bm25 = pickle.load(f)
    with open(movies_path, "rb") as f:
        _movies = pickle.load(f)

    _faiss_index = faiss.read_index(str(faiss_path))
    _encoder = SentenceTransformer("all-MiniLM-L6-v2")


def _get_indexes():
    """Return cached indexes, loading on first access."""
    if _bm25 is None:
        _load_indexes()
    return _bm25, _movies, _faiss_index, _encoder


# ---------------------------------------------------------------------------
# Step 2 — BM25 on extracted keywords
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    """Lowercase and split text into tokens, stripping punctuation."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [t for t in text.split() if t]


def _bm25_rank(bm25, keywords: list[str], n: int) -> list[int]:
    """
    Run BM25 with *keywords* and return the top-n document indexes
    ordered by descending score.
    """
    tokens = _tokenise(" ".join(keywords)) if keywords else []
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]


# ---------------------------------------------------------------------------
# Step 3 — FAISS semantic search
# ---------------------------------------------------------------------------

def _faiss_rank(
    query: str,
    encoder: SentenceTransformer,
    index: faiss.IndexFlatIP,
    n: int,
) -> tuple[list[int], list[float]]:
    """
    Embed *query* and run nearest-neighbour search against FAISS.

    Returns (doc_indexes, cosine_scores) for the top-n results.
    IndexFlatIP with L2-normalised vectors gives exact cosine similarity.
    """
    embedding = encoder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embedding)
    scores, idxs = index.search(embedding, n)
    return idxs[0].tolist(), scores[0].tolist()


# ---------------------------------------------------------------------------
# Step 4 — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    list_a: list[int],
    list_b: list[int],
    k: int = RRF_K,
) -> list[tuple[int, float]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score for document d = sum over lists L of 1 / (k + rank_in_L(d))
    where rank is 1-based.  Documents absent from a list contribute nothing.

    Returns list of (doc_index, rrf_score) sorted by descending score.
    """
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(list_a, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(list_b, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Run the hybrid BM25 + FAISS + RRF pipeline for *query*.

    Args:
        query:  Free-text search string from the user.
        top_k:  Number of results to return (default 5).

    Returns:
        List of dicts with keys: title, genre, reason.
        Ordered by descending RRF score.

    Raises:
        ValueError: If query is empty.
        FileNotFoundError: If indexes have not been built yet.
        EnvironmentError: If ANTHROPIC_API_KEY is not set.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    bm25, movies, faiss_index, encoder = _get_indexes()

    # Candidate pool — larger than top_k so fusion has signal to rerank from
    pool = top_k * 10

    # Steps 1 & 3 run concurrently: Anthropic API call (I/O) and FAISS
    # embedding + search (CPU) have no dependency on each other.
    intent_task = asyncio.create_task(extract_intent(query))
    faiss_task = asyncio.to_thread(_faiss_rank, query, encoder, faiss_index, pool)

    intent, (faiss_idxs, faiss_scores) = await asyncio.gather(intent_task, faiss_task)

    # Step 2: BM25 on extracted keywords (needs intent to arrive first)
    keywords = intent.get("keywords") or _tokenise(query)
    bm25_idxs = await asyncio.to_thread(_bm25_rank, bm25, keywords, pool)

    # Step 4: RRF fusion, keep top_k
    fused = _rrf_fuse(bm25_idxs, faiss_idxs)[:top_k]

    faiss_score_map = dict(zip(faiss_idxs, faiss_scores))
    themes = intent.get("themes", [])
    mood = intent.get("mood", "")

    results = []
    for doc_id, rrf_score in fused:
        movie = movies[doc_id]
        cosine = faiss_score_map.get(doc_id)

        reason_parts = []
        if cosine is not None:
            reason_parts.append(f"semantic similarity {cosine:.2f}")
        if themes:
            reason_parts.append(f"matched themes: {', '.join(themes)}")
        elif mood:
            reason_parts.append(f"mood: {mood}")
        reason = "; ".join(reason_parts) if reason_parts else f"RRF score {rrf_score:.4f}"

        results.append({
            "title": movie.get("title", "Unknown"),
            "genre": movie.get("listed_in", "Unknown"),
            "reason": reason,
        })

    return results


# if __name__ == "__main__":
#     # Run the async event loop
#     import asyncio
#     import os
#     from src.hybrid_retriever import retrieve

#     async def test_search():
#         # 1. Ensure you have your API key set (required by extract_intent)
#         if "ANTHROPIC_API_KEY" not in os.environ:
#             print("⚠️ Warning: ANTHROPIC_API_KEY not found in environment.")
#             # os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

#         print("--- Starting Hybrid Retrieval Test ---")
        
#         # 2. Define a few test queries
#         test_queries = [
#             "I want a dark psychological thriller set in space",
#             "Looking for a lighthearted romantic comedy for a date night",
#             "Movies about high-stakes bank heists with a twist"
#         ]

#         for query in test_queries:
#             print(f"\n🔍 Query: '{query}'")
#             try:
#                 # 3. Call the main function
#                 results = await retrieve(query, top_k=3)

#                 # 4. Print the results formatted nicely
#                 if not results:
#                     print("   No results found.")
#                 else:
#                     for i, res in enumerate(results, 1):
#                         print(f"   {i}. {res['title']} ({res['genre']})")
#                         print(f"      💡 Reason: {res['reason']}")
            
#             except FileNotFoundError as e:
#                 print(f"   ❌ Error: {e}")
#                 print("   Make sure you have run your ingestion script to build the indexes first!")
#                 break
#             except Exception as e:
#                 print(f"   ❌ Unexpected Error: {e}")

#         asyncio.run(test_search())