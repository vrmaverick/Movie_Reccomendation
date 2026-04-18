"""
test.py — Benchmark runner: content-based vs hybrid recommendation methods.

Runs 10 synthetic queries through both retrievers, computes 4 metrics per query,
and writes results to data/processed/benchmark_results.json.

Metrics computed per query
--------------------------
1. Score Transparency  — raw numeric scores from each method:
     content_based: top-5 BM25 scores, matched keyword count
     hybrid:        top-5 cosine (FAISS) scores, top-5 RRF scores
2. Result Overlap      — how much both methods agree:
     intersection_count, jaccard_similarity, shared_titles
3. Diversity           — variety within each method's result set:
     genre_diversity (unique listed_in), type_diversity (Movie vs TV Show),
     year_spread (max - min release_year)
4. Latency             — wall-clock milliseconds for each method

Usage
-----
    python src/test.py
"""

import asyncio
import json
import pathlib
import string
import sys
import time
from datetime import datetime, timezone

# ── path setup so src.* imports work when run from project root ─────────────
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── load .env so GROQ_API_KEY is available without manual export ─────────────
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from src.content_retriever import (  # noqa: E402
    _get_indexes as cb_get_indexes,
    _tokenise,
    _matched_keywords,
)
from src.hybrid_retriever import (  # noqa: E402
    _get_indexes as hybrid_get_indexes,
    _faiss_rank,
    _bm25_rank,
    _rrf_fuse,
)
from src.recommender import extract_intent  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "benchmark_results.json"

_JUDGE_SYSTEM_PROMPT = """\
You are a movie recommendation quality evaluator.
Given a user query and a list of recommended movie titles with their genres,
rate each recommendation's relevance to the query on a scale of 1 to 5:
  5 = perfect match — clearly what the user was looking for
  4 = strong match — highly relevant
  3 = partial match — somewhat relevant but misses key aspects
  2 = weak match — loosely related at best
  1 = irrelevant — unrelated to the query

Return ONLY a JSON array of integers (one per movie, same order).
No markdown, no explanation, no code fences. Pure JSON array only."""

# ── 10 synthetic benchmark queries ──────────────────────────────────────────
BENCHMARK_QUERIES: list[str] = [
    "dark psychological thriller with mind-bending plot twists",
    "heartwarming family animation about friendship and courage",
    "true crime documentary about a notorious murder investigation",
    "romantic comedy set in New York with witty dialogue",
    "gritty action film about organised crime and street gangs",
    "Spanish language drama about immigration and cultural identity",
    "stand-up comedy special with sharp social and political commentary",
    "nature documentary exploring deep ocean and marine life",
    "coming-of-age story about teenagers navigating grief and loss",
    "sci-fi series with complex world-building and moral philosophy",
]


JUDGE_MODEL = "llama-3.1-8b-instant"  # or any non-reasoning Groq model

async def llm_judge_quality(query: str, results: list[dict]) -> dict:
    """
    Return relevance ratings 1–5 per movie, as judged by an LLM.
    Output: only JSON array of ints; no reasoning used or stored.
    """
    from src.recommender import _get_client

    numbered = "\n".join(
        f"{i+1}. {r['title']} ({r['genre']})" for i, r in enumerate(results)
    )

    user_msg = (
        f'Query: "{query}"\n\n'
        "Movies:\n"
        f"{numbered}\n\n"
        "Rate each movie's relevance to the query on a scale from 1 to 5.\n"
        "Return ONLY a JSON array of integers, one per movie, same order.\n"
        "Example: [5, 4, 3, 2, 1]"
    )

    client = _get_client()
    raw = None
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=JUDGE_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator. "
                        "You rate movie recommendations from 1 (irrelevant) to 5 (perfect match). "
                        "Always respond with ONLY a JSON array of integers, no explanations."
                    ),
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=64,
        )

        msg = response.choices[0].message
        raw = msg.content
        if isinstance(raw, list):
            raw = "".join(
                (p.text if hasattr(p, "text") else str(p))
                for p in raw
            )
        raw = (raw or "").strip()
        print("LLM raw output:", repr(raw))

        if not raw:
            raise ValueError("Empty content from LLM")

        # If model is well-behaved, raw is already a JSON array like "[5,4,3,2,1]"
        # But we still defensively extract the first [...] block.
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON array found in LLM output: {raw!r}")
        json_str = raw[start:end + 1]

        scores = json.loads(json_str)
        scores = [max(1, min(5, int(s))) for s in scores[: len(results)]]
        scores += [0] * (len(results) - len(scores))

        avg = round(sum(scores) / len(scores), 2) if scores else 0.0
        return {"scores": scores, "avg_score": avg, "judge_failed": False}

    except Exception as e:
        print("LLM judge failed, raw=", repr(raw), "error=", e)
        fallback = [0] * len(results)
        return {"scores": fallback, "avg_score": 0.0, "judge_failed": True}

# ── metric 1 helper: content-based with raw BM25 scores ─────────────────────

async def _cb_retrieve_scored(query: str, top_k: int = 5) -> tuple[list[dict], list[float]]:
    """
    Run BM25 retrieval and return (results, bm25_scores).

    Mirrors content_retriever.retrieve() but also exposes per-result BM25 scores
    and full movie metadata (type, release_year) needed for diversity metrics.
    """
    bm25, movies = cb_get_indexes()
    tokens = _tokenise(query)
    all_scores = bm25.get_scores(tokens)
    ranked = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:top_k]

    results, scores = [], []
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
            "type": movie.get("type", "Unknown"),
            "release_year": movie.get("release_year"),
            "reason": reason,
            "matched_keyword_count": len(matched),
        })
        scores.append(float(all_scores[idx]))

    return results, scores


# ── metric 1 helper: hybrid with raw cosine + RRF scores ────────────────────

async def _hybrid_retrieve_scored(
    query: str, top_k: int = 5
) -> tuple[list[dict], list[float], list[float], dict]:
    """
    Run BM25 + FAISS + RRF pipeline and return:
      (results, cosine_scores, rrf_scores, intent)

    Mirrors hybrid_retriever.retrieve() but surfaces per-result numeric scores
    and the structured intent extracted by the LLM.
    """
    bm25, movies, faiss_index, encoder = hybrid_get_indexes()
    pool = top_k * 10

    intent_task = asyncio.create_task(extract_intent(query))
    faiss_task = asyncio.to_thread(_faiss_rank, query, encoder, faiss_index, pool)

    intent, (faiss_idxs, faiss_scores_raw) = await asyncio.gather(intent_task, faiss_task)

    keywords = intent.get("keywords") or _tokenise(query)
    bm25_idxs = await asyncio.to_thread(_bm25_rank, bm25, keywords, pool)

    fused = _rrf_fuse(bm25_idxs, faiss_idxs)[:top_k]

    faiss_score_map = dict(zip(faiss_idxs, faiss_scores_raw))
    themes = intent.get("themes", [])
    mood = intent.get("mood", "")

    results, cosine_scores, rrf_scores = [], [], []
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
            "type": movie.get("type", "Unknown"),
            "release_year": movie.get("release_year"),
            "reason": reason,
        })
        cosine_scores.append(float(cosine) if cosine is not None else 0.0)
        rrf_scores.append(float(rrf_score))

    return results, cosine_scores, rrf_scores, intent


# ── metric 2: result overlap ──────────────────────────────────────────────────

def compute_overlap(cb_results: list[dict], hybrid_results: list[dict]) -> dict:
    """
    Compute set-level overlap between two result lists.

    Returns intersection_count, jaccard_similarity, and shared_titles.
    Jaccard = |A ∩ B| / |A ∪ B| where A and B are sets of titles.
    """
    cb_titles = {r["title"] for r in cb_results}
    hy_titles = {r["title"] for r in hybrid_results}
    intersection = cb_titles & hy_titles
    union = cb_titles | hy_titles
    jaccard = len(intersection) / len(union) if union else 0.0
    return {
        "intersection_count": len(intersection),
        "jaccard_similarity": round(jaccard, 4),
        "shared_titles": sorted(intersection),
    }


# ── metric 3: diversity ───────────────────────────────────────────────────────

def compute_diversity(results: list[dict]) -> dict:
    """
    Measure variety within a result set.

    genre_diversity : number of unique listed_in values in top-k results
    type_diversity  : number of unique type values (Movie / TV Show)
    year_spread     : max - min release_year; None if years unavailable
    """
    genres = set()
    for r in results:
        # listed_in can be comma-separated (e.g. "Dramas, Crime TV Shows")
        for g in str(r.get("genre", "")).split(","):
            g = g.strip()
            if g and g != "Unknown":
                genres.add(g)

    types = {r.get("type", "Unknown") for r in results if r.get("type") not in (None, "Unknown")}

    years = [
        int(r["release_year"])
        for r in results
        if r.get("release_year") not in (None, "", "nan")
        and str(r["release_year"]).replace(".", "", 1).isdigit()
    ]
    year_spread = (max(years) - min(years)) if len(years) >= 2 else None

    return {
        "genre_diversity": len(genres),
        "unique_genres": sorted(genres),
        "type_diversity": len(types),
        "unique_types": sorted(types),
        "year_spread": year_spread,
    }


# ── per-query benchmark ───────────────────────────────────────────────────────

async def benchmark_query(query: str) -> dict:
    """
    Run one query through both methods, measure all 4 metrics, and return a
    structured result dict ready for JSON serialisation.
    """
    print(f"  Running: {query!r}")

    # ── metric 4: latency ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cb_results, bm25_scores = await _cb_retrieve_scored(query)
    cb_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    hybrid_results, cosine_scores, rrf_scores, intent = await _hybrid_retrieve_scored(query)
    hybrid_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # ── metric 3: LLM-as-judge quality (both lists judged concurrently) ───────
    cb_quality, hybrid_quality = await asyncio.gather(
        llm_judge_quality(query, cb_results),
        llm_judge_quality(query, hybrid_results),
    )

    # ── metric 1: score transparency ─────────────────────────────────────────
    score_transparency = {
        "content_based": {
            "bm25_scores": [round(s, 4) for s in bm25_scores],
            "avg_bm25_score": round(sum(bm25_scores) / len(bm25_scores), 4) if bm25_scores else 0,
            "avg_matched_keywords": round(
                sum(r["matched_keyword_count"] for r in cb_results) / len(cb_results), 2
            ) if cb_results else 0,
        },
        "hybrid": {
            "cosine_scores": [round(s, 4) for s in cosine_scores],
            "avg_cosine_score": round(sum(cosine_scores) / len(cosine_scores), 4) if cosine_scores else 0,
            "rrf_scores": [round(s, 6) for s in rrf_scores],
            "avg_rrf_score": round(sum(rrf_scores) / len(rrf_scores), 6) if rrf_scores else 0,
            "extracted_intent": {
                "mood": intent.get("mood", ""),
                "genre": intent.get("genre", ""),
                "themes": intent.get("themes", []),
                "keywords": intent.get("keywords", []),
            },
        },
    }

    # ── metric 2: result overlap ──────────────────────────────────────────────
    overlap = compute_overlap(cb_results, hybrid_results)

    # ── metric 3: diversity ───────────────────────────────────────────────────
    diversity = {
        "content_based": compute_diversity(cb_results),
        "hybrid": compute_diversity(hybrid_results),
    }

    # strip internal-only keys from results before saving
    def clean(results: list[dict]) -> list[dict]:
        """Remove benchmark-internal fields not part of the public result schema."""
        drop = {"matched_keyword_count"}
        return [{k: v for k, v in r.items() if k not in drop} for r in results]

    return {
        "query": query,
        "metrics": {
            "latency_ms": {
                "content_based": cb_latency_ms,
                "hybrid": hybrid_latency_ms,
            },
            "score_transparency": score_transparency,
            "result_overlap": overlap,
            "diversity": diversity,
            "llm_quality": {
                "content_based": cb_quality,
                "hybrid": hybrid_quality,
            },
        },
        "results": {
            "content_based": clean(cb_results),
            "hybrid": clean(hybrid_results),
        },
    }


# ── aggregate stats across all queries ───────────────────────────────────────

def compute_aggregate(query_results: list[dict]) -> dict:
    """
    Summarise key metrics averaged across all benchmark queries.

    Returns means for latency, Jaccard, scores, and diversity values
    for quick at-a-glance comparison of both methods.
    """
    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    cb_latencies = [q["metrics"]["latency_ms"]["content_based"] for q in query_results]
    hy_latencies = [q["metrics"]["latency_ms"]["hybrid"] for q in query_results]
    jaccards = [q["metrics"]["result_overlap"]["jaccard_similarity"] for q in query_results]
    intersections = [q["metrics"]["result_overlap"]["intersection_count"] for q in query_results]

    cb_bm25 = [q["metrics"]["score_transparency"]["content_based"]["avg_bm25_score"] for q in query_results]
    cb_kw = [q["metrics"]["score_transparency"]["content_based"]["avg_matched_keywords"] for q in query_results]
    hy_cosine = [q["metrics"]["score_transparency"]["hybrid"]["avg_cosine_score"] for q in query_results]
    hy_rrf = [q["metrics"]["score_transparency"]["hybrid"]["avg_rrf_score"] for q in query_results]

    cb_quality_scores = [
        q["metrics"]["llm_quality"]["content_based"]["avg_score"]
        for q in query_results
        if not q["metrics"]["llm_quality"]["content_based"]["judge_failed"]
    ]
    hy_quality_scores = [
        q["metrics"]["llm_quality"]["hybrid"]["avg_score"]
        for q in query_results
        if not q["metrics"]["llm_quality"]["hybrid"]["judge_failed"]
    ]

    cb_genre_div = [q["metrics"]["diversity"]["content_based"]["genre_diversity"] for q in query_results]
    hy_genre_div = [q["metrics"]["diversity"]["hybrid"]["genre_diversity"] for q in query_results]
    cb_year = [q["metrics"]["diversity"]["content_based"]["year_spread"] for q in query_results if q["metrics"]["diversity"]["content_based"]["year_spread"] is not None]
    hy_year = [q["metrics"]["diversity"]["hybrid"]["year_spread"] for q in query_results if q["metrics"]["diversity"]["hybrid"]["year_spread"] is not None]

    return {
        "num_queries": len(query_results),
        "avg_latency_ms": {
            "content_based": mean(cb_latencies),
            "hybrid": mean(hy_latencies),
            "speedup_ratio": round(mean(hy_latencies) / mean(cb_latencies), 2) if mean(cb_latencies) > 0 else None,
        },
        "avg_result_overlap": {
            "jaccard_similarity": mean(jaccards),
            "intersection_count": mean(intersections),
        },
        "avg_score_transparency": {
            "content_based": {
                "avg_bm25_score": mean(cb_bm25),
                "avg_matched_keywords": mean(cb_kw),
            },
            "hybrid": {
                "avg_cosine_score": mean(hy_cosine),
                "avg_rrf_score": mean(hy_rrf),
            },
        },
        "avg_diversity": {
            "content_based": {
                "genre_diversity": mean(cb_genre_div),
                "year_spread": mean(cb_year),
            },
            "hybrid": {
                "genre_diversity": mean(hy_genre_div),
                "year_spread": mean(hy_year),
            },
        },
        "avg_llm_quality": {
            "content_based": mean(cb_quality_scores),
            "hybrid": mean(hy_quality_scores),
            "queries_judged": len(cb_quality_scores),
        },
    }


# ── main entry point ──────────────────────────────────────────────────────────

async def main() -> None:
    """
    Run all benchmark queries, compute metrics, and write to benchmark_results.json.

    Queries run sequentially (not concurrently) to keep latency measurements
    independent — concurrent execution would share I/O and CPU with each other,
    making latency numbers unrepresentative of single-user response times.
    """
    print(f"Starting benchmark — {len(BENCHMARK_QUERIES)} queries\n")

    query_results = []
    for i, query in enumerate(BENCHMARK_QUERIES, start=1):
        print(f"[{i}/{len(BENCHMARK_QUERIES)}]", end=" ")
        result = await benchmark_query(query)
        query_results.append(result)

    aggregate = compute_aggregate(query_results)

    output = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "num_queries": len(BENCHMARK_QUERIES),
        "aggregate": aggregate,
        "queries": query_results,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nBenchmark complete. Results written to:\n  {OUTPUT_PATH}")
    print("\n── Aggregate Summary ──────────────────────────────────────")
    print(f"  Avg latency  content-based : {aggregate['avg_latency_ms']['content_based']} ms")
    print(f"  Avg latency  hybrid        : {aggregate['avg_latency_ms']['hybrid']} ms")
    print(f"  Hybrid is {aggregate['avg_latency_ms']['speedup_ratio']}x slower than content-based")
    print(f"  Avg Jaccard similarity      : {aggregate['avg_result_overlap']['jaccard_similarity']}")
    print(f"  Avg overlap (shared titles) : {aggregate['avg_result_overlap']['intersection_count']}")
    print(f"  Avg genre diversity  CB / H : {aggregate['avg_diversity']['content_based']['genre_diversity']} / {aggregate['avg_diversity']['hybrid']['genre_diversity']}")
    print(f"  Avg cosine score (hybrid)   : {aggregate['avg_score_transparency']['hybrid']['avg_cosine_score']}")
    print(f"  Avg BM25 score  (content)   : {aggregate['avg_score_transparency']['content_based']['avg_bm25_score']}")
    print(f"\n── LLM Quality Scores (1–5, higher = more relevant) ───────")
    print(f"  Avg quality  content-based : {aggregate['avg_llm_quality']['content_based']}")
    print(f"  Avg quality  hybrid        : {aggregate['avg_llm_quality']['hybrid']}")
    print(f"  Queries judged             : {aggregate['avg_llm_quality']['queries_judged']}")


if __name__ == "__main__":
    asyncio.run(main())
