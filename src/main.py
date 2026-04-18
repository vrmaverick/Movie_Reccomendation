"""
main.py — FastAPI application entry point.

Routes:
    GET  /            — serves static/index.html
    POST /recommend   — runs content-based and hybrid retrievers concurrently
                        and returns combined results
"""

import asyncio
import json
import pathlib

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

# Load .env from project root regardless of working directory
_ENV_FILE = pathlib.Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_ENV_FILE)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src.content_retriever import retrieve as content_retrieve
from src.hybrid_retriever import retrieve as hybrid_retrieve

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"
INDEX_HTML = STATIC_DIR / "index.html"
BENCHMARK_JSON = ROOT / "data" / "processed" / "benchmark_results.json"

app = FastAPI(title="Movie Recommender", version="1.0.0")


class RecommendRequest(BaseModel):
    """Request body for POST /recommend."""
    query: str


class MovieResult(BaseModel):
    """A single recommendation result."""
    title: str
    genre: str
    reason: str


class RecommendResponse(BaseModel):
    """Response body for POST /recommend."""
    content_based: list[MovieResult]
    hybrid: list[MovieResult]


@app.get("/", response_class=FileResponse)
async def serve_index():
    """Serve the single-page UI from static/index.html."""
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/benchmark")
async def get_benchmark():
    """
    Return pre-computed benchmark results from data/processed/benchmark_results.json.

    Raises:
        404: If the benchmark file has not been generated yet.
    """
    if not BENCHMARK_JSON.exists():
        raise HTTPException(status_code=404, detail="Benchmark results not found. Run benchmark.py first.")
    with open(BENCHMARK_JSON, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return JSONResponse(content=data)


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    Run both retrievers concurrently and return side-by-side results.

    Both content_retrieve and hybrid_retrieve are awaited in a single
    asyncio.gather call so they execute in parallel.

    Args:
        request: JSON body with a "query" string field.

    Returns:
        JSON with "content_based" and "hybrid" lists, each containing
        up to 5 results with title, genre, and reason fields.

    Raises:
        400: If the query is empty.
        500: If either retriever fails unexpectedly.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        content_results, hybrid_results = await asyncio.gather(
            content_retrieve(query),
            hybrid_retrieve(query),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Index not found: {exc}")
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}")

    return RecommendResponse(
        content_based=content_results,
        hybrid=hybrid_results,
    )
