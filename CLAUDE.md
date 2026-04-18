# Movie Recommendation System — Co-op Assessment

Content-based vs Hybrid movie recommender using Netflix data. One query runs through
both methods simultaneously. Results displayed side by side in a single HTML page so
the difference in recommendation quality is immediately visible.

## Stack
- Python 3.11+
- FastAPI — backend API and static file serving
- sentence-transformers (all-MiniLM-L6-v2) — semantic embeddings
- rank_bm25 — keyword matching for content-based method
- FAISS — vector similarity search for hybrid method
- Groq API — model :(qwen/qwen3-32b) for query intent extraction (hybrid method only)
- Pandas — CSV ingestion and data handling
- Docker — containerisation

## Data
- Source file: `data/netflix_data.csv`
- Expected columns: title, description, genre, cast, listed_in, type
- On startup, `ingest.py` builds FAISS index and BM25 index from the description column
- Processed indexes saved to `data/processed/` (git-ignored)

## Data Contract (read once, reuse always)
- Columns: title, director, cast, listed_in, description, release_year, rating, duration, type
- Shape: ~8800 rows
- Key fields for indexing: description (primary), title + listed_in + cast (BM25 metadata)
- Do NOT re-read netflix_data.csv in any subsequent step
- data/processed/ contains: faiss.index, bm25.pkl, movies.pkl — load these instead

## Project Structure
```
movie-recommender/
├── CLAUDE.md
├── Dockerfile
├── README.md
├── requirements.txt
├── .env                      # ANTHROPIC_API_KEY (never commit)
├── .gitignore
├── data/
│   ├── netflix_data.csv
│   └── processed/            # auto-generated, git-ignored
├── src/
│   ├── ingest.py             # load CSV, build FAISS + BM25 indexes
│   ├── content_retriever.py  # pure BM25 keyword search on metadata + description
│   ├── hybrid_retriever.py   # BM25 + FAISS semantic search fused via RRF
│   ├── recommender.py        # Claude API for query intent extraction (hybrid only)
│   └── main.py               # FastAPI: serves UI + /recommend endpoint
└── static/
    └── index.html            # single HTML file with side-by-side comparison UI
```

## Commands
- `pip install -r requirements.txt` — install dependencies
- `python src/ingest.py` — build indexes from netflix_data.csv (run once)
- `uvicorn src.main:app --reload --port 8080` — local dev server
- `docker build -t movie-recommender .` — build image
- `docker run -p 8080:80 movie-recommender` — run at localhost:8080

## API
- `POST /recommend` — single endpoint, runs BOTH methods in parallel
  - Request:  `{"query": "thrilling crime drama with unexpected ending"}`
  - Response:
    ```json
    {
      "content_based": [
        {"title": "...", "genre": "...", "reason": "matched keywords: crime, thriller"}
      ],
      "hybrid": [
        {"title": "...", "genre": "...", "reason": "semantically similar mood and theme"}
      ]
    }
    ```
- `GET /` — serves static/index.html

## Method Definitions

### Content-Based (left column)
1. Tokenise query into keywords (no Claude, pure text processing)
2. Run BM25 over combined field: title + description + listed_in + cast
3. Return top 5 by BM25 score
4. Reason shown: which keywords matched

### Hybrid (right column)
1. Use Claude to extract structured intent from query (mood, genre, themes, tone)
2. Run BM25 on extracted keywords → ranked list A
3. Embed query with sentence-transformers → FAISS cosine search → ranked list B
4. Fuse A and B using Reciprocal Rank Fusion (RRF)
5. Return top 5 fused results
6. Reason shown: semantic similarity score + matched themes from Claude extraction

## UI Design
- Single HTML file: `static/index.html` — inline CSS and JS, no CDN dependencies
- Three core elements (assessment requirement): text input, submit button, results area
- Results area contains two columns — treated as one results section:
  - Left column: "Content-Based" results (BM25 only)
  - Right column: "Hybrid" results (BM25 + Semantic + Claude)
- Each result card shows: title, genre/type, one-line reason
- Loading state shown while both methods run
- Both methods called in a single POST /recommend request (parallel execution in backend)
- Color distinction between columns so comparison is visually clear at a glance

## Docker Rules (strict — from assessment)
- Must work with exactly:
    docker build -t movie-recommender .
    docker run -p 8080:80 movie-recommender
- FastAPI runs on port 80 inside the container
- Indexes must be pre-built into the image (run ingest.py during Docker build)
- No manual steps between the two commands

## README Requirements (evaluated — do not skip)
- Mermaid architecture diagram showing full data flow for BOTH methods
- AI provider and model: Anthropic, claude-sonnet-4-6
- Env var the reviewer must set: ANTHROPIC_API_KEY
- Explanation of both methods and why hybrid produces better results
- Side-by-side screenshot showing a sample query with both columns populated

## Code Rules
- All functions must have docstrings
- content_retriever.py and hybrid_retriever.py must be fully independent modules
- Both retrievers called concurrently in main.py using asyncio.gather()
- Basic error handling: missing CSV, empty query, API failure, index not found
- Never commit .env or data/processed/
- Use async/await in all FastAPI route handlers
- index.html must be self-contained (inline CSS + JS, zero external dependencies)