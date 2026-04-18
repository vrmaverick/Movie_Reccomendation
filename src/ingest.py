"""
ingest.py — Load netflix_data.csv once, build FAISS + BM25 indexes,
save artifacts to data/processed/ for use by the retrievers.

Artifacts written:
  data/processed/faiss.index  — FAISS IndexFlatIP over sentence-transformer embeddings
  data/processed/bm25.pkl     — BM25Okapi index over combined text fields
  data/processed/movies.pkl   — list of dicts (one per row) with all metadata
"""

import os
import pickle
import pathlib

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CSV_PATH = DATA_DIR / "netflix_data.csv"


def load_data(csv_path: pathlib.Path) -> pd.DataFrame:
    """Load netflix_data.csv and drop rows with no description."""
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.dropna(subset=["description"]).reset_index(drop=True)
    print(f"Loaded {before} rows — kept {len(df)} with non-null description.")
    return df


def build_bm25(df: pd.DataFrame) -> BM25Okapi:
    """
    Build a BM25Okapi index over a concatenation of:
      title + description + listed_in + cast
    Each document is tokenised by whitespace after lower-casing.
    """
    def combine(row: pd.Series) -> str:
        parts = [
            str(row.get("title", "") or ""),
            str(row.get("description", "") or ""),
            str(row.get("listed_in", "") or ""),
            str(row.get("cast", "") or ""),
        ]
        return " ".join(parts)

    corpus = [combine(row).lower().split() for _, row in df.iterrows()]
    index = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents.")
    return index


def build_faiss(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> faiss.IndexFlatIP:
    """
    Embed the description column with sentence-transformers and build a
    FAISS IndexFlatIP (inner-product / cosine after L2-normalisation).

    Returns the FAISS index (embeddings are L2-normalised in-place before adding).
    """
    model = SentenceTransformer(model_name)
    descriptions = df["description"].tolist()
    print(f"Encoding {len(descriptions)} descriptions with '{model_name}'...")
    embeddings = model.encode(descriptions, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}.")
    return index


def build_movies_list(df: pd.DataFrame) -> list[dict]:
    """
    Convert the DataFrame to a list of plain dicts (one per row).
    Only the columns needed at query-time are kept.
    """
    keep = ["title", "type", "listed_in", "description", "cast",
            "director", "release_year", "rating", "duration"]
    available = [c for c in keep if c in df.columns]
    return df[available].to_dict(orient="records")


def save_artifacts(
    faiss_index: faiss.IndexFlatIP,
    bm25_index: BM25Okapi,
    movies: list[dict],
    out_dir: pathlib.Path,
) -> None:
    """Persist all three artifacts to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(out_dir / "faiss.index"))
    print(f"Saved FAISS index: {out_dir / 'faiss.index'}")

    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    print(f"Saved BM25 index:  {out_dir / 'bm25.pkl'}")

    with open(out_dir / "movies.pkl", "wb") as f:
        pickle.dump(movies, f)
    print(f"Saved movies list: {out_dir / 'movies.pkl'}")


def main() -> None:
    """Entry point: load CSV, build indexes, save to data/processed/."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}")

    df = load_data(CSV_PATH)
    bm25_index = build_bm25(df)
    faiss_index = build_faiss(df)
    movies = build_movies_list(df)
    save_artifacts(faiss_index, bm25_index, movies, PROCESSED_DIR)
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
