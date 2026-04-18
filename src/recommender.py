"""
recommender.py — Groq API wrapper for query intent extraction.

Calls qwen/qwen3-32b via Groq to parse a free-form user query into structured
search intent used by the hybrid retriever.

Public API:
    extract_intent(query: str) -> IntentResult
"""

import asyncio
import json
import os
import pathlib
from typing import TypedDict

from dotenv import load_dotenv
from groq import Groq

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")


class IntentResult(TypedDict):
    """Structured intent extracted from a user query."""
    mood: str           # Overall tone/mood (e.g. "dark and suspenseful")
    genre: str          # Primary genre (e.g. "crime drama")
    themes: list[str]   # 2-4 thematic labels (e.g. ["betrayal", "redemption"])
    keywords: list[str] # Expanded keyword tokens for BM25 search


_SYSTEM_PROMPT = """\
You are a movie search assistant. Given a user query, extract structured search intent.

Return ONLY a JSON object with exactly these keys:
  "mood":     string  — overall tone or emotional feel (e.g. "dark and tense")
  "genre":    string  — primary genre (e.g. "crime thriller")
  "themes":   array   — 2 to 4 short thematic labels (e.g. ["betrayal", "heist", "loyalty"])
  "keywords": array   — expanded keyword tokens useful for text search, including synonyms,
                        genre terms, mood words, and notable elements from the query

No markdown, no explanation, no code fences. Pure JSON only."""


def _get_client() -> Groq:
    """Instantiate a Groq client, raising clearly if the key is missing."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set.")
    return Groq(api_key=api_key)


def _parse_response(raw: str) -> IntentResult:
    """
    Parse the model's raw text into an IntentResult.

    qwen3-32b wraps its reasoning in <think>...</think> tags before the
    actual JSON output — strip that block first.
    Also strips accidental markdown code fences before JSON parsing.
    """
    text = raw.strip()

    # Strip <think>...</think> chain-of-thought block (qwen3 reasoning mode)
    if "<think>" in text:
        end = text.find("</think>")
        text = text[end + len("</think>"):].strip() if end != -1 else text

    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]

    data = json.loads(text)
    return IntentResult(
        mood=str(data.get("mood", "")),
        genre=str(data.get("genre", "")),
        themes=[str(t) for t in data.get("themes", [])],
        keywords=[str(k) for k in data.get("keywords", [])],
    )


async def extract_intent(query: str) -> IntentResult:
    """
    Call qwen/qwen3-32b via Groq to extract structured intent from *query*.

    Args:
        query: Free-form user search string.

    Returns:
        IntentResult with mood, genre, themes, and keywords.
        On API or parse failure, returns a minimal fallback derived from
        the raw query so the hybrid pipeline can still proceed.

    Raises:
        EnvironmentError: If GROQ_API_KEY is not set.
        ValueError: If query is empty.
    """
    query = query.strip()
    if not query:
        raise ValueError("Query must not be empty.")

    client = _get_client()

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="qwen/qwen3-32b",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f'Query: "{query}"'},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        return _parse_response(raw)

    except (json.JSONDecodeError, KeyError, IndexError):
        # Parse failure — fall back to raw query tokens so pipeline continues
        tokens = query.lower().split()
        return IntentResult(mood="", genre="", themes=[], keywords=tokens)
