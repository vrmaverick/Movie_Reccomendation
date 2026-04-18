# Code Review to Personally Track what's Done

```
src > content_retriever.py — Pure BM25 keyword-based retrieval.
    
    indexes are retirved like : 
    bm25_path = PROCESSED_DIR / "bm25.pkl"
    movies_path = PROCESSED_DIR / "movies.pkl"

    A function to tokenize query : 
    Lowercasing
    Removing Punctuation text.translate(...)
    Splitting and Filtering


    retrieve: 
    first preprocess query
    get indexes intialized 
    Tokenize queries
    Get scores based on tokens
    rank them top k
    returned combined results

```

```
src > hybrid_retriever.py — BM25 + FAISS semantic search fused via Reciprocal Rank Fusion.

    
    indexes are retirved like : 
    bm25_path = PROCESSED_DIR / "bm25.pkl"
    movies_path = PROCESSED_DIR / "movies.pkl"
    faiss_path = PROCESSED_DIR / "faiss.index"


    A function to tokenize query : 
    Lowercasing
    Removing Punctuation text.translate(...)
    Splitting and Filtering


Step 0 : - Get Intent using groq API
step 1 : -    Run BM25 with *keywords* and return the top-n document indexes ordered by descending score.
step 2 : -    Faiss : indexing as explained below in appendix
step 3 : -    Reciprocal Rank Fusion (RRF) is a clever "democracy" algorithm for search results. It’s used to combine rankings from different sources (like a keyword search and a vector search) into a single, unified list without needing to worry about the actual scores (like "0.85 cosine similarity" vs "12.5 BM25 score").




    retrieve: 
    first preprocess query
    get indexes intialized
    Get Candidate pool — larger than top_k so fusion has signal to rerank from 
    Tokenize queries
    run pipeline : Step 0 and Step 2 (parallel as they are independent)
    Using Intent execute step 1
    Step 4 to fuse Ranking 

    Reasoning using FAISS(Cosine Check), Theme and Mood check (Intent) => to Draft a reason 
    rank them top k
    returned combined results

```

```
src > ingest.py — Load netflix_data.csv once, build FAISS + BM25 indexes, save artifacts to data/processed/ for use by the retrievers.

Load dataset 
Fit BM25API on concatinated information
Build FAISS index ---- Embedding - > Fit into normalized + flattend indexes


Build movies : 

This function is the "Data Trimmer." Its job is to take a massive, heavy DataFrame (which might have dozens of columns you don't need) and turn it into a lightweight list of Python dictionaries that are easy to access during a search.

Think of it as packing a "go-bag" for your application—you're only taking the essentials so the search engine can run faster.

```

```
src > Reccomend.py

    Intent_classification : Call qwen/qwen3-32b via Groq to extract structured intent from *query*.
    Args:
        query: Free-form user search string.
    Returns:
        IntentResult with mood, genre, themes, and keywords.
        On API or parse failure, returns a minimal fallback derived from
        the raw query so the hybrid pipeline can still proceed.


    Final Groq Client intialization

    Final Output Parsing

```



# Appendix

# FAISS

1. The Setup (Indexing)Suppose we have three documents with the following 3D vectors:
Doc A: $[0.1, 0.9, 0.0]$ (Points mostly "up" the Y-axis)
Doc B: $[0.8, 0.1, 0.1]$ (Points mostly "right" along the X-axis)
Doc C: $[0.1, 0.1, 0.9]$ (Points mostly "out" toward the Z-axis)In IndexFlatIP, FAISS simply stores these as a matrix. It doesn't do anything fancy yet; it just keeps a clean record of where these "arrows" are pointing.

2. The QueryNow, you provide a query vector:
query = [0.2, 0.8, 0.1].
This vector is very similar to Doc A because they both have high values in the second dimension (the Y-axis).3. The Search ProcessWhen you call index.
search(query, k=1), FAISS performs a Brute Force calculation (because IndexFlat means "no compression/no clusters").Normalization: As we discussed, FAISS scales the query vector so its length is exactly 1.0.

Dot Product Calculation: It multiplies the query against every vector in the index:
Query $\cdot$ 
Doc A: $(0.2 \times 0.1) + (0.8 \times 0.9) + (0.1 \times 0.0) = \mathbf{0.74}$Query $\cdot$ 
Doc B: $(0.2 \times 0.8) + (0.8 \times 0.1) + (0.1 \times 0.1) = 0.25$Query $\cdot$ 
Doc C: $(0.2 \times 0.1) + (0.8 \times 0.1) + (0.1 \times 0.9) = 0.19$4. 
The ResultFAISS looks at the scores $(0.74, 0.25, 0.19)$ and identifies that 0.74 is the highest.
Internal Return: It finds the position of $0.74$ in the list (index 0).Function Return: Your _faiss_rank function returns ([0], [0.74]).


# RRF : 
Reciprocal Rank Fusion (RRF) is a clever "democracy" algorithm for search results. It’s used to combine rankings from different sources (like a keyword search and a vector search) into a single, unified list without needing to worry about the actual scores (like "0.85 cosine similarity" vs "12.5 BM25 score").

Here is the breakdown of why and how this code works:
1. The Core FormulaThe mathematical logic being applied in your loops is:
$$RRFscore(d \in D) = \sum_{r \in R} \frac{1}{k + r(d)}$$$r(d)$:
The rank of document $d$ in a list (1st place, 2nd place, etc.).
$k$: A constant (usually 60). It prevents documents ranked very high (like rank 1) from completely overwhelming the total score.

2. Why use this instead of just averaging scores?In search systems, you often have a "Scale Mismatch" problem:BM25 (Keyword search) might give scores from 0 to 100+.FAISS (Vector search) might give scores from 0.7 to 1.0.You can't just add $100 + 0.7$—the keyword search would always win. RRF ignores the raw scores entirely. It only cares that a document was "1st" in one list and "5th" in another. This makes it a "rank-based" ensemble method rather than a "score-based" one.


# Why There is a Reasoning Aspect in Hybrid : 
In modern AI applications, "Black Box" recommendations (where you get a result but don't know why) are often less trusted by users. This code provides Transparency.

By showing the Semantic Similarity (how well the meaning matched) alongside Themes (hard metadata matches), you are combining "fuzzy" AI logic with "exact" keyword logic in a way the user can understand.

