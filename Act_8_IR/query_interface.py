#!/usr/bin/env python3
"""
Query Interface (CLI) with TF-IDF + PageRank ranking + Query Logging
Activity 8 – Mini Search Engine (Part 2)
"""

import json
import os
import re
import csv
import math
import random
import datetime
from collections import defaultdict

# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------
CRAWLED_FILE   = "data/crawled_pages.json"
INDEX_FILE     = "data/inverted_index.json"
PR_FILE        = "data/pagerank_scores.json"
HITS_FILE      = "data/hits_scores.json"
LOG_FILE       = "data/query_log.csv"

# Default user ID for logging
USER_ID = "user1"

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data():
    """Load crawled pages, inverted index and PageRank scores."""
    with open(CRAWLED_FILE) as f:
        pages = json.load(f)
    with open(INDEX_FILE) as f:
        inv_index = json.load(f)
    with open(PR_FILE) as f:
        pr_data = json.load(f)
    pr_scores = pr_data["pagerank_scores"]
    # Optional HITS (if present)
    hits = None
    if os.path.exists(HITS_FILE):
        with open(HITS_FILE) as f:
            hits = json.load(f)
    return pages, inv_index, pr_scores, hits

# -------------------------------------------------------------------
# Text preprocessing (same as indexer)
# -------------------------------------------------------------------
STOPWORDS = {
    'a','an','and','are','as','at','be','by','for','from','has','he',
    'in','is','it','its','of','on','that','the','to','was','were','will','with'
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

# -------------------------------------------------------------------
# Reconstruct document vectors from inverted index
# -------------------------------------------------------------------
def build_doc_vectors(inv_index, doc_ids):
    """Re‑create sparse document vectors from the inverted index."""
    doc_vecs = {did: defaultdict(float) for did in doc_ids}
    for term, postings in inv_index.items():
        for did, score in postings.items():
            doc_vecs[did][term] = score
    return doc_vecs

# -------------------------------------------------------------------
# Query TF‑IDF vector (simple)
# -------------------------------------------------------------------
def query_tfidf(query_terms, inv_index, num_docs):
    """Build a query vector using IDF from the inverted index."""
    vec = defaultdict(float)
    for term in query_terms:
        if term in inv_index:
            df = len(inv_index[term])
            idf = math.log((num_docs + 1) / (df + 1)) + 1
            tf = 1.0            # presence only in a short query
            vec[term] = tf * idf
    return vec

# -------------------------------------------------------------------
# Cosine similarity
# -------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[t] * vec2[t] for t in common)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# -------------------------------------------------------------------
# Core search function
# -------------------------------------------------------------------
def search(query, inv_index, doc_vecs, pr_scores, doc_ids, alpha=0.6):
    """
    Return top‑5 results as list of tuples:
    (doc_id, cosine_sim, pagerank_score, combined_score)
    """
    tokens = preprocess(query)
    if not tokens:
        return []
    qvec = query_tfidf(tokens, inv_index, len(doc_ids))
    results = []
    for did in doc_ids:
        sim = cosine_similarity(qvec, doc_vecs[did])
        pr = pr_scores.get(did, 0.0)
        combined = alpha * sim + (1 - alpha) * pr
        results.append((did, sim, pr, combined))
    results.sort(key=lambda x: x[3], reverse=True)
    return results[:5]

# -------------------------------------------------------------------
# Query log initialisation
# -------------------------------------------------------------------
def init_log():
    """Create log file with header if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "user_id", "query_text", "top_result_id"])

# -------------------------------------------------------------------
# Main interactive CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Load all data once
    pages, inv_index, pr_scores, hits = load_data()
    doc_ids = [p['doc_id'] for p in pages]
    doc_vecs = build_doc_vectors(inv_index, doc_ids)
    id2title = {p['doc_id']: p['title'] for p in pages}

    # Initialize query log
    init_log()

    print("=" * 60)
    print("MINI SEARCH ENGINE – Information Retrieval Domain")
    print("Type 'quit' to exit, 'log' to see last logged queries.\n")

    while True:
        q = input("Enter query: ").strip()
        if q.lower() == 'quit':
            break
        if q.lower() == 'log':
            # Show last 5 logged entries
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    print("\nLast 5 logged queries:")
                    for row in rows[-5:]:
                        print("  ", row)
            else:
                print("No log file yet.")
            continue
        if not q:
            continue

        # Perform search
        results = search(q, inv_index, doc_vecs, pr_scores, doc_ids)
        if not results:
            print("No results found.\n")
            continue

        print(f"\nTop-5 results for '{q}':")
        print(f"{'Rank':<5} {'Doc ID':<6} {'Title':<50} {'Sim':>6} {'PR':>8} {'Combined':>8}")
        print("-" * 85)
        for rank, (did, sim, pr, comb) in enumerate(results, 1):
            title = id2title.get(did, '')[:50]
            print(f"{rank:<5} {did:<6} {title:<50} {sim:6.3f} {pr:8.6f} {comb:8.6f}")

        # ---- Simulate a click on one of the top results (weighted towards top) ----
        # Weights: rank 1 = 5, rank 2 = 4, ... rank 5 = 1
        weights = [5, 4, 3, 2, 1]
        clicked_id = random.choices([r[0] for r in results], weights=weights, k=1)[0]

        # ---- Log the query ----
        timestamp = datetime.datetime.now().isoformat()
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, USER_ID, q, clicked_id])

        print(f"\n(Logged: top clicked = {clicked_id})\n")
