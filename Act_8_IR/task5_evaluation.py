#!/usr/bin/env python3
"""
Evaluation: Precision@5 and Recall@5 for different ranking methods
"""

import json
import query_interface as qi  # or copy functions
from collections import defaultdict

# Load data
pages, inv_index, pr_scores, hits = qi.load_data()
doc_ids = [p['doc_id'] for p in pages]
doc_vecs = qi.build_doc_vectors(inv_index, doc_ids)

# Manually define relevance (based on our mock data content)
relevance = {
    "tfidf": ["doc1", "doc5", "doc11"],  # doc IDs that actually discuss TF-IDF
    "pagerank": ["doc4", "doc8", "doc13"],
    "information retrieval": ["doc0", "doc3", "doc6", "doc9", "doc12"],
    "inverted index": ["doc1", "doc4", "doc7", "doc11"],
    "web crawling": ["doc3", "doc7", "doc14"]
}
# Ensure these doc_ids exist in our mock data; adjust if necessary.

def eval_query(query, ranking_func, relevant_set, k=5):
    results = ranking_func(query, inv_index, doc_vecs, pr_scores, doc_ids)[:k]
    retrieved = [r[0] for r in results]
    relevant_retrieved = set(retrieved) & relevant_set
    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_set) if relevant_set else 0
    return precision, recall, retrieved

def method_baseline(query, inv_idx, doc_vecs, pr, docs):
    return qi.search(query, inv_idx, doc_vecs, pr, docs)  # TF-IDF + PR

def method_tfidf_only(query, inv_idx, doc_vecs, pr, docs):
    # TF-IDF only (alpha=1.0)
    return qi.search(query, inv_idx, doc_vecs, pr, docs, alpha=1.0)

# We'll add a personalized version later, but for simplicity we'll skip in evaluation.

print("="*60)
print("EVALUATION: Precision@5 and Recall@5")
print("="*60)

for query, rel_set in relevance.items():
    rel_set = set(rel_set)
    # TF-IDF + PageRank
    p1, r1, _ = eval_query(query, method_baseline, rel_set)
    # TF-IDF only
    p2, r2, _ = eval_query(query, method_tfidf_only, rel_set)
    print(f"Query: '{query}'")
    print(f"  TF-IDF+PageRank -> Precision@5: {p1:.2f}, Recall@5: {r1:.2f}")
    print(f"  TF-IDF only      -> Precision@5: {p2:.2f}, Recall@5: {r2:.2f}")
    print()
