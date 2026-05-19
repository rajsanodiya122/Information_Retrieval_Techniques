#!/usr/bin/env python3
"""
Personalization demo: same query, different user profiles
"""

import json
import math
from collections import defaultdict
import re

# Copy necessary functions from query_interface.py (import them or define here)
# For brevity, we'll reuse code from query_interface by importing it as a module.
# But if not possible, copy the functions. We'll assume we can import.

import query_interface as qi

def load_data():
    return qi.load_data()

def build_user_profile(query_history):
    """Create a dict of term->frequency from a list of past queries"""
    profile = defaultdict(float)
    for query in query_history:
        tokens = qi.preprocess(query)
        for t in tokens:
            profile[t] += 1.0
    # Normalise
    total = sum(profile.values())
    if total > 0:
        for t in profile:
            profile[t] /= total
    return profile

def personal_search(query, inv_index, doc_vecs, pr_scores, doc_ids, user_profile, alpha=0.6, boost=0.2):
    tokens = qi.preprocess(query)
    if not tokens:
        return []
    qvec = qi.query_tfidf(tokens, inv_index, len(doc_ids))
    scores = []
    for did in doc_ids:
        sim = qi.cosine_similarity(qvec, doc_vecs[did])
        pr = pr_scores.get(did, 0.0)
        combined = alpha * sim + (1 - alpha) * pr
        # Personalization boost: dot product of doc vector with user profile
        profile_boost = 0.0
        for term, weight in user_profile.items():
            if term in doc_vecs[did]:
                profile_boost += doc_vecs[did][term] * weight
        combined *= (1.0 + boost * profile_boost)
        scores.append((did, sim, pr, combined, profile_boost))
    scores.sort(key=lambda x: x[3], reverse=True)
    return scores[:5]

if __name__ == "__main__":
    pages, inv_index, pr_scores, _ = load_data()
    doc_ids = [p['doc_id'] for p in pages]
    doc_vecs = qi.build_doc_vectors(inv_index, doc_ids)
    id2title = {p['doc_id']: p['title'] for p in pages}
    
    # Simulate two user profiles
    userA_history = ["tfidf", "term frequency", "document ranking", "vector space model"]
    userB_history = ["pagerank", "link analysis", "hits", "web graph", "authority"]
    
    profileA = build_user_profile(userA_history)
    profileB = build_user_profile(userB_history)
    
    query = "information retrieval"   # neutral query
    
    print("="*60)
    print("PERSONALIZATION DEMO")
    print(f"Query: '{query}'\n")
    
    # No personalization (baseline)
    print("--- Baseline (no personalization) ---")
    base_results = qi.search(query, inv_index, doc_vecs, pr_scores, doc_ids)
    for rank, (did, sim, pr, comb) in enumerate(base_results, 1):
        print(f"{rank}. {did} - {id2title[did][:50]} (sim={sim:.3f}, PR={pr:.6f}, comb={comb:.6f})")
    
    # User A personalisation
    print("\n--- User A (TF-IDF focused) ---")
    resultsA = personal_search(query, inv_index, doc_vecs, pr_scores, doc_ids, profileA)
    for rank, (did, sim, pr, comb, pboost) in enumerate(resultsA, 1):
        print(f"{rank}. {did} - {id2title[did][:50]} (sim={sim:.3f}, PR={pr:.6f}, comb={comb:.6f}, boost={pboost:.3f})")
    
    # User B personalisation
    print("\n--- User B (PageRank/link analysis focused) ---")
    resultsB = personal_search(query, inv_index, doc_vecs, pr_scores, doc_ids, profileB)
    for rank, (did, sim, pr, comb, pboost) in enumerate(resultsB, 1):
        print(f"{rank}. {did} - {id2title[did][:50]} (sim={sim:.3f}, PR={pr:.6f}, comb={comb:.6f}, boost={pboost:.3f})")
