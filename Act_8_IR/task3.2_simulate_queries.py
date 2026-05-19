#!/usr/bin/env python3
"""Simulate 20 queries to fill query_log.csv"""

import csv
import datetime
import random
import os

LOG_FILE = "data/query_log.csv"
USER_ID = "user1"

queries = [
    "information retrieval",
    "tfidf",
    "pagerank",
    "inverted index",
    "vector space model",
    "precision recall",
    "web crawling",
    "tfidf",               # repeated
    "search engine",
    "information retrieval",
    "document ranking",
    "boolean retrieval",
    "relevance feedback",
    "query expansion",
    "indexing",
    "pagerank",            # repeated
    "information retrieval",
    "latent semantic indexing",
    "bm25",
    "hits algorithm",
    "user profiling"
]

# Create log with header if needed
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "user_id", "query_text", "top_result_id"])

for q in queries:
    timestamp = datetime.datetime.now().isoformat()
    # Simulate a "clicked" doc id randomly (doc0..doc29)
    clicked = f"doc{random.randint(0, 29)}"
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, USER_ID, q, clicked])

print(f"Logged {len(queries)} queries to {LOG_FILE}")
