#!/usr/bin/env python3
"""
Inverted Index Builder with TF-IDF for IR Domain
"""

import json
import re
import os
import numpy as np

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Simple text preprocessing without NLTK dependency
STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now'
}

def preprocess(text):
    """Clean and tokenize text"""
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and short tokens
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def compute_tfidf(documents):
    """Compute TF-IDF weights manually"""
    print("Computing TF-IDF weights manually...")
    
    # Get vocabulary
    vocab = set()
    for doc in documents:
        vocab.update(doc)
    
    vocab = sorted(list(vocab))
    print(f"Vocabulary size: {len(vocab)} terms")
    
    # Compute document frequency
    N = len(documents)
    df = {}
    for term in vocab:
        df[term] = sum(1 for doc in documents if term in doc)
    
    # Compute TF-IDF
    tfidf_matrix = []
    for doc in documents:
        doc_vector = {}
        # Term frequency
        tf = {}
        for term in doc:
            tf[term] = tf.get(term, 0) + 1
        
        # TF-IDF
        for term in set(doc):
            if df[term] > 0:
                # Sublinear TF scaling
                tf_scaled = 1 + np.log(tf[term]) if tf[term] > 0 else 0
                # IDF
                idf = np.log((N + 1) / (df[term] + 1)) + 1
                tfidf = tf_scaled * idf
                doc_vector[term] = tfidf
        
        tfidf_matrix.append(doc_vector)
    
    return tfidf_matrix, vocab, df

def build_index(input_file, output_file):
    """Build inverted index with TF-IDF weights"""
    print("=" * 60)
    print("BUILDING INVERTED INDEX WITH TF-IDF")
    print("=" * 60)
    
    # Load crawled data
    with open(input_file, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    print(f"Loaded {len(pages)} documents")
    
    # Preprocess documents
    print("Preprocessing documents...")
    doc_ids = [page['doc_id'] for page in pages]
    processed_docs = []
    for page in pages:
        tokens = preprocess(page['text'])
        processed_docs.append(tokens)
    
    # Compute TF-IDF
    tfidf_matrix, vocab, df = compute_tfidf(processed_docs)
    
    # Build inverted index
    print("Building inverted index...")
    inverted_index = {}
    
    for term in vocab:
        postings = {}
        for i, doc_vector in enumerate(tfidf_matrix):
            if term in doc_vector:
                postings[doc_ids[i]] = doc_vector[term]
        
        if postings:
            inverted_index[term] = postings
    
    print(f"Inverted index built with {len(inverted_index)} terms")
    
    # Save inverted index
    print(f"Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inverted_index, f, indent=2)
    
    return inverted_index, pages

def display_sample_terms(inverted_index):
    """Display sample index entries"""
    # IR-specific terms
    ir_terms = ['tfidf', 'pagerank', 'retrieval', 'index', 'query', 
                 'document', 'relevance', 'precision', 'recall', 'search']
    
    print("\n" + "=" * 60)
    print("SAMPLE INVERTED INDEX ENTRIES (IR TERMS)")
    print("=" * 60)
    
    for term in ir_terms:
        if term in inverted_index:
            entries = inverted_index[term]
            # Show top 5 documents by TF-IDF score
            top_entries = dict(sorted(entries.items(), key=lambda x: x[1], reverse=True)[:5])
            print(f"\nTerm: '{term}'")
            print(f"  Appears in {len(entries)} documents")
            print(f"  Top scores: ", end="")
            for doc_id, score in list(top_entries.items())[:3]:
                print(f"{doc_id}:{score:.3f} ", end="")
            print()
        else:
            print(f"\nTerm: '{term}' - NOT IN INDEX")

if __name__ == "__main__":
    try:
        # Build index
        inverted_index, pages = build_index(
            "data/crawled_pages.json",
            "data/inverted_index.json"
        )
        
        # Display sample terms
        display_sample_terms(inverted_index)
        
        # Statistics
        print("\n" + "=" * 60)
        print("INDEX STATISTICS")
        print("=" * 60)
        print(f"Total unique terms: {len(inverted_index)}")
        
        postings_lengths = [len(v) for v in inverted_index.values()]
        print(f"Average postings per term: {np.mean(postings_lengths):.2f}")
        print(f"Maximum postings: {max(postings_lengths)}")
        print(f"Minimum postings: {min(postings_lengths)}")
        
        # Show most common terms
        sorted_terms = sorted(inverted_index.items(), 
                            key=lambda x: len(x[1]), 
                            reverse=True)[:10]
        
        print("\nTop 10 most common terms:")
        for term, postings in sorted_terms:
            print(f"  '{term}': appears in {len(postings)} documents")
    
    except FileNotFoundError:
        print("Error: crawled_pages.json not found. Run crawler.py first.")
    except Exception as e:
        print(f"Error: {e}")
