#!/usr/bin/env python3
"""
HITS Algorithm for Activity 8
"""

import json
import networkx as nx
import numpy as np

def build_graph(pages_file):
    with open(pages_file, 'r') as f:
        pages = json.load(f)
    G = nx.DiGraph()
    url_to_id = {p['url']: p['doc_id'] for p in pages}
    for p in pages:
        G.add_node(p['doc_id'], url=p['url'], title=p['title'])
    for p in pages:
        src = p['doc_id']
        for link in p['links']:
            if link in url_to_id:
                tgt = url_to_id[link]
                if src != tgt:
                    G.add_edge(src, tgt)
    return G, pages

def hits(G, max_iter=100, tol=1e-4):
    nodes = list(G.nodes())
    n = len(nodes)
    # Initialise
    auth = {node: 1.0 for node in nodes}
    hub = {node: 1.0 for node in nodes}
    
    for it in range(max_iter):
        # Update authority scores: sum of hub scores of predecessors
        new_auth = {}
        for node in nodes:
            new_auth[node] = sum(hub[p] for p in G.predecessors(node))
        # Update hub scores: sum of authority scores of successors
        new_hub = {}
        for node in nodes:
            new_hub[node] = sum(new_auth[s] for s in G.successors(node))
        
        # Normalise
        max_auth = max(new_auth.values()) if max(new_auth.values()) > 0 else 1
        max_hub = max(new_hub.values()) if max(new_hub.values()) > 0 else 1
        for node in nodes:
            new_auth[node] /= max_auth
            new_hub[node] /= max_hub
        
        # Delta
        delta = max(
            max(abs(new_auth[n]-auth[n]) for n in nodes),
            max(abs(new_hub[n]-hub[n]) for n in nodes)
        )
        auth, hub = new_auth, new_hub
        print(f"HITS Iteration {it+1}: delta={delta:.6f}")
        if delta < tol:
            print(f"Converged after {it+1} iterations")
            break
    else:
        print(f"Reached max iterations {max_iter}")
    return hub, auth

def top_k(scores, k=5):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

if __name__ == "__main__":
    G, pages = build_graph("data/crawled_pages.json")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    print("\n" + "="*60)
    print("HITS ALGORITHM")
    print("="*60)
    hub_scores, auth_scores = hits(G)
    
    print("\nTop-5 Hub Pages:")
    for rank, (doc_id, score) in enumerate(top_k(hub_scores, 5), 1):
        title = G.nodes[doc_id].get('title', '')[:60]
        print(f"{rank}. {doc_id} (score={score:.4f}): {title}")
    
    print("\nTop-5 Authority Pages:")
    for rank, (doc_id, score) in enumerate(top_k(auth_scores, 5), 1):
        title = G.nodes[doc_id].get('title', '')[:60]
        print(f"{rank}. {doc_id} (score={score:.4f}): {title}")
    
    # Save scores
    with open("data/hits_scores.json", "w") as f:
        json.dump({
            "hub": {k: float(v) for k,v in hub_scores.items()},
            "auth": {k: float(v) for k,v in auth_scores.items()}
        }, f, indent=2)
    print("\n✓ HITS scores saved to data/hits_scores.json")

    # Load PageRank for comparison
    try:
        with open("data/pagerank_scores.json") as f:
            pr_data = json.load(f)
        pr_scores = pr_data["pagerank_scores"]
        print("\nComparison: PageRank vs HITS (Top-5 Authority)")
        pr_top = top_k(pr_scores, 5)
        print(f"{'Rank':<6} {'PageRank (doc_id)':<20} {'HITS Authority (doc_id)':<25}")
        for i in range(5):
            pr_id = pr_top[i][0] if i < len(pr_top) else "N/A"
            hits_id = top_k(auth_scores, 5)[i][0] if i < len(top_k(auth_scores,5)) else "N/A"
            print(f"{i+1:<6} {pr_id:<20} {hits_id:<25}")
    except FileNotFoundError:
        print("No pagerank_scores.json found for comparison.")
