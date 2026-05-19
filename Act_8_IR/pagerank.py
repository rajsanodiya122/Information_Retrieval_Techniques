#!/usr/bin/env python3
"""
PageRank Implementation for IR Domain - Fixed Version
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

def build_link_graph(pages_file):
    """Build directed graph from crawled pages"""
    print("=" * 60)
    print("BUILDING LINK GRAPH")
    print("=" * 60)
    
    with open(pages_file, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    
    G = nx.DiGraph()
    
    # Create URL to doc_id mapping
    url_to_id = {page['url']: page['doc_id'] for page in pages}
    
    # Add nodes with attributes
    for page in pages:
        G.add_node(page['doc_id'], 
                   url=page['url'], 
                   title=page['title'])
    
    # Add edges based on links
    edges_added = 0
    for page in pages:
        src_id = page['doc_id']
        for link in page['links']:
            if link in url_to_id:
                tgt_id = url_to_id[link]
                if src_id != tgt_id:  # Avoid self-loops
                    G.add_edge(src_id, tgt_id)
                    edges_added += 1
    
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {edges_added}")
    
    if G.number_of_nodes() > 0:
        avg_degree = np.mean([d for n, d in G.degree()])
        print(f"Average degree: {avg_degree:.2f}")
    
    return G, pages

def pagerank_iterative(G, d=0.85, max_iter=100, tol=1e-4):
    """Compute PageRank iteratively"""
    print("\n" + "=" * 60)
    print("COMPUTING PAGERANK")
    print("=" * 60)
    
    nodes = list(G.nodes())
    N = len(nodes)
    
    if N == 0:
        print("No nodes in graph. Exiting.")
        return {}, []
    
    # Initialize ranks uniformly
    pr = {node: 1.0/N for node in nodes}
    
    delta_history = []
    
    for iteration in range(max_iter):
        new_pr = {}
        
        # Calculate new PageRank for each node
        for node in nodes:
            # Sum of incoming contributions
            incoming_sum = 0.0
            predecessors = list(G.predecessors(node))
            
            if predecessors:
                for pred in predecessors:
                    out_deg = G.out_degree(pred)
                    if out_deg > 0:
                        incoming_sum += pr[pred] / out_deg
            
            # PageRank formula
            new_pr[node] = (1 - d) / N + d * incoming_sum
        
        # Calculate convergence metric
        delta = sum(abs(new_pr[n] - pr[n]) for n in nodes)
        delta_history.append(delta)
        
        pr = new_pr
        
        print(f"Iteration {iteration+1:3d}: delta = {delta:.6f}")
        
        if delta < tol:
            print(f"\n✓ Converged after {iteration+1} iterations")
            break
    
    if iteration == max_iter - 1:
        print(f"\n⚠️  Reached max iterations ({max_iter}) without convergence")
    
    return pr, delta_history

def display_top_pages(pr_scores, G, pages, top_n=10):
    """Display top pages by PageRank"""
    print("\n" + "=" * 60)
    print(f"TOP {top_n} PAGES BY PAGERANK")
    print("=" * 60)
    
    # Sort by PageRank score
    sorted_pr = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Rank':<6} {'Doc ID':<8} {'Score':<12} Title")
    print("-" * 80)
    
    for rank, (doc_id, score) in enumerate(sorted_pr[:top_n], 1):
        title = G.nodes[doc_id].get('title', 'Unknown')[:50]
        url = G.nodes[doc_id].get('url', 'Unknown')[:60]
        print(f"{rank:<6} {doc_id:<8} {score:<12.6f} {title}")
    
    return sorted_pr[:top_n]

def visualize_graph(G, pr_scores, output_file="outputs/link_graph.png"):
    """Visualize the link graph - Fixed colorbar issue"""
    print("\n" + "=" * 60)
    print("GENERATING GRAPH VISUALIZATION")
    print("=" * 60)
    
    # Create figure and axis explicitly
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes proportional to PageRank
    node_sizes = [pr_scores.get(n, 1.0/len(G)) * 10000 for n in G.nodes()]
    
    # Node colors based on PageRank (heatmap)
    node_colors = [pr_scores.get(n, 1.0/len(G)) for n in G.nodes()]
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes, 
                          node_color=node_colors,
                          cmap=plt.cm.YlOrRd,
                          alpha=0.8,
                          ax=ax)
    
    nx.draw_networkx_edges(G, pos, 
                          edge_color='gray', 
                          alpha=0.3, 
                          arrows=True, 
                          arrowsize=10,
                          ax=ax)
    
    # Add labels
    labels = {n: n for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Fixed colorbar creation
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                               norm=plt.Normalize(vmin=min(node_colors), 
                                                 vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='PageRank Score', shrink=0.8)
    
    ax.set_title("Information Retrieval Link Graph\n(Node size and color ∝ PageRank)", 
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Graph saved to {output_file}")
    
    # Try to display, but don't crash if no display available
    try:
        plt.show()
    except:
        print("  (Display not available - image saved to file)")
    
    plt.close()

def plot_convergence(delta_history, output_file="outputs/convergence_plot.png"):
    """Plot PageRank convergence - Fixed version"""
    print("\n" + "=" * 60)
    print("GENERATING CONVERGENCE PLOT")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(delta_history, marker='o', linewidth=2, markersize=4)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Sum of Absolute Changes (δ)", fontsize=12)
    ax.set_title("PageRank Convergence\n(Information Retrieval Domain)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Use log scale to show convergence better
    if min(delta_history) > 0:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"✓ Convergence plot saved to {output_file}")
    
    # Try to display
    try:
        plt.show()
    except:
        print("  (Display not available - image saved to file)")
    
    plt.close()

def compare_rankings(pages, pr_scores, G):
    """Compare PageRank ranking vs crawl order"""
    print("\n" + "=" * 60)
    print("RANKING COMPARISON ANALYSIS")
    print("=" * 60)
    
    # Crawl order (first crawled = rank 1)
    crawl_order = {page['doc_id']: i+1 for i, page in enumerate(pages)}
    
    # PageRank order
    pr_sorted = sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)
    pr_order = {doc_id: rank+1 for rank, (doc_id, _) in enumerate(pr_sorted)}
    
    print(f"\n{'Doc ID':<8} {'Crawl Rank':<12} {'PR Rank':<12} {'PR Score':<12} Title")
    print("-" * 90)
    
    for page in pages[:10]:  # First 10 crawled pages
        doc_id = page['doc_id']
        pr_rank = pr_order.get(doc_id, 'N/A')
        pr_score = pr_scores.get(doc_id, 0)
        title = page['title'][:40]
        print(f"{doc_id:<8} {crawl_order[doc_id]:<12} {str(pr_rank):<12} {pr_score:<12.6f} {title}")
    
    # Identify hubs and authorities
    print("\n" + "-" * 60)
    print("HUB AND AUTHORITY ANALYSIS:")
    
    if G.number_of_nodes() > 0:
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        
        # Top hubs (high out-degree)
        top_hubs = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nTop Hubs (many outgoing links):")
        for doc_id, deg in top_hubs:
            title = G.nodes[doc_id].get('title', 'Unknown')[:50]
            print(f"  {doc_id}: {deg} outgoing links - {title}")
        
        # Top authorities (high in-degree)
        top_auth = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nTop Authorities (many incoming links):")
        for doc_id, deg in top_auth:
            title = G.nodes[doc_id].get('title', 'Unknown')[:50]
            print(f"  {doc_id}: {deg} incoming links - {title}")
        
        # Correlation between PageRank and in-degree
        pr_values = [pr_scores.get(node, 0) for node in G.nodes()]
        in_deg_values = [in_degrees.get(node, 0) for node in G.nodes()]
        
        if len(pr_values) > 1:
            correlation = np.corrcoef(pr_values, in_deg_values)[0, 1]
            print(f"\nCorrelation (PageRank vs In-degree): {correlation:.3f}")
            if correlation > 0.7:
                print("  → Strong positive correlation: Pages with more incoming links tend to have higher PageRank")
            elif correlation > 0.3:
                print("  → Moderate positive correlation")

def save_results(pr_scores):
    """Save PageRank results to file"""
    output_data = {
        'pagerank_scores': {doc_id: float(score) for doc_id, score in pr_scores.items()},
        'top_10': dict(sorted(pr_scores.items(), key=lambda x: x[1], reverse=True)[:10])
    }
    
    with open("data/pagerank_scores.json", 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ PageRank scores saved to data/pagerank_scores.json")

if __name__ == "__main__":
    print("=" * 60)
    print("PAGERANK COMPUTATION FOR IR DOMAIN")
    print("=" * 60)
    
    try:
        # Build link graph
        G, pages = build_link_graph("data/crawled_pages.json")
        
        if G.number_of_nodes() == 0:
            print("\n⚠️  No nodes in graph. Check if crawled data exists.")
            exit(1)
        
        # Compute PageRank
        pr_scores, delta_history = pagerank_iterative(G, d=0.85)
        
        if not pr_scores:
            print("⚠️  PageRank computation failed.")
            exit(1)
        
        # Display top pages
        top10 = display_top_pages(pr_scores, G, pages)
        
        # Save PageRank scores
        save_results(pr_scores)
        
        # Generate visualizations
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        visualize_graph(G, pr_scores, "outputs/link_graph.png")
        
        if delta_history:
            plot_convergence(delta_history, "outputs/convergence_plot.png")
        
        # Analysis
        compare_rankings(pages, pr_scores, G)
        
        print("\n" + "=" * 60)
        print("PAGERANK ANALYSIS COMPLETE")
        print("=" * 60)
        print("\nGenerated files:")
        print("  ✓ data/pagerank_scores.json")
        print("  ✓ outputs/link_graph.png")
        print("  ✓ outputs/convergence_plot.png")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  Error: {e}")
        print("Make sure to run crawler.py first to generate data/crawled_pages.json")
    except Exception as e:
        print(f"\n⚠️  Error: {e}")
        print("Please check the error and try again.")
