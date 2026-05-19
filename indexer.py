#!/usr/bin/env python3
"""
IR Domain Web Crawler - Fixed Version
Crawls Information Retrieval related pages
"""

import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
from collections import deque
import os
import sys

# Configuration
SEED_URLS = [
    "https://en.wikipedia.org/wiki/Information_retrieval",
    "https://en.wikipedia.org/wiki/Tf-idf"
]

MAX_PAGES = 30
DELAY = 1.0  # Politeness delay in seconds

# Create output directories
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

OUTPUT_FILE = "data/crawled_pages.json"
LOG_FILE = "data/crawl_log.txt"

class FocusedCrawler:
    def __init__(self, seeds, max_pages=30, delay=1.0):
        self.frontier = deque(seeds)
        self.visited = set()
        self.max_pages = max_pages
        self.delay = delay
        self.pages = []
        
    def is_valid(self, url):
        """Filter valid URLs"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False
            if url.endswith(('.pdf', '.zip', '.png', '.jpg', '.gif', '.css', '.js', '.svg')):
                return False
            # Accept Wikipedia and other educational domains
            allowed_domains = ['wikipedia.org', 'stanford.edu', 'mit.edu', 'acm.org', 'arxiv.org']
            return any(domain in parsed.netloc for domain in allowed_domains)
        except:
            return False
    
    def crawl(self):
        """Main crawling loop"""
        print(f"Starting crawl with {len(SEED_URLS)} seed URLs")
        print(f"Target: {self.max_pages} pages with {self.delay}s delay")
        print("-" * 60)
        
        try:
            with open(LOG_FILE, 'w', encoding='utf-8') as log:
                log.write("Crawl Log - IR Domain\n")
                log.write("="*50 + "\n\n")
                
                while self.frontier and len(self.pages) < self.max_pages:
                    url = self.frontier.popleft()
                    
                    if url in self.visited:
                        continue
                    
                    progress = len(self.pages) + 1
                    print(f"[{progress}/{self.max_pages}] Crawling: {url[:80]}...")
                    
                    try:
                        headers = {
                            'User-Agent': 'MiniIRSearchEngine/1.0 (Educational Project)'
                        }
                        resp = requests.get(url, timeout=10, headers=headers)
                        
                        if resp.status_code != 200:
                            log.write(f"SKIP [{resp.status_code}]: {url}\n")
                            print(f"  ⚠ HTTP {resp.status_code} - Skipping")
                            continue
                        
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        
                        # Extract title
                        title = soup.title.string.strip() if soup.title else "No Title"
                        
                        # Clean text - remove script, style, nav elements
                        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                            tag.decompose()
                        text = soup.get_text(separator=' ', strip=True)
                        
                        # Extract links
                        links = set()
                        for a_tag in soup.find_all('a', href=True):
                            href = a_tag['href']
                            absolute = urljoin(url, href)
                            if self.is_valid(absolute):
                                links.add(absolute)
                        
                        # Store page data (limit text to first 5000 chars)
                        page_data = {
                            'doc_id': f"doc{len(self.pages)}",
                            'url': url,
                            'title': title,
                            'text': text[:5000],
                            'links': list(links)[:50]  # Limit to first 50 links
                        }
                        self.pages.append(page_data)
                        self.visited.add(url)
                        
                        log.write(f"CRAWLED: {url}\n")
                        log.write(f"  Title: {title}\n")
                        log.write(f"  Links found: {len(links)}\n\n")
                        
                        # Add new URLs to frontier
                        new_links = 0
                        for link in links:
                            if link not in self.visited:
                                self.frontier.append(link)
                                new_links += 1
                        
                        print(f"  ✓ Crawled. {len(links)} links found, {new_links} new URLs added")
                        log.write(f"  Added {new_links} new URLs to frontier\n")
                        
                        # Politeness delay
                        print(f"  Sleeping {self.delay}s...")
                        time.sleep(self.delay)
                        
                    except requests.RequestException as e:
                        print(f"  ✗ Network Error: {e}")
                        log.write(f"ERROR [{url}]: {e}\n")
                        continue
                    except Exception as e:
                        print(f"  ✗ Unexpected error: {e}")
                        log.write(f"ERROR [{url}]: {e}\n")
                        continue
                
        except IOError as e:
            print(f"Error writing to log file: {e}")
        
        print(f"\nCrawling complete. {len(self.pages)} pages collected.")
        
        # Save results
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.pages, f, indent=2, ensure_ascii=False)
            print(f"✓ Data saved to {OUTPUT_FILE}")
        except IOError as e:
            print(f"Error saving crawled data: {e}")
        
        return self.pages

def create_mock_data():
    """Create comprehensive mock IR dataset"""
    print("\n⚠️  Creating mock IR dataset with 30 documents...")
    mock_pages = []
    
    # IR-related topics with realistic content
    ir_documents = [
        {
            "title": "Information Retrieval - Overview",
            "text": "Information retrieval (IR) is the process of obtaining information from a collection of resources. The most common form is text retrieval where documents are ranked by relevance to a query using techniques like TF-IDF and BM25. Modern IR systems use inverted indexes for efficient retrieval.",
            "links": [
                "https://en.wikipedia.org/wiki/Inverted_index",
                "https://en.wikipedia.org/wiki/Tf-idf",
                "https://en.wikipedia.org/wiki/Precision_and_recall"
            ]
        },
        {
            "title": "TF-IDF Weighting Scheme",
            "text": "TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic that reflects how important a word is to a document in a collection. TF measures how frequently a term occurs in a document, while IDF measures how rare or common a term is across documents.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Vector_space_model"
            ]
        },
        {
            "title": "Inverted Index Structure",
            "text": "An inverted index is a data structure that maps terms to their locations in documents. It is the fundamental component of modern search engines, enabling fast full-text searches. Each term points to a posting list containing document IDs and optionally term frequencies and positions.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/PageRank"
            ]
        },
        {
            "title": "PageRank Algorithm Explained",
            "text": "PageRank is an algorithm used by Google Search to rank web pages in their search engine results. It works by counting the number and quality of links to a page to determine a rough estimate of the website's importance. The algorithm uses a damping factor typically set to 0.85.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Tf-idf"
            ]
        },
        {
            "title": "Vector Space Model in IR",
            "text": "The Vector Space Model represents documents and queries as vectors in a high-dimensional space. Each dimension corresponds to a term. Cosine similarity is commonly used to measure the similarity between document and query vectors.",
            "links": [
                "https://en.wikipedia.org/wiki/Tf-idf",
                "https://en.wikipedia.org/wiki/Cosine_similarity"
            ]
        },
        {
            "title": "Precision and Recall Metrics",
            "text": "Precision and recall are fundamental evaluation metrics in information retrieval. Precision measures the fraction of retrieved documents that are relevant, while recall measures the fraction of relevant documents that were retrieved. F-measure combines both into a single metric.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/F-score"
            ]
        },
        {
            "title": "Boolean Retrieval Model",
            "text": "The Boolean model is a simple retrieval model based on set theory and Boolean algebra. Queries are expressed as Boolean expressions using AND, OR, and NOT operators. While simple, it lacks the ranking capability of more advanced models like the vector space model.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Vector_space_model"
            ]
        },
        {
            "title": "Web Crawling Fundamentals",
            "text": "Web crawlers systematically browse the World Wide Web to index pages for search engines. They start from seed URLs and follow links to discover new pages. Politeness policies ensure crawlers don't overwhelm web servers.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/PageRank"
            ]
        },
        {
            "title": "Latent Semantic Indexing",
            "text": "Latent Semantic Indexing (LSI) is a technique in natural language processing that analyzes relationships between documents and terms. It uses singular value decomposition to identify patterns in the relationships between terms and concepts.",
            "links": [
                "https://en.wikipedia.org/wiki/Tf-idf",
                "https://en.wikipedia.org/wiki/Vector_space_model"
            ]
        },
        {
            "title": "Query Expansion Techniques",
            "text": "Query expansion improves retrieval performance by adding additional terms to the original query. Techniques include relevance feedback, thesaurus-based expansion, and automatic local analysis like pseudo-relevance feedback.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Precision_and_recall"
            ]
        },
        {
            "title": "Document Indexing Methods",
            "text": "Document indexing is the process of creating data structures to enable fast information retrieval. Common methods include inverted indexing, signature files, and suffix trees. Modern search engines primarily use inverted indexes for efficiency.",
            "links": [
                "https://en.wikipedia.org/wiki/Inverted_index",
                "https://en.wikipedia.org/wiki/Tf-idf"
            ]
        },
        {
            "title": "Search Engine Architecture",
            "text": "Modern search engines consist of crawlers, indexers, and query processors. The crawler collects web pages, the indexer builds efficient data structures, and the query processor ranks documents based on relevance algorithms.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/PageRank"
            ]
        },
        {
            "title": "Relevance Feedback in IR",
            "text": "Relevance feedback is a technique where users indicate which retrieved documents are relevant. The system then modifies the query based on this feedback to improve future retrieval results.",
            "links": [
                "https://en.wikipedia.org/wiki/Vector_space_model",
                "https://en.wikipedia.org/wiki/Precision_and_recall"
            ]
        },
        {
            "title": "BM25 Ranking Function",
            "text": "BM25 is a ranking function used by search engines to rank matching documents by their relevance to a query. It's based on the probabilistic retrieval framework and improves upon TF-IDF by considering document length normalization.",
            "links": [
                "https://en.wikipedia.org/wiki/Tf-idf",
                "https://en.wikipedia.org/wiki/Information_retrieval"
            ]
        },
        {
            "title": "Information Filtering Systems",
            "text": "Information filtering systems deal with removing irrelevant information from streams. Unlike traditional IR systems that respond to queries, filtering systems monitor continuous information streams and select items matching user profiles.",
            "links": [
                "https://en.wikipedia.org/wiki/Information_retrieval",
                "https://en.wikipedia.org/wiki/Precision_and_recall"
            ]
        }
    ]
    
    # Generate 30 documents by repeating and varying the base documents
    for i in range(30):
        base_doc = ir_documents[i % len(ir_documents)]
        
        # Add variation to text
        variant_texts = [
            " This concept is fundamental to modern search systems.",
            " Understanding this topic is crucial for IR practitioners.",
            " Many commercial search engines implement variations of this approach.",
            " Academic research continues to improve upon these foundations."
        ]
        
        mock_pages.append({
            'doc_id': f"doc{i}",
            'url': f"https://example.ir/pages/document{i}",
            'title': f"{base_doc['title']} - Part {i//len(ir_documents) + 1}",
            'text': base_doc['text'] + variant_texts[i % len(variant_texts)],
            'links': base_doc['links'] + [
                f"https://example.ir/pages/document{j}" 
                for j in range(max(0, i-2), min(30, i+3)) 
                if j != i and j % 3 == 0
            ]
        })
    
    return mock_pages[:30]

if __name__ == "__main__":
    print("=" * 60)
    print("IR DOMAIN MINI SEARCH ENGINE - CRAWLER")
    print("=" * 60)
    print()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    pages = None
    
    try:
        # Attempt live crawling
        crawler = FocusedCrawler(SEED_URLS, MAX_PAGES, DELAY)
        pages = crawler.crawl()
        
        if len(pages) < 10:
            print(f"\n⚠️  Only {len(pages)} pages crawled (minimum 10 required for good analysis).")
            print("Using mock data instead for complete demonstration...")
            pages = create_mock_data()
            # Save mock data
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(pages, f, indent=2, ensure_ascii=False)
            print(f"✓ Mock data saved to {OUTPUT_FILE}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Crawling interrupted by user.")
        print("Using mock data for demonstration...")
        pages = create_mock_data()
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)
        print(f"✓ Mock data saved to {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"\n⚠️  Crawling error: {e}")
        print("Using mock data for demonstration...")
        pages = create_mock_data()
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(pages, f, indent=2, ensure_ascii=False)
        print(f"✓ Mock data saved to {OUTPUT_FILE}")
    
    # Print summary
    if pages:
        print("\n" + "=" * 60)
        print("CRAWLING SUMMARY")
        print("=" * 60)
        print(f"Total pages collected: {len(pages)}")
        print(f"Output file: {OUTPUT_FILE}")
        print(f"Log file: {LOG_FILE}")
        
        # Show first few titles
        print("\nFirst 5 page titles:")
        for page in pages[:5]:
            print(f"  - {page['title'][:80]}")
