"""Interactive and sample queries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_rag.config import DEVICE
from hybrid_rag.pipeline import search

print("Enter your query below to get a RAG-generated answer.")

if "reranker" in globals() and reranker.model:
    reranker.model.to(DEVICE)
    if reranker.linear:
        reranker.linear.to(DEVICE)

if "rag" in globals() and rag.model is None:
    rag.load_model()

query = input("Your Query> ").strip()

if query:
    print("Processing your query...")
    search(query, bm25, reranker, rag, doc_map)
else:
    print("No query entered. Exiting interactive query mode.")

print("Sample retrieval and ranking explanation")

sample_query = "role of vitamin d in bone health"
search_results = search(sample_query, bm25, reranker, rag, doc_map)
