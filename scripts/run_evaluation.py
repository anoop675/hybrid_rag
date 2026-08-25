"""Build or load indexes, encode the corpus, and run full evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid_rag.config import (
    BM25_INDEX_PATH,
    DOCSTORE_PATH,
    COLBERT_EMB_PATH,
    BM25_TOP_K,
    DEVICE,
)
from hybrid_rag.data import load_docstore, load_queries, load_qrels, load_dataset, save_docstore
from hybrid_rag.bm25 import BM25Retriever
from hybrid_rag.colbert import ColBERTReranker
from hybrid_rag.rag import RAGModule
from hybrid_rag.evaluate import run_full_evaluation

import os
import time
import gc
import torch

print("Phase 1: BM25 index")

bm25_exists = os.path.exists(os.path.join(BM25_INDEX_PATH, "doc_ids.json"))
store_exists = os.path.exists(DOCSTORE_PATH)

if bm25_exists and store_exists:
    print("Index and docstore exist, loading...")
    start_time = time.time()
    docs = load_docstore()
    doc_map = {doc["doc_id"]: doc for doc in docs}
    bm25 = BM25Retriever()
    bm25.load_index()
    print(f"{len(docs):,} docs loaded ({time.time() - start_time:.1f}s)")
    queries = load_queries()
    qrels = load_qrels()

elif bm25_exists and not store_exists:
    print("BM25 index found but docstore.pkl is missing.")
    print("Rebuilding docstore from raw data (BM25 index will be reused)...")
    start_time = time.time()
    docs, queries, qrels = load_dataset(max_docs=None, qrels_split="test", verbose=True)
    save_docstore(docs)
    doc_map = {doc["doc_id"]: doc for doc in docs}
    bm25 = BM25Retriever()
    bm25.load_index()
    print(f"{len(docs):,} docs ready, docstore saved ({time.time() - start_time:.1f}s)")

else:
    print("No index found, building from scratch...")
    start_time = time.time()
    docs, queries, qrels = load_dataset(max_docs=None, qrels_split="test", verbose=True)
    save_docstore(docs)
    doc_map = {doc["doc_id"]: doc for doc in docs}
    bm25 = BM25Retriever()
    bm25.build_index(docs)
    bm25.save_index()
    print(f"{len(docs):,} docs indexed ({time.time() - start_time:.1f}s)")

print("BM25 speed test:")
for query in queries[:5]:
    start_time = time.time()
    results = bm25.retrieve(query["text"], top_k=BM25_TOP_K)
    elapsed = time.time() - start_time
    top_score = results[0][1] if results else 0
    print(f"[{elapsed:.4f}s] {query['text'][:60]} -> {len(results)} results (top: {top_score:.2f})")

print("Phase 2: ColBERT re-ranker")

if "fine_tuned_reranker" in locals() and fine_tuned_reranker.model is not None:
    print("Using previously fine-tuned ColBERTReranker model.")
    reranker = fine_tuned_reranker
else:
    print("Loading a fresh ColBERTReranker model (no fine-tuning applied).")
    reranker = ColBERTReranker()
    reranker.load_model()

print("Phase 2b: ColBERT document embeddings")

embeddings_exist = os.path.exists(COLBERT_EMB_PATH)
force_reencode = (
    "fine_tuned_reranker" in locals()
    and fine_tuned_reranker.model is not None
    and getattr(fine_tuned_reranker, "best_epoch", 0) > 1
)
if "fine_tuned_reranker" in locals() and fine_tuned_reranker.model is not None:
    if not force_reencode:
        print(
            "Fine-tuned model exists but best_epoch <= 1. "
            "Skipping re-encode and using base model embeddings instead."
        )

if embeddings_exist and not force_reencode:
    print(f"Found saved embeddings at {COLBERT_EMB_PATH}")
    print("Loading pre-computed ColBERT embeddings...")
    start_time = time.time()
    reranker.load_embeddings()
    size_mb = os.path.getsize(COLBERT_EMB_PATH) / (1024 * 1024)
    print(
        f"Loaded {len(reranker.doc_embeddings):,} doc embeddings "
        f"({size_mb:.1f} MB, {time.time() - start_time:.1f}s)"
    )
elif force_reencode and embeddings_exist:
    print("Fine-tuned model detected and embeddings exist. Re-encoding with the fine-tuned model...")
    os.remove(COLBERT_EMB_PATH)
    reranker.doc_embeddings = None
else:
    print("No saved embeddings found, or re-encoding was forced. Encoding corpus with ColBERT.")
    print("Offloading RAG model from GPU to free VRAM for encoding...")

    if "rag" not in dir() or rag.model is None:
        rag = RAGModule()

    reranker.model.to("cpu")
    if reranker.linear is not None:
        reranker.linear.to("cpu")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"GPU free after offload: {free_gb:.1f} GB")

    reranker.model.to(DEVICE)
    if reranker.linear is not None:
        reranker.linear.to(DEVICE)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    encode_batch_size = 8
    print(f"Encoding {len(docs):,} documents with ColBERT (runs once, then saved)...")
    start_time = time.time()
    reranker.encode_corpus(docs, batch_size=encode_batch_size)
    reranker.save_embeddings()
    size_mb = os.path.getsize(COLBERT_EMB_PATH) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"Embeddings saved to: {COLBERT_EMB_PATH}")
    print(f"Size: {size_mb:.1f} MB Time: {elapsed:.1f}s")
    print("Download this file to reuse embeddings in future sessions.")

    reranker.model.to("cpu")
    if reranker.linear is not None:
        reranker.linear.to("cpu")
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

if "rag" not in dir() or rag.model is None:
    rag = RAGModule()

print("Moving ColBERT re-ranker back to DEVICE for evaluation...")
reranker.model.to(DEVICE)
if reranker.linear is not None:
    reranker.linear.to(DEVICE)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"GPU free after ColBERT re-activation: {free_gb:.1f} GB")

print("Phase 3: Full evaluation with relevance judgments")

report = run_full_evaluation(
    bm25=bm25,
    reranker=reranker,
    docs=docs,
    doc_map=doc_map,
    queries=queries,
    qrels=qrels,
)
print("Done!")
