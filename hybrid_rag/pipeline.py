from .config import BM25_TOP_K, RERANKER_TOP_N, RAG_NUM_EVIDENCE_DOCS
from .data import load_corpus, save_docstore
from .bm25 import BM25Retriever

import time


def build_indexes():
    print("Offline phase: building indexes")

    docs = load_corpus()
    save_docstore(docs)

    bm25 = BM25Retriever()
    bm25.build_index(docs)
    bm25.save_index()

    print(f"Offline phase complete. {len(docs):,} documents indexed.")
    return docs


def search(query: str, bm25, reranker, rag, doc_map: dict, verbose: bool = True):
    start_time = time.time()

    if verbose:
        print(f"Query: {query}")
        print(f"Step 1: BM25 retrieval (top-{BM25_TOP_K})...")

    bm25_results = bm25.retrieve(query, top_k=BM25_TOP_K)
    bm25_time = time.time() - start_time

    if verbose:
        print(f"Retrieved {len(bm25_results)} candidates ({bm25_time:.3f}s)")
        for i, (doc_id, score) in enumerate(bm25_results[:3], 1):
            doc = doc_map.get(doc_id, {})
            print(f"{i}. [{score:.2f}] {doc.get('title', doc_id)}")

    rerank_start = time.time()
    if verbose:
        print(f"Step 2: ColBERT re-ranking (top-{RERANKER_TOP_N})...")

    candidates = [doc_map[doc_id] for doc_id, _ in bm25_results if doc_id in doc_map]
    rerank_results = reranker.rerank(query, candidates, top_n=RERANKER_TOP_N)
    rerank_time = time.time() - rerank_start

    if verbose:
        print(f"Re-ranked to {len(rerank_results)} results ({rerank_time:.2f}s)")
        for i, (doc_id, score) in enumerate(rerank_results[:3], 1):
            doc = doc_map.get(doc_id, {})
            print(f"{i}. [{score:.4f}] {doc.get('title', doc_id)}")

    rag_start = time.time()
    if verbose:
        print(f"Step 3: RAG answer generation (from top-{RAG_NUM_EVIDENCE_DOCS} docs)...")

    evidence_docs = [
        doc_map[doc_id]
        for doc_id, _ in rerank_results[:RAG_NUM_EVIDENCE_DOCS]
        if doc_id in doc_map
    ]
    rag_result = rag.generate(query, evidence_docs)
    rag_time = time.time() - rag_start
    total_time = time.time() - start_time

    if verbose:
        print(f"ANSWER: {rag_result['answer']}")
        print("Sources:")
        for source in rag_result["sources"]:
            print(f"{source['citation']} {source.get('title', 'untitled')}")

        print("Evidence documents fed to the LLM:")
        for i, doc in enumerate(evidence_docs, 1):
            print(f"[{i}] {doc['title']}")
            print(doc["body"][:500] + "..." if len(doc["body"]) > 500 else doc["body"])

        print("RAG prompt sent to the LLM:")
        print(rag_result["prompt"])
        print(
            f"Timing: BM25={bm25_time:.3f}s Rerank={rerank_time:.2f}s "
            f"RAG={rag_time:.2f}s Total={total_time:.2f}s"
        )

    return {
        "query": query,
        "bm25_results": bm25_results,
        "rerank_results": rerank_results,
        "rag_result": rag_result,
        "timing": {
            "bm25": bm25_time,
            "rerank": rerank_time,
            "rag": rag_time,
            "total": total_time,
        },
    }
