"""Hybrid RAG engine: BM25 + ColBERT re-ranking + RAG generation on SciFact."""

from .config import (
    DATASET_NAME,
    DATASET_SPLIT,
    BASE_DIR,
    INDEX_DIR,
    BM25_INDEX_PATH,
    COLBERT_EMB_PATH,
    DOCSTORE_PATH,
    BM25_K1,
    BM25_B,
    BM25_TOP_K,
    RERANKER_MODEL,
    RERANKER_BATCH_SIZE,
    RERANKER_TOP_N,
    RERANKER_MAX_LENGTH,
    RAG_MODEL,
    RAG_MAX_CONTEXT_TOKENS,
    RAG_MAX_NEW_TOKENS,
    RAG_NUM_EVIDENCE_DOCS,
    DEVICE,
    EVAL_MAX_QUERIES,
    EVAL_METRICS,
)
from .data import (
    clean_text,
    tokenize_simple,
    load_corpus,
    load_queries,
    load_train_queries,
    load_qrels,
    load_dataset,
    save_docstore,
    load_docstore,
)
from .bm25 import BM25Retriever
from .colbert import ColBERTReranker
from .finetune import FineTuneColBERTReranker, create_training_triplets, run_fine_tuning_loop
from .rag import RAGModule
from .pipeline import build_indexes, search
from .evaluate import (
    evaluate_retrieval,
    per_query_scores,
    compare_pipelines,
    rank_displacement,
    run_full_evaluation,
)

__all__ = [
    "BM25Retriever",
    "ColBERTReranker",
    "FineTuneColBERTReranker",
    "RAGModule",
    "build_indexes",
    "search",
    "run_full_evaluation",
    "create_training_triplets",
    "run_fine_tuning_loop",
]
