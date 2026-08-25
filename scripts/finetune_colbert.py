"""Run ColBERT projection-layer fine-tuning."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from hybrid_rag.config import BM25_TOP_K, DOCSTORE_PATH, BM25_INDEX_PATH
from hybrid_rag.data import load_docstore, load_corpus, save_docstore, load_queries, load_qrels, load_train_queries
from hybrid_rag.bm25 import BM25Retriever
from hybrid_rag.finetune import FineTuneColBERTReranker, create_training_triplets, run_fine_tuning_loop

print("Fine-tuning ColBERT (projection layer only)")

if "docs" not in globals() or "doc_map" not in globals():
    print("Loading corpus...")
    if os.path.exists(DOCSTORE_PATH):
        docs = load_docstore()
    else:
        docs = load_corpus(max_docs=None, verbose=True)
        save_docstore(docs)
    doc_map = {doc["doc_id"]: doc for doc in docs}

if "bm25" not in globals():
    print("Loading BM25 index...")
    bm25 = BM25Retriever()
    if os.path.exists(os.path.join(BM25_INDEX_PATH, "doc_ids.json")):
        bm25.load_index()
    else:
        bm25.build_index(docs)
        bm25.save_index()

if "queries" not in globals() or "qrels" not in globals():
    print("Loading test queries and qrels (for reference)...")
    queries = load_queries()
    qrels = load_qrels(split="test")

print("Loading train queries and qrels...")
train_queries = load_train_queries()
train_qrels = load_qrels(split="train", verbose=True)

training_triplets = create_training_triplets(
    train_queries,
    docs,
    train_qrels,
    bm25_retriever=bm25,
    doc_map=doc_map,
    num_negatives=3,
    bm25_top_k_for_negatives=BM25_TOP_K,
)

if training_triplets:
    EPOCHS = 3
    BATCH_SIZE = 4
    VAL_RATIO = 0.15

    n_train = int(len(training_triplets) * (1 - VAL_RATIO))
    total_steps = max(1, (n_train // BATCH_SIZE)) * EPOCHS

    fine_tuned_reranker = FineTuneColBERTReranker()
    fine_tuned_reranker.load_model()
    fine_tuned_reranker.setup_training(
        learning_rate=1e-4,
        total_steps=total_steps,
        warmup_ratio=0.1,
    )

    run_fine_tuning_loop(
        fine_tuned_reranker,
        training_triplets,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        patience=2,
    )

    print("To save: fine_tuned_reranker.model.save_pretrained('./fine_tuned_colbert_model')")
else:
    print("No training triplets created. Check that train_qrels is non-empty.")
