import os
import torch

DATASET_NAME = "beir/scifact"
DATASET_SPLIT = "corpus"

BASE_DIR = os.getcwd()
INDEX_DIR = os.path.join(BASE_DIR, "indexes")
BM25_INDEX_PATH = os.path.join(INDEX_DIR, "bm25_index")
COLBERT_EMB_PATH = os.path.join(INDEX_DIR, "colbert_embeddings.pt")
DOCSTORE_PATH = os.path.join(INDEX_DIR, "docstore.pkl")

BM25_K1 = 1.5
BM25_B = 0.75
BM25_TOP_K = 100

RERANKER_MODEL = "colbert-ir/colbertv2.0"
RERANKER_BATCH_SIZE = 32
RERANKER_TOP_N = 100
RERANKER_MAX_LENGTH = 256

# RAG_MODEL = "Qwen/Qwen2-1.5B-Instruct"
RAG_MODEL = "Qwen/Qwen2.5-3B-Instruct"
RAG_MAX_CONTEXT_TOKENS = 4096
RAG_MAX_NEW_TOKENS = 500
RAG_NUM_EVIDENCE_DOCS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVAL_MAX_QUERIES = 500
EVAL_METRICS = ["nDCG@10", "AP@10", "R@100", "P@10", "RR@10"]

os.makedirs(INDEX_DIR, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Base dir: {BASE_DIR}")
print(f"Dataset: {DATASET_NAME}")
