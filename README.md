# Hybrid RAG Pipeline

BM25 retrieval, ColBERT-style late-interaction re-ranking (optional projection-layer fine-tuning), and grounded RAG generation on [SciFact](https://huggingface.co/datasets/BeIR/scifact).

## Layout

```
.
├── hybrid_rag/                 Core library
│   ├── config.py               Paths and hyperparameters
│   ├── data.py                 SciFact corpus, queries, qrels
│   ├── bm25.py                 First-stage BM25 retrieval
│   ├── colbert.py              ColBERT re-ranking
│   ├── finetune.py             Projection-layer fine-tuning
│   ├── rag.py                  Qwen generation
│   ├── pipeline.py             Index build and search
│   └── evaluate.py             Retrieval metrics
├── scripts/                    Command-line entry points
│   ├── inspect_dataset.py      List SciFact files
│   ├── finetune_colbert.py     Optional ColBERT fine-tune
│   ├── run_evaluation.py       Indexes, embeddings, eval
│   └── interactive_query.py    Query after models are loaded
└── notebooks/
    └── run_hybrid_rag.ipynb    Notebook runner
```

## Setup

A GPU is strongly recommended (the original notebook used an A100). RAG uses 4-bit Qwen2.5-3B via `bitsandbytes`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or: `pip install -r requirements.txt` and keep the repo root on `PYTHONPATH`.

Run commands from the **repository root**. Index paths are `os.getcwd()/indexes/`.

## Run

Optional ColBERT fine-tune (train split only; test qrels are not used):

```bash
python scripts/finetune_colbert.py
```

Build/load BM25, encode (or load) ColBERT document embeddings, then evaluate BM25 vs hybrid:

```bash
python scripts/run_evaluation.py
```

Interactive RAG is easiest from `notebooks/run_hybrid_rag.ipynb` after the evaluation cell, because `bm25`, `reranker`, `rag`, and `doc_map` stay in session state.

First-time runs download SciFact, ColBERT, and Qwen weights, then write:

- `indexes/bm25_index/`
- `indexes/docstore.pkl`
- `indexes/colbert_embeddings.pt`

Those artifacts are gitignored. Reuse them on later runs.

## Pipeline

1. BM25 retrieves `BM25_TOP_K` (100) candidates  
2. ColBERT MaxSim re-ranks to `RERANKER_TOP_N` (100)  
3. Top `RAG_NUM_EVIDENCE_DOCS` (5) passages go to Qwen for a cited answer  
4. Evaluation reports nDCG@10, AP@10, R@100, P@10, RR@10 for BM25 vs hybrid  

Fine-tuning freezes the BERT backbone and trains only the ColBERT linear projection with InfoNCE on BM25 hard negatives.
