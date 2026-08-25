from .config import BASE_DIR, DATASET_NAME, DOCSTORE_PATH

import re
import time
import pickle
from typing import Dict, List
from datasets import load_dataset as hf_load_dataset
import csv
import urllib.request
import os

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_simple(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return [token for token in tokens if len(token) > 1]


SCIFACT_QRELS_TEST_URL = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv"
SCIFACT_QRELS_TRAIN_URL = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/train.tsv"

CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _download(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def load_corpus(max_docs=None, verbose=True):
    start_time = time.time()
    if verbose:
        print("Loading SciFact corpus from HuggingFace...")

    dataset = hf_load_dataset(DATASET_NAME, "corpus", split="corpus")

    docs = []
    for i, record in enumerate(dataset):
        if max_docs and i >= max_docs:
            break
        doc_id = str(record["_id"]).strip()
        if not doc_id:
            continue
        title = clean_text(str(record.get("title", "")))
        body = clean_text(str(record.get("text", "")))
        docs.append({
            "doc_id": f"D{doc_id}",
            "title": title,
            "body": body,
            "body_tokens": tokenize_simple(f"{title} {body}"),
        })
    if verbose:
        print(f"Loaded {len(docs):,} docs in {time.time() - start_time:.2f}s")
    return docs


def load_queries(verbose=True):
    if verbose:
        print("Loading SciFact queries from HuggingFace...")

    dataset = hf_load_dataset(DATASET_NAME, "queries", split="queries")

    queries = []
    for record in dataset:
        query_id = str(record["_id"]).strip()
        text = clean_text(str(record.get("text", "")))
        if query_id and text:
            queries.append({"query_id": query_id, "text": text})

    if verbose:
        print(f"Loaded {len(queries):,} queries")
    return queries


def load_train_queries(verbose=True):
    """Load queries that appear in the training qrels. Do not use these for evaluation."""
    if verbose:
        print("Loading SciFact train queries from train qrels...")

    train_qrels_path = _download(
        SCIFACT_QRELS_TRAIN_URL, os.path.join(CACHE_DIR, "qrels_train.tsv")
    )
    train_query_ids = set()
    with open(train_qrels_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            train_query_ids.add(str(record["query-id"]).strip())

    all_queries = load_queries(verbose=False)
    train_queries = [query for query in all_queries if query["query_id"] in train_query_ids]

    if verbose:
        print(f"Loaded {len(train_queries):,} train queries (fine-tuning only)")
    return train_queries


def load_qrels(split="test", verbose=True):
    if verbose:
        print(f"Loading SciFact qrels ({split}) from HuggingFace...")

    url = SCIFACT_QRELS_TEST_URL if split == "test" else SCIFACT_QRELS_TRAIN_URL
    path = _download(url, os.path.join(CACHE_DIR, f"qrels_{split}.tsv"))

    qrels = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for record in reader:
            query_id = str(record["query-id"]).strip()
            doc_id = str(record["corpus-id"]).strip()
            score = int(record.get("score", 1))
            if query_id and doc_id:
                qrels.append({
                    "query_id": query_id,
                    "doc_id": f"D{doc_id}",
                    "relevance": score,
                })
    if verbose:
        print(f"Loaded {len(qrels):,} relevance judgments")
    return qrels


def load_dataset(max_docs, qrels_split, verbose):
    start_time = time.time()
    if verbose:
        print(f"Loading SciFact dataset from HuggingFace ({DATASET_NAME})...")
    docs = load_corpus(max_docs=max_docs, verbose=verbose)
    queries = load_queries(verbose=verbose)
    qrels = load_qrels(split=qrels_split, verbose=verbose)
    if verbose:
        elapsed = time.time() - start_time
        print(f"Total: {len(docs):,} docs, {len(queries):,} queries, {len(qrels):,} qrels ({elapsed:.2f}s)")
    return docs, queries, qrels


def save_docstore(docs: List[Dict], path: str = None):
    path = path or DOCSTORE_PATH
    store = [{k: v for k, v in doc.items() if k != "body_tokens"} for doc in docs]
    with open(path, "wb") as handle:
        pickle.dump(store, handle)
    print(f"Docstore saved: {len(store):,} docs -> {path}")


def load_docstore(path: str = None) -> List[Dict]:
    path = path or DOCSTORE_PATH
    with open(path, "rb") as handle:
        store = pickle.load(handle)
    print(f"Docstore loaded: {len(store):,} docs")
    return store
