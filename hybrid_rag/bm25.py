from .config import BM25_INDEX_PATH, BM25_K1, BM25_B, BM25_TOP_K
from .data import tokenize_simple, load_corpus

import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import bm25s


class BM25Retriever:
    def __init__(self):
        self.bm25: Optional[bm25s.BM25] = None
        self.doc_ids: List[str] = []

    def _get_save_dir(self, path: Optional[str]) -> Path:
        base_path = Path(path or BM25_INDEX_PATH)
        return base_path.parent / base_path.stem

    def build_index(self, docs: List[Dict]):
        print(f"Building BM25 index over {len(docs):,} documents...")
        start_time = time.time()

        self.doc_ids = [doc["doc_id"] for doc in docs]
        corpus_tokens = [doc["body_tokens"] for doc in docs]

        self.bm25 = bm25s.BM25(k1=BM25_K1, b=BM25_B)
        self.bm25.index(corpus_tokens, show_progress=True)

        print(f"BM25 index built in {time.time() - start_time:.1f}s")

    def save_index(self, path: str = None):
        if not self.bm25:
            raise ValueError("No index to save. Call build_index first.")

        save_dir = self._get_save_dir(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "doc_ids.json", "w") as handle:
            json.dump(self.doc_ids, handle)

        self.bm25.save(save_dir, corpus=None)
        print(f"BM25 index saved -> {save_dir}")

    def load_index(self, path: str = None):
        save_dir = self._get_save_dir(path)

        with open(save_dir / "doc_ids.json", "r") as handle:
            self.doc_ids = json.load(handle)

        self.bm25 = bm25s.BM25.load(save_dir, load_corpus=False)
        print(f"BM25 index loaded: {len(self.doc_ids):,} docs")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        return self.retrieve_batch([query], top_k)[0]

    def retrieve_batch(self, queries: List[str], top_k: int = None) -> List[List[Tuple[str, float]]]:
        if not self.bm25:
            raise ValueError("Index not initialized. Load or build an index first.")

        top_k = top_k or BM25_TOP_K
        all_tokens = [tokenize_simple(query) for query in queries]

        results, scores = self.bm25.retrieve(
            all_tokens, k=top_k, show_progress=False, sorted=True
        )

        batch_output = []
        for query_index in range(len(queries)):
            query_results = [
                (self.doc_ids[int(idx)], float(score))
                for idx, score in zip(results[query_index], scores[query_index])
                if score > 0 and int(idx) < len(self.doc_ids)
            ]
            batch_output.append(query_results)

        return batch_output


if __name__ == "__main__":
    docs = load_corpus(max_docs=5000, verbose=True)
    retriever = BM25Retriever()
    retriever.build_index(docs)

    query_text = "how does moxibustion affect intragastric pressure"
    results = retriever.retrieve(query_text, top_k=5)

    print(f"Query: {query_text}")
    doc_map = {doc["doc_id"]: doc for doc in docs}
    for rank, (doc_id, score) in enumerate(results, 1):
        title = doc_map[doc_id]["title"][:80]
        print(f"{rank}. [{score:.2f}] {title}")
