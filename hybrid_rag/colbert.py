from .config import (
    RERANKER_MODEL,
    DEVICE,
    RERANKER_MAX_LENGTH,
    RERANKER_BATCH_SIZE,
    COLBERT_EMB_PATH,
    RERANKER_TOP_N,
)

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer, AutoModel


class ColBERTReranker:
    """ColBERT re-ranker using late-interaction (MaxSim) scoring."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or RERANKER_MODEL
        self.device = device or DEVICE
        self.tokenizer = None
        self.model = None
        self.linear = None
        self.doc_embeddings: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None

    def load_model(self):
        print(f"Loading re-ranker model: {self.model_name}...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        try:
            from huggingface_hub import hf_hub_download
            import safetensors.torch
            weights_path = hf_hub_download(self.model_name, "model.safetensors")
            state = safetensors.torch.load_file(weights_path)
            if "linear.weight" in state:
                hidden_size = self.model.config.hidden_size
                projection_dim = state["linear.weight"].shape[0]
                self.linear = nn.Linear(hidden_size, projection_dim, bias=False)
                self.linear.weight.data = state["linear.weight"]
                self.linear.to(self.device)
                self.linear.eval()
                print(f"Loaded ColBERT linear projection ({hidden_size} -> {projection_dim})")
        except Exception as exc:
            print(f"Note: no linear projection loaded ({exc})")

        self.model.to(self.device)
        self.model.eval()

        print(f"Re-ranker loaded on {self.device} ({time.time() - start_time:.1f}s)")

    def _encode(self, texts: List[str], max_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = max_length or RERANKER_MAX_LENGTH

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state

        if self.linear is not None:
            embeddings = self.linear(embeddings)

        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings, inputs["attention_mask"]

    def _maxsim_score(
        self,
        query_emb: torch.Tensor,
        query_mask: torch.Tensor,
        doc_emb: torch.Tensor,
        doc_mask: torch.Tensor,
    ) -> torch.Tensor:
        sim_matrix = torch.matmul(query_emb, doc_emb.T)
        doc_mask_exp = doc_mask.unsqueeze(0).float()
        sim_matrix = sim_matrix * doc_mask_exp + (1 - doc_mask_exp) * (-1e9)
        max_sims = sim_matrix.max(dim=-1).values
        max_sims = max_sims * query_mask.float()
        return max_sims.sum()

    def encode_corpus(
        self,
        docs: List[Dict],
        batch_size: int = None,
        shard_size: int = 10_000,
        shard_folder: str = None,
    ) -> Dict:
        import gc

        batch_size = batch_size or RERANKER_BATCH_SIZE
        shard_folder = shard_folder or os.path.dirname(COLBERT_EMB_PATH)
        os.makedirs(shard_folder, exist_ok=True)

        if self.model is None:
            self.load_model()

        print(f"Encoding {len(docs):,} documents with ColBERT (batch={batch_size}, shard={shard_size:,})...")
        start_time = time.time()
        shard_files = []
        shard_buffer = {}
        shard_index = 0

        for start in range(0, len(docs), batch_size):
            batch = docs[start:start + batch_size]
            texts = [f"{doc['title']}. {doc['body']}" for doc in batch]

            with torch.no_grad():
                doc_emb_batch, doc_mask_batch = self._encode(texts)

            for j, doc in enumerate(batch):
                shard_buffer[doc["doc_id"]] = (
                    doc_emb_batch[j].cpu(),
                    doc_mask_batch[j].cpu(),
                )

            del doc_emb_batch, doc_mask_batch
            torch.cuda.empty_cache()

            done = min(start + batch_size, len(docs))

            if len(shard_buffer) >= shard_size or done >= len(docs):
                shard_path = os.path.join(shard_folder, f"_emb_shard_{shard_index:04d}.pt")
                torch.save(shard_buffer, shard_path)
                shard_files.append(shard_path)
                print(
                    f"{done}/{len(docs):,} docs, shard {shard_index} saved "
                    f"({len(shard_buffer):,} docs, {time.time() - start_time:.0f}s)"
                )
                shard_buffer = {}
                shard_index += 1
                gc.collect()

        print(f"Merging {len(shard_files)} shard(s)...")
        all_embeddings = {}
        for shard_path in shard_files:
            all_embeddings.update(torch.load(shard_path, map_location="cpu", weights_only=True))
            os.remove(shard_path)

        print(f"Corpus encoded: {len(all_embeddings):,} docs in {time.time() - start_time:.1f}s")
        self.doc_embeddings = all_embeddings
        return all_embeddings

    def save_embeddings(self, path: str = None):
        path = path or COLBERT_EMB_PATH
        if self.doc_embeddings is None:
            raise ValueError("No embeddings to save. Run encode_corpus() first.")
        torch.save(self.doc_embeddings, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"ColBERT embeddings saved: {len(self.doc_embeddings):,} docs -> {path} ({size_mb:.1f} MB)")

    def load_embeddings(self, path: str = None):
        path = path or COLBERT_EMB_PATH
        self.doc_embeddings = torch.load(path, map_location="cpu", weights_only=True)
        print(f"ColBERT embeddings loaded: {len(self.doc_embeddings):,} docs from {path}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_n: int = None,
        batch_size: int = None,
    ) -> List[Tuple[str, float]]:
        top_n = top_n or RERANKER_TOP_N
        batch_size = batch_size or RERANKER_BATCH_SIZE

        if self.model is None:
            self.load_model()

        with torch.no_grad():
            query_emb, query_mask = self._encode([query])
        query_emb = query_emb[0]
        query_mask = query_mask[0]

        scores = []

        if self.doc_embeddings is not None:
            for candidate in candidates:
                doc_id = candidate["doc_id"]
                if doc_id in self.doc_embeddings:
                    doc_emb, doc_mask = self.doc_embeddings[doc_id]
                    doc_emb = doc_emb.to(self.device)
                    doc_mask = doc_mask.to(self.device)
                    score = self._maxsim_score(query_emb, query_mask, doc_emb, doc_mask).item()
                    scores.append((doc_id, score))
        else:
            doc_texts = [f"{candidate['title']}. {candidate['body']}" for candidate in candidates]
            for start in range(0, len(doc_texts), batch_size):
                batch_texts = doc_texts[start:start + batch_size]
                with torch.no_grad():
                    doc_emb_batch, doc_mask_batch = self._encode(batch_texts)

                for j in range(doc_emb_batch.size(0)):
                    score = self._maxsim_score(
                        query_emb, query_mask, doc_emb_batch[j], doc_mask_batch[j]
                    ).item()
                    scores.append((candidates[start + j]["doc_id"], score))

        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:top_n]


if __name__ == "__main__":
    reranker = ColBERTReranker()
    reranker.load_model()

    query = "How do stars emit radiation?"
    candidates = [
        {
            "doc_id": "D0",
            "title": "Stellar Radiation",
            "body": "Stars emit energy as electromagnetic radiation from their hot surfaces.",
        },
        {
            "doc_id": "D1",
            "title": "Fishing Guide",
            "body": "How to set up a fishing rod and choose the right bait for river fishing.",
        },
        {
            "doc_id": "D2",
            "title": "Stefan-Boltzmann Law",
            "body": "The Stefan-Boltzmann law states energy flux by radiation is proportional to the fourth power of temperature.",
        },
    ]

    results = reranker.rerank(query, candidates, top_n=3)
    print(f"Query: {query}")
    for rank, (doc_id, score) in enumerate(results, 1):
        candidate = next(item for item in candidates if item["doc_id"] == doc_id)
        print(f"{rank}. [{score:.4f}] {candidate['title']}")
