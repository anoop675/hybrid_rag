from .config import RAG_MODEL, DEVICE, RAG_NUM_EVIDENCE_DOCS, RAG_MAX_NEW_TOKENS, RAG_MAX_CONTEXT_TOKENS

import time
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class RAGModule:
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or RAG_MODEL
        self.device = device or DEVICE
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"Loading RAG model: {self.model_name} (4-bit quantized)...")
        start_time = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"RAG model loaded ({time.time() - start_time:.1f}s)")

    def _build_messages(self, query: str, evidence_docs: List[Dict]) -> List[Dict]:
        chars_per_doc = 1200
        evidence_parts = []
        for i, doc in enumerate(evidence_docs[:RAG_NUM_EVIDENCE_DOCS], 1):
            body = doc["body"][:chars_per_doc].strip()
            evidence_parts.append(f"[{i}] {doc['title']}\n{body}")

        context = "\n\n".join(evidence_parts)

        system_message = (
            "You are a helpful research assistant. Answer the question using ONLY the provided passages.\n"
            "RULES:\n"
            "- You MUST answer if relevant information is present in the passages\n"
            "- Every statement MUST be supported by a citation [n]\n"
            "- ONLY answer using the in the passages\n"
            "Be concise, factual, and grounded strictly in the evidence."
        )
        user_message = (
            f"Here are the retrieved passages:\n\n"
            f"{context}\n\n"
            f"Question: {query}"
        )

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def generate(self, query: str, evidence_docs: List[Dict], max_new_tokens: int = None) -> Dict:
        if max_new_tokens is None:
            base_tokens = 150
            per_doc_tokens = 50
            max_new_tokens = min(
                base_tokens + per_doc_tokens * len(evidence_docs),
                RAG_MAX_NEW_TOKENS,
            )

        if self.model is None:
            self.load_model()

        messages = self._build_messages(query, evidence_docs)

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=RAG_MAX_CONTEXT_TOKENS,
            truncation=True,
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        start_time = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.1,
                top_p=0.95,
                no_repeat_ngram_size=3,
            )

        answer = self.tokenizer.decode(
            output_ids[0][input_length:], skip_special_tokens=True
        ).strip()
        generation_time = time.time() - start_time

        sources = [
            {"citation": f"[{i}]", "doc_id": doc["doc_id"], "title": doc["title"]}
            for i, doc in enumerate(evidence_docs, 1)
        ]

        return {
            "answer": answer,
            "sources": sources,
            "generation_time": generation_time,
            "prompt": prompt,
        }
