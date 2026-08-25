from .config import EVAL_METRICS, EVAL_MAX_QUERIES, BM25_TOP_K, RERANKER_TOP_N

import time
import statistics
import random
from typing import List, Dict, Tuple
import ir_measures
from ir_measures import Qrel, ScoredDoc


def _to_ir_format(run: List[Dict], qrels: List[Dict]) -> Tuple[List[ScoredDoc], List[Qrel]]:
    qrels_named = [Qrel(item["query_id"], item["doc_id"], item["relevance"]) for item in qrels]
    run_named = [ScoredDoc(item["query_id"], item["doc_id"], item["score"]) for item in run]
    return run_named, qrels_named


def evaluate_retrieval(run: List[Dict], qrels: List[Dict], metrics_str: List[str] = None) -> Dict[str, float]:
    metrics_str = metrics_str or EVAL_METRICS
    metric_objects = [ir_measures.parse_measure(metric) for metric in metrics_str]

    run_named, qrels_named = _to_ir_format(run, qrels)
    results = ir_measures.calc_aggregate(metric_objects, qrels_named, run_named)
    return {str(key): round(value, 4) for key, value in results.items()}


def per_query_scores(run: List[Dict], qrels: List[Dict], metric_str: str = "nDCG@10") -> Dict[str, float]:
    metric_obj = ir_measures.parse_measure(metric_str)
    run_named, qrels_named = _to_ir_format(run, qrels)
    results = ir_measures.iter_calc([metric_obj], qrels_named, run_named)
    return {item.query_id: item.value for item in results}


def _print_metric_row(metric: str, score_a: float, score_b: float, use_percent: bool = False):
    delta = score_b - score_a
    if use_percent:
        diff = f"{(delta / score_a * 100):.1f}%" if score_a > 0 else "0.0%"
    else:
        diff = f"{delta:.4f}"

    sign = "+" if delta >= 0 else ""
    print(f"{metric} BM25={score_a:.4f} Hybrid={score_b:.4f} Delta={sign}{diff}")


def compare_pipelines(bm25_run: List[Dict], hybrid_run: List[Dict], qrels: List[Dict], label: str = ""):
    metrics = EVAL_METRICS
    bm25_scores = evaluate_retrieval(bm25_run, qrels, metrics)
    hybrid_scores = evaluate_retrieval(hybrid_run, qrels, metrics)

    if label:
        print(f"Comparison: {label}")
    print("Metric BM25 Hybrid Delta")
    for metric in metrics:
        _print_metric_row(metric, bm25_scores.get(metric, 0.0), hybrid_scores.get(metric, 0.0))

    return {"bm25": bm25_scores, "hybrid": hybrid_scores}


def rank_displacement(bm25_results, hybrid_results):
    bm25_top1 = bm25_results[0][0] if bm25_results else None
    hybrid_top1 = hybrid_results[0][0] if hybrid_results else None

    hybrid_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(hybrid_results)}
    bm25_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm25_results)}

    displacements = [
        abs(hybrid_ranks[doc_id] - bm25_ranks[doc_id])
        for doc_id, _ in bm25_results[:10]
        if doc_id in hybrid_ranks
    ]

    avg_displacement = statistics.mean(displacements) if displacements else 0
    top1_preserved = 1 if bm25_top1 == hybrid_top1 else 0
    return avg_displacement, top1_preserved


def run_full_evaluation(bm25, reranker, docs, doc_map, queries, qrels):
    report = {}

    qrel_query_ids = {item["query_id"] for item in qrels}
    eval_queries = [query for query in queries if query["query_id"] in qrel_query_ids]

    max_queries = EVAL_MAX_QUERIES
    if max_queries and len(eval_queries) > max_queries:
        random.seed(42)
        eval_queries = random.sample(eval_queries, max_queries)
        eval_query_ids = {query["query_id"] for query in eval_queries}
        qrels = [item for item in qrels if item["query_id"] in eval_query_ids]
        print(f"Sampled {max_queries} queries for evaluation.")

    print("Part 1: BM25 only")
    bm25_run, bm25_latencies = [], []

    for i, query in enumerate(eval_queries):
        start_time = time.time()
        results = bm25.retrieve(query["text"], top_k=BM25_TOP_K)
        bm25_latencies.append(time.time() - start_time)

        for doc_id, score in results:
            bm25_run.append({"query_id": query["query_id"], "doc_id": doc_id, "score": score})

        if (i + 1) % 20 == 0:
            print(f"BM25: {i + 1}/{len(eval_queries)}")

    report["bm25"] = evaluate_retrieval(bm25_run, qrels)
    avg_bm25_ms = statistics.mean(bm25_latencies) * 1000

    print("Part 2: Hybrid (BM25 + ColBERT)")
    hybrid_run, rerank_latencies, displacements = [], [], []

    for i, query in enumerate(eval_queries):
        bm25_results = bm25.retrieve(query["text"], top_k=BM25_TOP_K)
        candidates = [doc_map[doc_id] for doc_id, _ in bm25_results if doc_id in doc_map]

        if candidates:
            start_time = time.time()
            hybrid_results = reranker.rerank(query["text"], candidates, top_n=RERANKER_TOP_N)
            rerank_latencies.append(time.time() - start_time)

            for doc_id, score in hybrid_results:
                hybrid_run.append({"query_id": query["query_id"], "doc_id": doc_id, "score": score})

            avg_displacement, top1_preserved = rank_displacement(bm25_results, hybrid_results)
            displacements.append({"avg_disp": avg_displacement, "top1": top1_preserved})

        if (i + 1) % 10 == 0:
            print(f"Hybrid: {i + 1}/{len(eval_queries)}")

    report["hybrid"] = evaluate_retrieval(hybrid_run, qrels)
    avg_rerank_ms = statistics.mean(rerank_latencies) * 1000 if rerank_latencies else 0

    print("Part 3: Head-to-head")
    compare_pipelines(bm25_run, hybrid_run, qrels, label=None)

    bm25_per_query = per_query_scores(bm25_run, qrels, "nDCG@10")
    hybrid_per_query = per_query_scores(hybrid_run, qrels, "nDCG@10")

    wins = sum(
        1 for query_id, bm25_score in bm25_per_query.items()
        if hybrid_per_query.get(query_id, 0) > bm25_score + 0.001
    )
    losses = sum(
        1 for query_id, bm25_score in bm25_per_query.items()
        if hybrid_per_query.get(query_id, 0) < bm25_score - 0.001
    )
    ties = len(bm25_per_query) - wins - losses

    print("Summary")
    print(f"Dataset: {len(eval_queries)} queries, {len(qrels)} qrels")
    print("Metric BM25 Hybrid Improvement")
    for metric in EVAL_METRICS:
        _print_metric_row(metric, report["bm25"].get(metric, 0), report["hybrid"].get(metric, 0), use_percent=True)

    print(f"BM25 latency: {avg_bm25_ms:.1f}ms/q Rerank latency: {avg_rerank_ms:.1f}ms/q")
    print(f"Wins/Losses/Ties: {wins}/{losses}/{ties}")

    return report
