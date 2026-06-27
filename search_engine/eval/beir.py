"""BEIR benchmark connector for our eval harness.

Wraps SearchEngineService as a BEIR-compatible retriever. No dependency on the
beir package itself — loads BEIR-format datasets directly from JSONL/TSV files
and delegates scoring to search_engine.eval.harness.

Usage:
    dataset = BEIRDataset.load("data/nfcorpus")
    retriever = BEIRRetriever(engine, embed_fn)
    results = retriever.retrieve(dataset.queries, top_k=100)
    metrics = evaluate_beir(results, dataset.qrels)
"""

from __future__ import annotations

import csv
import json
import os
from typing import Callable, Dict, List, Tuple

from search_engine.eval.harness import ndcg_at_k, mrr, recall_at_k, precision_at_k


class BEIRDataset:
    """Minimal BEIR-format dataset (queries.jsonl, corpus.jsonl, qrels/test.tsv)."""

    def __init__(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, dict],
        qrels: Dict[str, Dict[str, int]],
    ):
        self.queries = queries   # qid  -> text
        self.corpus = corpus     # did  -> {title, text}
        self.qrels = qrels       # qid  -> {did -> relevance_grade}

    @classmethod
    def load(cls, dataset_dir: str) -> "BEIRDataset":
        queries: Dict[str, str] = {}
        corpus: Dict[str, dict] = {}
        qrels: Dict[str, Dict[str, int]] = {}

        q_path = os.path.join(dataset_dir, "queries.jsonl")
        if os.path.exists(q_path):
            with open(q_path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    queries[obj["_id"]] = obj.get("text", "")

        c_path = os.path.join(dataset_dir, "corpus.jsonl")
        if os.path.exists(c_path):
            with open(c_path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    corpus[obj["_id"]] = {
                        "title": obj.get("title", ""),
                        "text": obj.get("text", ""),
                    }

        for qrel_name in ("test.tsv", "dev.tsv", "train.tsv"):
            qrel_path = os.path.join(dataset_dir, "qrels", qrel_name)
            if os.path.exists(qrel_path):
                with open(qrel_path, encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        qid = row.get("query-id") or row.get("qid", "")
                        did = row.get("corpus-id") or row.get("doc-id", "")
                        rel = int(row.get("score") or row.get("relevance", 0))
                        qrels.setdefault(qid, {})[did] = rel
                break

        return cls(queries, corpus, qrels)


class BEIRRetriever:
    """Wraps our SearchEngineService as BEIR retriever."""

    def __init__(
        self,
        engine,
        embed_fn: Callable[[str], List[float]],
        collection_id: str = "beir",
    ):
        self.engine = engine
        self.embed_fn = embed_fn
        self.collection_id = collection_id

    def retrieve(
        self, queries: Dict[str, str], top_k: int = 100
    ) -> Dict[str, List[Tuple[str, float]]]:
        results: Dict[str, List[Tuple[str, float]]] = {}
        for qid, query_text in queries.items():
            vec = self.embed_fn(query_text)
            raw = self.engine.search(
                query=query_text, query_vector=vec,
                collection_id=self.collection_id, top_k=top_k,
            )
            results[qid] = [(r["vector_id"], r["score"]) for r in raw.get("results", [])]
        return results


def evaluate_beir(
    retrieved: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> dict:
    """Compute nDCG@k, MRR, Recall@k, Precision@k over all queries."""
    ndcgs, mrrs, recalls, precisions = [], [], [], []
    for qid, ranked in retrieved.items():
        relevant = {did for did, rel in qrels.get(qid, {}).items() if rel > 0}
        if not relevant:
            continue
        doc_ids = [did for did, _ in ranked]
        ndcgs.append(ndcg_at_k(doc_ids, relevant, k))
        mrrs.append(mrr(doc_ids, relevant))
        recalls.append(recall_at_k(doc_ids, relevant, k))
        precisions.append(precision_at_k(doc_ids, relevant, k))
    n = max(len(ndcgs), 1)
    return {
        "ndcg_at_k": round(sum(ndcgs) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "recall_at_k": round(sum(recalls) / n, 4),
        "precision_at_k": round(sum(precisions) / n, 4),
        "n_queries": n,
        "k": k,
    }
