"""AgentOps — Hybrid Retriever (BM25 + Semantic + RRF)"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import logging
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass

logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from rich.console import Console

console = Console()

CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "agentops_knowledge"


@dataclass
class RetrievedDoc:
    """Retrieved document with score"""
    text: str
    source: str
    score: float
    rank: int
    method: str  # bm25 | semantic | hybrid


class HybridRetriever:
    """Combines BM25 and semantic search with RRF fusion"""

    def __init__(self):
        console.print("[cyan]🔧 Initializing Hybrid Retriever...[/cyan]")

        # ChromaDB
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
        )

        # Load all docs for BM25
        self._load_docs_for_bm25()

        console.print(f"[green]✅ Hybrid Retriever ready ({len(self.all_docs)} docs)[/green]")

    def _load_docs_for_bm25(self):
        """Load all docs into BM25 index"""
        results = self.collection.get()

        self.all_docs = results["documents"]
        self.all_metadata = results["metadatas"]
        self.all_ids = results["ids"]

        # Tokenize for BM25 (simple word-level)
        tokenized = [doc.lower().split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(tokenized)

    # ─────────────────────────────
    # Semantic Search
    # ─────────────────────────────
    def _semantic_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        results = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, len(self.all_docs)),
        )

        docs = []
        for rank, (doc, meta, doc_id) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["ids"][0],
        )):
            docs.append(RetrievedDoc(
                text=doc,
                source=meta.get("source", "unknown"),
                score=1.0 / (rank + 1),  # Placeholder
                rank=rank,
                method="semantic",
            ))
        return docs

    # ─────────────────────────────
    # BM25 Search
    # ─────────────────────────────
    def _bm25_search(self, query: str, top_k: int = 10) -> List[RetrievedDoc]:
        if not self.all_docs:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Top-K indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        docs = []
        for rank, idx in enumerate(top_indices):
            docs.append(RetrievedDoc(
                text=self.all_docs[idx],
                source=self.all_metadata[idx].get("source", "unknown"),
                score=float(scores[idx]),
                rank=rank,
                method="bm25",
            ))
        return docs

    # ─────────────────────────────
    # RRF Fusion
    # ─────────────────────────────
    def _rrf_fusion(
        self,
        semantic_docs: List[RetrievedDoc],
        bm25_docs: List[RetrievedDoc],
        k: int = 60,
    ) -> List[RetrievedDoc]:
        """Combine ranked lists with Reciprocal Rank Fusion"""
        scores = {}
        doc_lookup = {}

        # Semantic contributions
        for doc in semantic_docs:
            key = doc.text[:100]
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + doc.rank + 1)

        # BM25 contributions
        for doc in bm25_docs:
            key = doc.text[:100]
            if key not in doc_lookup:
                doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1 / (k + doc.rank + 1)

        # Sort by fused score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for rank, (key, score) in enumerate(ranked):
            doc = doc_lookup[key]
            fused.append(RetrievedDoc(
                text=doc.text,
                source=doc.source,
                score=score,
                rank=rank,
                method="hybrid",
            ))
        return fused

    # ─────────────────────────────
    # Main retrieve
    # ─────────────────────────────
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedDoc]:
        """Hybrid retrieval: BM25 + Semantic + RRF"""
        # 1. Semantic
        semantic = self._semantic_search(query, top_k=top_k * 2)

        # 2. BM25
        bm25 = self._bm25_search(query, top_k=top_k * 2)

        # 3. RRF Fusion
        fused = self._rrf_fusion(semantic, bm25)

        # 4. Top-K
        return fused[:top_k]

    # ─────────────────────────────
    # Format for LLM
    # ─────────────────────────────
    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Retrieve and format as context string for LLM"""
        docs = self.retrieve(query, top_k=top_k)

        if not docs:
            return "No relevant context found."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}: {doc.source}]\n{doc.text}"
            )

        return "\n\n".join(context_parts)


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — Hybrid Retriever Test[/bold magenta]")
    console.print("="*60)

    retriever = HybridRetriever()

    # Test queries
    test_queries = [
        "Which Python framework is best for APIs?",
        "How does RAG work?",
        "What is the critic pattern in multi-agent systems?",
    ]

    for query in test_queries:
        console.print(f"\n[bold cyan]🔍 Query:[/bold cyan] {query}")
        docs = retriever.retrieve(query, top_k=3)
        for i, doc in enumerate(docs, 1):
            console.print(f"\n  [{i}] [dim]{doc.source}[/dim] (score: {doc.score:.4f}, method: {doc.method})")
            console.print(f"      {doc.text[:150]}...")