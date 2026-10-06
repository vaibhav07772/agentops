"""AgentOps — RAG Ingestion (load documents into ChromaDB)"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Silence telemetry
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("posthog").setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from rich.console import Console
from rich.progress import track

console = Console()

# ─────────────────────────────────────
# Config
# ─────────────────────────────────────
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "agentops_knowledge"
DOCS_DIR = "data/documents"

Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)


class RAGIngestor:
    """Ingests documents into ChromaDB"""

    def __init__(self):
        console.print("[cyan]📚 Initializing RAG Ingestor...[/cyan]")

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )

        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Delete + recreate collection
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        console.print("[green]✅ Ingestor ready[/green]")

    def _hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
    ) -> List[str]:
        """Chunk text with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        return chunks

    def ingest_text(
        self,
        text: str,
        source: str,
        metadata: Dict = None,
    ) -> int:
        """Ingest a single text document"""
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        ids = [self._hash(c + source) for c in chunks]
        metadatas = []
        for i, _ in enumerate(chunks):
            meta = {
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "ingested_at": datetime.now().isoformat(),
            }
            if metadata:
                meta.update(metadata)
            metadatas.append(meta)

        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids,
        )

        return len(chunks)

    def ingest_directory(self, directory: str = DOCS_DIR):
        """Ingest all .txt and .md files from a directory"""
        dir_path = Path(directory)

        if not dir_path.exists():
            console.print(f"[yellow]⚠ Directory missing: {directory}[/yellow]")
            return

        files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))

        if not files:
            console.print(f"[yellow]⚠ No files in {directory}[/yellow]")
            return

        console.print(f"\n[cyan]📥 Ingesting {len(files)} files...[/cyan]")

        total_chunks = 0
        for file_path in track(files, description="Processing"):
            try:
                text = file_path.read_text(encoding="utf-8")
                count = self.ingest_text(
                    text=text,
                    source=file_path.name,
                )
                total_chunks += count
            except Exception as e:
                console.print(f"[red]❌ {file_path.name}: {e}[/red]")

        console.print(f"\n[green]✅ Ingested {total_chunks} chunks from {len(files)} files[/green]")

    def stats(self) -> Dict:
        """Collection stats"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": COLLECTION_NAME,
        }


# ─────────────────────────────────────────
# Create sample knowledge base
# ─────────────────────────────────────────
def create_sample_knowledge():
    """Create sample documents for testing"""
    docs_dir = Path(DOCS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "python_frameworks.txt": """Python Web Frameworks Overview

Django (2005): A high-level, full-stack web framework that encourages rapid development
and clean, pragmatic design. Includes ORM, admin panel, authentication, and templating
out of the box. Best for large-scale applications requiring a batteries-included approach.

Flask (2010): A lightweight WSGI web application framework. Designed to make getting
started quick and easy, with the ability to scale up to complex applications. It's a
micro-framework with a small core and many extensions. Ideal for microservices and APIs.

FastAPI (2018): A modern, fast web framework for building APIs with Python type hints.
Based on Starlette (ASGI) and Pydantic. Features automatic OpenAPI documentation,
async support, and very high performance. Great for modern APIs and ML services.

Performance: FastAPI > Flask > Django
Learning curve: Flask < FastAPI < Django
Community: Django > Flask > FastAPI (but FastAPI growing fastest)
Ecosystem: Django > Flask ≈ FastAPI
""",
        "rag_systems.txt": """Retrieval-Augmented Generation (RAG)

RAG combines retrieval systems with LLM generation. Key components:
1. Document chunking — split documents into semantic chunks
2. Embeddings — convert chunks to vectors using models like MiniLM
3. Vector store — ChromaDB, Qdrant, Pinecone
4. Retrieval — semantic search finds relevant chunks
5. Reranking — Cross-Encoder reorders results for precision
6. Generation — LLM produces answer from retrieved context

Best practices:
- Chunk size 300-800 tokens with 10-20% overlap
- Hybrid retrieval (BM25 + semantic) improves precision
- Cross-Encoder reranking boosts top-5 accuracy
- Always validate citations against retrieved context

Common pitfalls:
- Chunking too large loses context
- Pure semantic search fails on exact matches (IDs, numbers)
- No reranking = mediocre top results
""",
        "llm_agents.txt": """LLM Multi-Agent Systems

Multi-agent patterns:
1. Planner-Executor — one agent plans, another executes
2. Critic pattern — one agent critiques another's output
3. Debate — multiple agents argue different views
4. Hierarchical — supervisor delegates to workers

Key considerations:
- Each agent needs clear system prompt and role
- State management via LangGraph
- Tool calling enables real-world actions
- Retry loops with feedback for quality
- Token budgets per agent

Production challenges:
- LLM inconsistency — same input, different output
- Token costs add up with multi-turn loops
- Reviewers can be too harsh or too lenient
- Best-of-N sampling helps with variance
""",
    }

    for filename, content in samples.items():
        file_path = docs_dir / filename
        file_path.write_text(content, encoding="utf-8")

    console.print(f"[green]✅ Created {len(samples)} sample documents in {docs_dir}[/green]")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — RAG Ingestion[/bold magenta]")
    console.print("="*60)

    # 1. Create samples if no docs
    docs_path = Path(DOCS_DIR)
    if not any(docs_path.glob("*.txt")):
        create_sample_knowledge()

    # 2. Ingest
    ingestor = RAGIngestor()
    ingestor.ingest_directory()

    # 3. Stats
    stats = ingestor.stats()
    console.print(f"\n[bold]Collection Stats:[/bold]")
    console.print(f"  Total chunks: {stats['total_documents']}")

    # 4. Test query
    console.print(f"\n[bold cyan]🔍 Test Query:[/bold cyan]")
    query = "Which Python framework is best for APIs?"
    results = ingestor.collection.query(
        query_texts=[query],
        n_results=2,
    )

    console.print(f"\nQuery: {query}")
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        console.print(f"\n[bold]{i}. {meta['source']}[/bold]")
        console.print(f"   {doc[:200]}...")