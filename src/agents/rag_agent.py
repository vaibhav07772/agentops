"""AgentOps — RAG Agent (retrieval + generation with citations)"""
import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from src.rag.retriever import HybridRetriever
from rich.console import Console

console = Console()


RAG_SYSTEM_PROMPT = """You are a RAG Agent in a multi-agent system.

Your job: Answer questions using ONLY the provided context.

RULES (strict):
1. Use ONLY information from the context below
2. Cite sources using [Source N] format
3. If context doesn't contain the answer, set answer to "Insufficient context" and confidence to 0.0
4. Do NOT use prior knowledge
5. Be concise (under 150 words)
6. Confidence rules:
   - If answer found in context: 0.7-0.95
   - If context is weak/partial: 0.4-0.7
   - If "Insufficient context": MUST be 0.0

Output format (valid JSON only):
{
  "answer": "Your answer with citations [Source 1]...",
  "citations": ["Source 1", "Source 2"],
  "confidence": 0.85,
  "used_context": true
}"""


class RAGAgent(BaseAgent):
    """Retrieval-Augmented Generation agent"""

    def __init__(self):
        super().__init__(
            name="RAG",
            description="Answers questions using retrieved documents",
            temperature=0.2,
            max_tokens=800,
        )
        self.retriever = HybridRetriever()
        console.print("[green]✅ RAG Agent ready[/green]")

    @property
    def system_prompt(self) -> str:
        return RAG_SYSTEM_PROMPT

    def query(
        self,
        question: str,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Answer a question using RAG"""
        self.log(f"Querying: {question[:80]}...")

        # 1. Retrieve context
        docs = self.retriever.retrieve(question, top_k=top_k)

        if not docs:
            self.log("❌ No relevant documents found", style="red")
            return {
                "answer": "Insufficient context — no relevant documents found.",
                "citations": [],
                "confidence": 0.0,
                "used_context": False,
                "retrieved_docs": [],
            }

        # 2. Build context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(
                f"[Source {i}: {doc.source}]\n{doc.text}"
            )
        context = "\n\n".join(context_parts)

        # 3. Build prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""CONTEXT:
{context}

QUESTION: {question}

Answer using ONLY the context above. Return valid JSON.""",
            },
        ]

        try:
            response = self.call_llm(messages, json_mode=True)
            content = response["message"].content.strip()
            result = json.loads(content)

            # ─── Enforce confidence rule ───
            answer_lower = result.get("answer", "").lower()
            if "insufficient context" in answer_lower:
                result["confidence"] = 0.0
                result["used_context"] = False

            # Add metadata
            result["retrieved_docs"] = [
                {
                    "source": d.source,
                    "score": d.score,
                    "preview": d.text[:100],
                }
                for d in docs
            ]
            result["rag_tokens"] = response["usage"]["total_tokens"]

            self.log(
                f"✅ Answer generated (confidence: {result.get('confidence', 0):.2f})",
                style="green",
            )

            return result

        except Exception as e:
            error_msg = str(e)[:100]
            self.log(f"❌ RAG query failed: {error_msg}", style="red")
            return {
                "answer": f"Error: {error_msg}",
                "citations": [],
                "confidence": 0.0,
                "used_context": False,
                "retrieved_docs": [],
                "error": error_msg,
            }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute from state (for orchestrator compatibility)"""
        question = state.get("question") or state.get("task", "")

        if not question:
            return {"error": "No question provided"}

        result = self.query(question)
        return result


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]RAG Agent Test[/bold magenta]")
    console.print("="*60)

    rag = RAGAgent()

    test_questions = [
        "Which Python framework is best for APIs and why?",
        "What is the critic pattern in multi-agent systems?",
        "What are best practices for RAG chunking?",
        "What is the price of Bitcoin?",  # Should fail — not in KB
    ]

    for question in test_questions:
        console.print(f"\n[bold cyan]❓ {question}[/bold cyan]")
        result = rag.query(question)

        console.print(f"\n[bold green]💡 Answer:[/bold green]")
        console.print(f"{result.get('answer', 'N/A')}")

        if result.get("citations"):
            console.print(f"\n[dim]Citations: {', '.join(result['citations'])}[/dim]")

        console.print(f"[dim]Confidence: {result.get('confidence', 0):.2f}[/dim]")
        console.print(f"[dim]Used context: {result.get('used_context')}[/dim]")
        console.print("-" * 60)