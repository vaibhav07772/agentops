"""AgentOps — Orchestrator v2 (with RAG routing)"""
import json
from typing import TypedDict, List, Dict, Any
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console

from src.agents.planner_agent import PlannerAgent
from src.agents.executor_agent import ExecutorAgent
from src.agents.reviewer_agent import ReviewerAgent
from src.agents.rag_agent import RAGAgent
from src.core.llm_client import call_llm

console = Console()


# ─────────────────────────────────────────
# State
# ─────────────────────────────────────────
class AgentOpsState(TypedDict):
    task: str
    route: str
    rag_result: Dict
    plan: Dict
    execution_results: List[Dict]
    review: Dict
    retry_count: int
    max_retries: int
    final_output: str
    total_tokens: int


# ─────────────────────────────────────────
# Agents (singletons)
# ─────────────────────────────────────────
planner = PlannerAgent()
executor = ExecutorAgent()
reviewer = ReviewerAgent()
rag_agent = RAGAgent()


# ─────────────────────────────────────────
# Router
# ─────────────────────────────────────────
ROUTER_PROMPT = """You are a Router in a multi-agent system.

Decide the BEST strategy for the user's task.

Options:
- "rag": For simple factual questions answerable from a knowledge base
        (definitions, concepts, specific info, short Q&A)
- "multi_agent": For tasks requiring planning, comparison, research,
                analysis, multi-step reasoning, or long outputs

Return ONLY valid JSON:
{"route": "rag", "reason": "brief explanation"}

or

{"route": "multi_agent", "reason": "brief explanation"}"""


def router_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ ROUTER NODE ═══[/bold cyan]")

    response = call_llm(
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": f"Task: {state['task']}"},
        ],
        json_mode=True,
        max_tokens=200,
    )

    try:
        result = json.loads(response["message"].content)
        route = result.get("route", "multi_agent")
        reason = result.get("reason", "")
    except Exception:
        route = "multi_agent"
        reason = "Fallback"

    console.print(f"[yellow]→ Route: {route}[/yellow]")
    console.print(f"[dim]  Reason: {reason}[/dim]")

    return {
        "route": route,
        "total_tokens": state.get("total_tokens", 0) + response["usage"]["total_tokens"],
    }


def route_decision(state: AgentOpsState) -> str:
    return state.get("route", "multi_agent")


# ─────────────────────────────────────────
# RAG Path
# ─────────────────────────────────────────
def rag_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ RAG NODE ═══[/bold cyan]")
    result = rag_agent.query(state["task"])
    return {
        "rag_result": result,
        "total_tokens": state.get("total_tokens", 0) + result.get("rag_tokens", 0),
    }


def rag_finish_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ FINALIZING (RAG) ═══[/bold cyan]")

    rag = state.get("rag_result", {})

    report = f"""# AgentOps Report — RAG Route

**Task:** {state['task']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Route:** RAG

---

## 💡 Answer

{rag.get('answer', 'N/A')}

**Citations:** {', '.join(rag.get('citations', []))}
**Confidence:** {rag.get('confidence', 0):.2f}

---

## 📚 Retrieved Sources
"""
    for i, doc in enumerate(rag.get("retrieved_docs", []), 1):
        report += f"\n**{i}. {doc.get('source', 'unknown')}** (score: {doc.get('score', 0):.4f})\n"
        report += f"_{doc.get('preview', '')}..._\n"

    report += f"\n---\n\n## 📊 Metrics\n- **Total tokens:** {state.get('total_tokens', 0)}\n"

    reports_dir = Path("data/experiments")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"rag_report_{timestamp}.md"
    report_file.write_text(report, encoding="utf-8")

    console.print(f"[green]✓ Report: {report_file.name}[/green]")

    return {"final_output": report}


# ─────────────────────────────────────────
# Multi-Agent Path
# ─────────────────────────────────────────
def planner_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ PLANNER NODE ═══[/bold cyan]")
    result = planner.execute({"task": state["task"]})
    return {
        "plan": result.get("plan", {}),
        "total_tokens": state.get("total_tokens", 0) + result.get("planner_tokens", 0),
    }


def executor_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ EXECUTOR NODE ═══[/bold cyan]")
    result = executor.execute({
        "task": state["task"],
        "plan": state["plan"],
    })
    return {
        "execution_results": result.get("execution_results", []),
        "total_tokens": state.get("total_tokens", 0) + result.get("executor_tokens", 0),
    }


def reviewer_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ REVIEWER NODE ═══[/bold cyan]")
    result = reviewer.execute({
        "task": state["task"],
        "plan": state["plan"],
        "execution_results": state["execution_results"],
    })
    return {
        "review": result.get("review", {}),
        "total_tokens": state.get("total_tokens", 0) + result.get("reviewer_tokens", 0),
    }


def decide_next(state: AgentOpsState) -> str:
    review = state.get("review", {})
    verdict = review.get("verdict", "UNKNOWN")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if verdict == "PASS":
        return "finish"
    if retry_count >= max_retries:
        return "finish"
    return "retry"


def retry_node(state: AgentOpsState) -> Dict:
    review = state.get("review", {})
    gaps = review.get("gaps", [])
    suggestions = review.get("suggestions", [])

    feedback = ""
    if gaps:
        feedback += "\n\nGaps:\n" + "\n".join(f"- {g}" for g in gaps)
    if suggestions:
        feedback += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)

    return {
        "task": state["task"] + feedback,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def multi_finish_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ FINALIZING (Multi-Agent) ═══[/bold cyan]")

    review = state.get("review", {})
    results = state.get("execution_results", [])
    plan = state.get("plan", {})

    report = f"""# AgentOps Report — Multi-Agent Route

**Task:** {state['task']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Route:** Multi-Agent

---

## 📋 Plan
**Complexity:** {plan.get('complexity', 'N/A')}

**Steps:**
"""
    for step in plan.get("steps", []):
        report += f"\n{step['id']}. {step['description']}"

    report += "\n\n---\n\n## 🔧 Execution Results\n"
    for r in results:
        report += f"\n### Step {r.get('step_id')}\n{r.get('output', '')}\n"

    report += f"\n---\n\n## 🔍 Review\n"
    report += f"**Quality:** {review.get('overall_quality', 0):.2f}\n"
    report += f"**Verdict:** {review.get('verdict', 'N/A')}\n"

    if review.get("final_summary"):
        report += f"\n**Summary:** {review['final_summary']}\n"

    report += f"\n---\n\n## 📊 Metrics\n"
    report += f"- **Retries:** {state.get('retry_count', 0)}\n"
    report += f"- **Total tokens:** {state.get('total_tokens', 0)}\n"

    reports_dir = Path("data/experiments")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"multi_report_{timestamp}.md"
    report_file.write_text(report, encoding="utf-8")

    console.print(f"[green]✓ Report: {report_file.name}[/green]")

    return {"final_output": report}


# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentOpsState)

    workflow.add_node("router", router_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("rag_finish", rag_finish_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("multi_finish", multi_finish_node)

    workflow.set_entry_point("router")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {"rag": "rag", "multi_agent": "planner"},
    )

    workflow.add_edge("rag", "rag_finish")
    workflow.add_edge("rag_finish", END)

    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        decide_next,
        {"retry": "retry", "finish": "multi_finish"},
    )
    workflow.add_edge("retry", "planner")
    workflow.add_edge("multi_finish", END)

    return workflow.compile()


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────
class AgentOpsV2:
    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self.graph = build_graph()
        console.print("[green]✅ AgentOps v2 ready (RAG + Multi-Agent routing)[/green]")

    def run(self, task: str) -> Dict:
        console.print(f"\n[bold magenta]🚀 AgentOps v2[/bold magenta]")
        console.print(f"[dim]Task: {task}[/dim]")

        initial_state = {
            "task": task,
            "route": "",
            "rag_result": {},
            "plan": {},
            "execution_results": [],
            "review": {},
            "retry_count": 0,
            "max_retries": self.max_retries,
            "final_output": "",
            "total_tokens": 0,
        }

        final = self.graph.invoke(initial_state)

        return {
            "route": final.get("route"),
            "final_output": final.get("final_output"),
            "total_tokens": final.get("total_tokens"),
            "rag_result": final.get("rag_result"),
            "review": final.get("review"),
        }


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps v2 — Smart Routing Test[/bold magenta]")
    console.print("="*60)

    ops = AgentOpsV2(max_retries=1)

    tests = [
        "What is the critic pattern?",                       # → RAG
        "Research and compare Django, Flask, and FastAPI in detail.",  # → Multi-agent
    ]

    for task in tests:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Testing: {task}[/bold]")
        console.print(f"{'='*60}")

        result = ops.run(task)

        console.print(f"\n[bold green]Route: {result['route']}[/bold green]")
        console.print(f"[bold]Total tokens: {result['total_tokens']}[/bold]")