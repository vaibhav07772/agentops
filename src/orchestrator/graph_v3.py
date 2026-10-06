"""AgentOps — Orchestrator v3 (with MLflow tracking)"""
import json
import time
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
from src.mlops.tracker import AgentOpsTracker

console = Console()


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


# Agents
planner = PlannerAgent()
executor = ExecutorAgent()
reviewer = ReviewerAgent()
rag_agent = RAGAgent()


# ─────────────────────────────────────
# Router
# ─────────────────────────────────────
ROUTER_PROMPT = """You are a Router in a multi-agent system.

Decide the BEST strategy for the user's task.

Options:
- "rag": Simple factual questions from knowledge base
- "multi_agent": Tasks needing planning, comparison, analysis

Return ONLY valid JSON:
{"route": "rag", "reason": "brief"}
or
{"route": "multi_agent", "reason": "brief"}"""


def router_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ ROUTER ═══[/bold cyan]")

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

    console.print(f"[yellow]→ {route}[/yellow] [dim]({reason})[/dim]")

    return {
        "route": route,
        "total_tokens": state.get("total_tokens", 0) + response["usage"]["total_tokens"],
    }


def route_decision(state: AgentOpsState) -> str:
    return state.get("route", "multi_agent")


# ─────────────────────────────────────
# RAG Path
# ─────────────────────────────────────
def rag_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ RAG ═══[/bold cyan]")
    result = rag_agent.query(state["task"])
    return {
        "rag_result": result,
        "total_tokens": state.get("total_tokens", 0) + result.get("rag_tokens", 0),
    }


def rag_finish_node(state: AgentOpsState) -> Dict:
    rag = state.get("rag_result", {})
    report = f"# RAG Report\n\n**Task:** {state['task']}\n\n**Answer:** {rag.get('answer')}\n\n**Citations:** {', '.join(rag.get('citations', []))}\n**Confidence:** {rag.get('confidence', 0):.2f}\n"
    return {"final_output": report}


# ─────────────────────────────────────
# Multi-Agent Path
# ─────────────────────────────────────
def planner_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ PLANNER ═══[/bold cyan]")
    result = planner.execute({"task": state["task"]})
    return {
        "plan": result.get("plan", {}),
        "total_tokens": state.get("total_tokens", 0) + result.get("planner_tokens", 0),
    }


def executor_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ EXECUTOR ═══[/bold cyan]")
    result = executor.execute({
        "task": state["task"],
        "plan": state["plan"],
    })
    return {
        "execution_results": result.get("execution_results", []),
        "total_tokens": state.get("total_tokens", 0) + result.get("executor_tokens", 0),
    }


def reviewer_node(state: AgentOpsState) -> Dict:
    console.print("\n[bold cyan]═══ REVIEWER ═══[/bold cyan]")
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
    review = state.get("review", {})
    results = state.get("execution_results", [])
    plan = state.get("plan", {})

    report = f"# Multi-Agent Report\n\n**Task:** {state['task']}\n\n"
    report += f"**Complexity:** {plan.get('complexity', 'N/A')}\n\n## Steps\n"
    for step in plan.get("steps", []):
        report += f"\n{step['id']}. {step['description']}"
    report += f"\n\n## Results\n"
    for r in results:
        report += f"\n### Step {r.get('step_id')}\n{r.get('output', '')}\n"
    report += f"\n## Review\n**Quality:** {review.get('overall_quality', 0):.2f}\n**Verdict:** {review.get('verdict', 'N/A')}\n"

    return {"final_output": report}


# ─────────────────────────────────────
# Build Graph
# ─────────────────────────────────────
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


# ─────────────────────────────────────
# Public API with MLflow tracking
# ─────────────────────────────────────
class AgentOpsV3:
    def __init__(self, max_retries: int = 1, enable_tracking: bool = True):
        self.max_retries = max_retries
        self.graph = build_graph()
        self.tracker = AgentOpsTracker() if enable_tracking else None
        console.print("[green]✅ AgentOps v3 ready (with MLflow)[/green]")

    def run(self, task: str) -> Dict:
        console.print(f"\n[bold magenta]🚀 AgentOps v3[/bold magenta]")
        console.print(f"[dim]Task: {task}[/dim]")

        start_time = time.time()

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

        latency = time.time() - start_time
        route = final.get("route", "unknown")
        total_tokens = final.get("total_tokens", 0)
        retries = final.get("retry_count", 0)

        # Extract quality
        quality = None
        if final.get("review"):
            quality = final["review"].get("overall_quality")
        elif final.get("rag_result"):
            quality = final["rag_result"].get("confidence")

        # Log to MLflow
        if self.tracker:
            self.tracker.log_run(
                task=task,
                route=route,
                total_tokens=total_tokens,
                latency=latency,
                quality=quality,
                retries=retries,
            )
            console.print(f"[dim]📊 Logged to MLflow[/dim]")

        return {
            "route": route,
            "final_output": final.get("final_output"),
            "total_tokens": total_tokens,
            "latency": latency,
            "quality": quality,
            "retries": retries,
        }


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps v3 — with MLflow Tracking[/bold magenta]")
    console.print("="*60)

    ops = AgentOpsV3(max_retries=1)

    tests = [
        "What is the critic pattern?",
        "Compare Django, Flask, and FastAPI.",
    ]

    for task in tests:
        console.print(f"\n{'='*60}")
        result = ops.run(task)
        console.print(f"\n[bold green]Route: {result['route']}[/bold green]")
        console.print(f"[bold]Latency: {result['latency']:.1f}s[/bold]")
        console.print(f"[bold]Tokens: {result['total_tokens']}[/bold]")
        if result['quality']:
            console.print(f"[bold]Quality: {result['quality']:.2f}[/bold]")