"""AgentOps — LangGraph Orchestrator (connects all agents)"""
import json
from typing import TypedDict, List, Dict, Any
from datetime import datetime
from pathlib import Path

from langgraph.graph import StateGraph, END
from rich.console import Console

from src.agents.planner_agent import PlannerAgent
from src.agents.executor_agent import ExecutorAgent
from src.agents.reviewer_agent import ReviewerAgent

console = Console()

# ─────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────
class AgentOpsState(TypedDict):
    task: str
    plan: Dict
    execution_results: List[Dict]
    review: Dict
    retry_count: int
    max_retries: int
    final_output: str
    total_tokens: int


# ─────────────────────────────────────────
# Initialize agents (singletons)
# ─────────────────────────────────────────
planner = PlannerAgent()
executor = ExecutorAgent()
reviewer = ReviewerAgent()


# ─────────────────────────────────────────
# Nodes
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
    """Conditional edge: retry or finish?"""
    review = state.get("review", {})
    verdict = review.get("verdict", "UNKNOWN")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 1)

    if verdict == "PASS":
        console.print("[green]✓ Verdict PASS — finishing[/green]")
        return "finish"

    if retry_count >= max_retries:
        console.print(f"[yellow]⚠ Max retries reached ({max_retries}) — finishing[/yellow]")
        return "finish"

    console.print(f"[yellow]↻ Verdict {verdict} — retrying (attempt {retry_count + 1})[/yellow]")
    return "retry"


def retry_node(state: AgentOpsState) -> Dict:
    """Increment retry counter and enrich the task with feedback"""
    review = state.get("review", {})
    suggestions = review.get("suggestions", [])
    gaps = review.get("gaps", [])

    feedback = ""
    if gaps:
        feedback += "\n\nGaps to address:\n" + "\n".join(f"- {g}" for g in gaps)
    if suggestions:
        feedback += "\n\nSuggestions:\n" + "\n".join(f"- {s}" for s in suggestions)

    enriched_task = state["task"] + feedback

    console.print(f"[yellow]Enriched task with {len(gaps)} gaps + {len(suggestions)} suggestions[/yellow]")

    return {
        "task": enriched_task,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def finish_node(state: AgentOpsState) -> Dict:
    """Compile final output"""
    console.print("\n[bold cyan]═══ FINALIZING ═══[/bold cyan]")

    review = state.get("review", {})
    results = state.get("execution_results", [])
    plan = state.get("plan", {})

    # Build markdown report
    report = f"""# AgentOps Report

**Task:** {state['task']}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Plan

**Analysis:** {plan.get('task_analysis', 'N/A')}

**Complexity:** {plan.get('complexity', 'N/A')}

**Steps:**
"""
    for step in plan.get("steps", []):
        report += f"\n{step['id']}. {step['description']}"

    report += f"\n\n---\n\n## 🔧 Execution Results\n"
    for r in results:
        report += f"\n### Step {r.get('step_id')}\n"
        report += f"\n{str(r.get('output', ''))}\n"
        report += f"\n*Confidence: {r.get('confidence', 0):.2f}*\n"

    report += f"\n---\n\n## 🔍 Review\n"
    report += f"\n**Quality:** {review.get('overall_quality', 0):.2f}\n"
    report += f"\n**Verdict:** {review.get('verdict', 'N/A')}\n"

    if review.get("strengths"):
        report += "\n### Strengths\n"
        for s in review["strengths"]:
            report += f"- {s}\n"

    if review.get("weaknesses"):
        report += "\n### Weaknesses\n"
        for w in review["weaknesses"]:
            report += f"- {w}\n"

    if review.get("final_summary"):
        report += f"\n### Summary\n\n{review['final_summary']}\n"

    report += f"\n---\n\n## 📊 Metrics\n"
    report += f"\n- **Retries:** {state.get('retry_count', 0)}\n"
    report += f"- **Total tokens:** {state.get('total_tokens', 0)}\n"
    report += f"- **Steps:** {len(results)}\n"

    # Save report
    reports_dir = Path("data/experiments")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_dir / f"report_{timestamp}.md"
    report_file.write_text(report, encoding="utf-8")

    console.print(f"[green]✓ Report saved: {report_file}[/green]")

    return {"final_output": report}


# ─────────────────────────────────────────
# Build Graph
# ─────────────────────────────────────────
def build_graph():
    workflow = StateGraph(AgentOpsState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("retry", retry_node)
    workflow.add_node("finish", finish_node)

    # Entry point
    workflow.set_entry_point("planner")

    # Edges
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        decide_next,
        {
            "retry": "retry",
            "finish": "finish",
        },
    )
    workflow.add_edge("retry", "planner")
    workflow.add_edge("finish", END)

    return workflow.compile()


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────
class AgentOps:
    """Full orchestrator"""

    def __init__(self, max_retries: int = 1):
        self.max_retries = max_retries
        self.graph = build_graph()
        console.print("[green]✅ AgentOps orchestrator ready[/green]")

    def run(self, task: str) -> Dict:
        console.print(f"\n[bold magenta]🚀 AgentOps Run[/bold magenta]")
        console.print(f"[dim]Task: {task}[/dim]\n")

        initial_state = {
            "task": task,
            "plan": {},
            "execution_results": [],
            "review": {},
            "retry_count": 0,
            "max_retries": self.max_retries,
            "final_output": "",
            "total_tokens": 0,
        }

        final_state = self.graph.invoke(initial_state)

        return {
            "final_output": final_state.get("final_output", ""),
            "plan": final_state.get("plan"),
            "results": final_state.get("execution_results"),
            "review": final_state.get("review"),
            "retries": final_state.get("retry_count", 0),
            "total_tokens": final_state.get("total_tokens", 0),
        }


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — Full Pipeline Test[/bold magenta]")
    console.print("="*60)

    ops = AgentOps(max_retries=1)

    result = ops.run(
        "Research the top 3 Python web frameworks and compare them."
    )

    console.print(f"\n{'='*60}")
    console.print("[bold]Final Metrics:[/bold]")
    console.print(f"  Retries: {result['retries']}")
    console.print(f"  Total tokens: {result['total_tokens']}")
    console.print(f"  Final verdict: {result['review'].get('verdict', 'N/A')}")
    console.print(f"  Quality: {result['review'].get('overall_quality', 0):.2f}")
    console.print(f"{'='*60}\n")