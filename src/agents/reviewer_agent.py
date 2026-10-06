"""AgentOps — Reviewer Agent (validates execution quality)"""
import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from rich.console import Console

console = Console()


REVIEWER_SYSTEM_PROMPT = """You are a Reviewer Agent in a multi-agent system.

Your job: Critically review the execution results against the original task.

Evaluation criteria:
1. Task Completeness — Did the steps cover all parts of the task?
2. Output Quality — Are outputs specific, accurate, and useful?
3. Consistency — Do the steps build on each other logically?
4. Gaps — What's missing or weak?

Guidelines:
- Be CONSTRUCTIVE, not harsh
- Judge against the TASK, not perfection
- 5/5 steps passed = minimum 0.6 quality
- Small omissions = minor issue, not FAIL
- Focus on whether the answer is USABLE

Output format (valid JSON only):
{
  "overall_quality": 0.75,
  "verdict": "PASS | NEEDS_REVISION | FAIL",
  "strengths": ["strength 1"],
  "weaknesses": ["weakness 1"],
  "gaps": ["gap 1"],
  "suggestions": ["improvement 1"],
  "final_summary": "Brief overall assessment (under 80 words)"
}

Verdict rules (BE CONSISTENT):
- PASS: quality >= 0.65 (task addressed, answer usable)
- NEEDS_REVISION: quality 0.40-0.65 (some gaps, still useful)
- FAIL: quality < 0.40 (fundamentally broken or missing)

CRITICAL: Same content = same score. Do NOT vary based on retry attempt."""


class ReviewerAgent(BaseAgent):
    """Reviews execution quality"""

    def __init__(self):
        super().__init__(
            name="Reviewer",
            description="Reviews and validates execution results",
            temperature=0.1,  # Very low for consistent scoring
            max_tokens=1000,
        )

    @property
    def system_prompt(self) -> str:
        return REVIEWER_SYSTEM_PROMPT

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task = state.get("task", "")
        plan = state.get("plan", {})
        exec_results = state.get("execution_results", [])

        if not exec_results:
            return {
                "error": "No execution results to review",
                "review": None,
            }

        self.log(f"Reviewing {len(exec_results)} execution results...")

        # Build a summary of execution
        results_text = "\n\n".join([
            f"Step {r.get('step_id')}:\n{str(r.get('output', ''))[:500]}"
            for r in exec_results
        ])

        # Build plan summary
        plan_text = "\n".join([
            f"  {s['id']}. {s['description']}"
            for s in plan.get("steps", [])
        ])

        # Count successful steps (confidence > 0)
        successful = sum(1 for r in exec_results if r.get("confidence", 0) > 0)
        total = len(exec_results)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Original task: {task}

Planned steps:
{plan_text}

Execution summary: {successful}/{total} steps completed successfully

Execution results:
{results_text}

Review constructively and return valid JSON.""",
            },
        ]

        try:
            response = self.call_llm(messages, json_mode=True)
            content = response["message"].content.strip()
            review = json.loads(content)

            # ─── Enforce consistency ───
            quality = review.get("overall_quality", 0.5)

            # Sanity check: if all steps passed, min quality 0.6
            if successful == total and quality < 0.6:
                quality = 0.6
                self.log(f"⚠ Adjusted quality up: {quality:.2f} (all steps passed)")

            review["overall_quality"] = quality

            # Re-derive verdict from quality to ensure consistency
            if quality >= 0.65:
                review["verdict"] = "PASS"
            elif quality >= 0.40:
                review["verdict"] = "NEEDS_REVISION"
            else:
                review["verdict"] = "FAIL"

            review["reviewer_tokens"] = response["usage"]["total_tokens"]

            verdict = review["verdict"]
            style = "green" if verdict == "PASS" else "yellow" if verdict == "NEEDS_REVISION" else "red"
            self.log(f"✅ Review complete: {verdict} (quality: {quality:.2f})", style=style)

            return {
                "review": review,
                "reviewer_tokens": review["reviewer_tokens"],
            }

        except Exception as e:
            error_msg = str(e)[:100]
            self.log(f"❌ Review failed: {error_msg}", style="red")
            return {
                "error": error_msg,
                "review": None,
            }


if __name__ == "__main__":
    from src.agents.planner_agent import PlannerAgent
    from src.agents.executor_agent import ExecutorAgent

    console.print("\n" + "="*60)
    console.print("[bold magenta]Reviewer Agent Test — Full Pipeline[/bold magenta]")
    console.print("="*60)

    task = "Research the top 3 Python web frameworks and compare them."

    # Step 1: Plan
    planner = PlannerAgent()
    plan_result = planner.execute({"task": task})

    if not plan_result.get("plan"):
        console.print("[red]❌ Planner failed[/red]")
        exit(1)

    # Step 2: Execute
    executor = ExecutorAgent()
    exec_result = executor.execute({
        "task": task,
        "plan": plan_result["plan"],
    })

    # Step 3: Review
    reviewer = ReviewerAgent()
    review_result = reviewer.execute({
        "task": task,
        "plan": plan_result["plan"],
        "execution_results": exec_result["execution_results"],
    })

    # Show review
    if review_result.get("review"):
        review = review_result["review"]
        console.print(f"\n[bold]Review Results:[/bold]")
        console.print(f"[bold]Quality:[/bold] {review.get('overall_quality', 0):.2f}")
        console.print(f"[bold]Verdict:[/bold] {review.get('verdict')}")

        if review.get("strengths"):
            console.print(f"\n[green]✅ Strengths:[/green]")
            for s in review["strengths"]:
                console.print(f"  • {s}")

        if review.get("weaknesses"):
            console.print(f"\n[yellow]⚠️ Weaknesses:[/yellow]")
            for w in review["weaknesses"]:
                console.print(f"  • {w}")

        if review.get("gaps"):
            console.print(f"\n[red]🔍 Gaps:[/red]")
            for g in review["gaps"]:
                console.print(f"  • {g}")

        console.print(f"\n[bold]Summary:[/bold] {review.get('final_summary', '')}")
    else:
        console.print(f"[red]❌ Review failed: {review_result.get('error')}[/red]")