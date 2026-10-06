"""AgentOps — Executor Agent (executes plan steps)"""
import json
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent
from rich.console import Console

console = Console()


EXECUTOR_SYSTEM_PROMPT = """You are an Executor Agent in a multi-agent system.

Your job: Execute ONE specific step from a larger plan.

Guidelines:
- Focus ONLY on the assigned step
- BE CONCISE — keep output under 120 words
- Use bullet points if listing items
- Do NOT repeat the task context
- Do NOT add preamble or conclusion

Output format (valid JSON only):
{
  "step_id": 1,
  "output": "Concise result (under 120 words)",
  "confidence": 0.85,
  "notes": "Brief caveat if any"
}"""


class ExecutorAgent(BaseAgent):
    """Executes individual plan steps"""

    def __init__(self):
        super().__init__(
            name="Executor",
            description="Executes individual steps from a plan",
            temperature=0.4,
            max_tokens=1500,  # ← Increased for JSON completion
        )

    @property
    def system_prompt(self) -> str:
        return EXECUTOR_SYSTEM_PROMPT

    def execute_step(
        self,
        step: Dict[str, Any],
        task: str,
        previous_results: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute a single step"""
        step_id = step.get("id", "?")
        description = step.get("description", "")
        expected = step.get("expected_output", "")

        self.log(f"Executing step {step_id}: {description[:60]}...")

        # Build context (limited to 2 previous steps to save tokens)
        context = ""
        if previous_results:
            recent = previous_results[-2:] if len(previous_results) > 2 else previous_results
            prev_summary = "\n".join([
                f"Step {r.get('step_id')}: {str(r.get('output', ''))[:150]}"
                for r in recent
            ])
            context = f"\n\nPrevious results (for context):\n{prev_summary}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Main task: {task}

Step #{step_id} to execute:
{description}

Expected output:
{expected}
{context}

Execute concisely and return valid JSON.""",
            },
        ]

        try:
            response = self.call_llm(messages, json_mode=True)
            content = response["message"].content.strip()
            result = json.loads(content)

            result["step_id"] = step_id
            result["executor_tokens"] = response["usage"]["total_tokens"]

            self.log(f"✅ Step {step_id} complete", style="green")

            return result

        except Exception as e:
            error_msg = str(e)[:100]
            self.log(f"❌ Step {step_id} failed: {error_msg}", style="red")
            return {
                "step_id": step_id,
                "output": f"[FAILED] {error_msg}",
                "confidence": 0.0,
                "error": error_msg,
            }

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute all plan steps"""
        plan = state.get("plan", {})
        task = state.get("task", "")

        if not plan or not plan.get("steps"):
            return {"error": "No plan provided", "execution_results": []}

        steps = plan["steps"]
        self.log(f"Executing {len(steps)} steps...")

        results = []
        for step in steps:
            result = self.execute_step(step, task, previous_results=results)
            results.append(result)

        total_tokens = sum(r.get("executor_tokens", 0) for r in results)
        successful = sum(1 for r in results if r.get("confidence", 0) > 0)

        self.log(
            f"✅ Complete: {successful}/{len(steps)} steps passed",
            style="green" if successful == len(steps) else "yellow",
        )

        return {
            "execution_results": results,
            "executor_tokens": total_tokens,
            "steps_successful": successful,
            "steps_total": len(steps),
        }


if __name__ == "__main__":
    from src.agents.planner_agent import PlannerAgent

    console.print("\n" + "="*60)
    console.print("[bold magenta]Executor Agent Test — Full Pipeline[/bold magenta]")
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

    # Show results
    console.print(f"\n[bold]Execution Results:[/bold]")
    for r in exec_result.get("execution_results", []):
        console.print(f"\n[cyan]Step {r.get('step_id')}:[/cyan]")
        output = str(r.get("output", ""))[:300]
        console.print(f"  {output}...")
        console.print(f"  [dim]Confidence: {r.get('confidence', 0):.2f}[/dim]")

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Steps passed: {exec_result.get('steps_successful')}/{exec_result.get('steps_total')}")
    console.print(f"  Total tokens: {exec_result.get('executor_tokens')}")