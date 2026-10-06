"""AgentOps — Planner Agent (breaks down tasks)"""
import json
from typing import Dict, Any
from src.agents.base_agent import BaseAgent
from rich.console import Console

console = Console()


PLANNER_SYSTEM_PROMPT = """You are a Planner Agent in a multi-agent system.

Your job: Break down a complex user task into 2-5 clear, actionable steps.

Rules:
- Each step should be atomic (one clear action)
- Steps should be ordered logically
- Return valid JSON only — no markdown, no explanations

Output format:
{
  "task_analysis": "Brief analysis of what the user wants",
  "steps": [
    {
      "id": 1,
      "description": "Clear action to take",
      "expected_output": "What should be produced"
    }
  ],
  "complexity": "low | medium | high"
}"""


class PlannerAgent(BaseAgent):
    """Breaks down tasks into steps"""

    def __init__(self):
        super().__init__(
            name="Planner",
            description="Breaks complex tasks into actionable steps",
            temperature=0.3,
        )

    @property
    def system_prompt(self) -> str:
        return PLANNER_SYSTEM_PROMPT

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task = state.get("task", "")

        if not task:
            return {"error": "No task provided", "plan": None}

        self.log(f"Planning task: {task[:80]}...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"},
        ]

        try:
            response = self.call_llm(messages, json_mode=True)
            content = response["message"].content.strip()

            plan = json.loads(content)

            self.log(
                f"✅ Plan created: {len(plan.get('steps', []))} steps",
                style="green",
            )

            return {
                "plan": plan,
                "plan_raw": content,
                "planner_tokens": response["usage"]["total_tokens"],
            }

        except Exception as e:
            self.log(f"❌ Planning failed: {e}", style="red")
            return {"error": str(e), "plan": None}


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]Planner Agent Test[/bold magenta]")
    console.print("="*60)

    planner = PlannerAgent()

    state = {
        "task": "Research the top 3 Python web frameworks and compare their performance, learning curve, and community support.",
    }

    result = planner.execute(state)

    if result.get("plan"):
        plan = result["plan"]
        console.print(f"\n[bold]Analysis:[/bold] {plan.get('task_analysis')}")
        console.print(f"[bold]Complexity:[/bold] {plan.get('complexity')}")
        console.print(f"\n[bold]Steps:[/bold]")
        for step in plan.get("steps", []):
            console.print(f"  {step['id']}. {step['description']}")
            console.print(f"     → {step.get('expected_output', 'N/A')}")
    else:
        console.print(f"[red]❌ Failed: {result.get('error')}[/red]")