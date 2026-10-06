"""AgentOps — LLM Client with Helicone proxy support"""
import os
from dotenv import load_dotenv
from groq import Groq
from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type,
)
from rich.console import Console

load_dotenv()
console = Console()


def get_llm_client(use_helicone: bool = False):
    """Get Groq client with optional Helicone proxy"""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY missing in .env")

    if use_helicone:
        helicone_key = os.getenv("HELICONE_API_KEY")
        if helicone_key:
            console.print("[cyan]→ Using Helicone proxy[/cyan]")
            return Groq(
                api_key=groq_key,
                base_url="https://groq.helicone.ai/openai/v1",
                default_headers={
                    "Helicone-Auth": f"Bearer {helicone_key}",
                },
            )

    return Groq(api_key=groq_key)


DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_llm(
    messages: list,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 500,
    tools: list = None,
    json_mode: bool = False,
    use_helicone: bool = False,
) -> dict:
    """Call LLM with retry + optional JSON mode"""
    client = get_llm_client(use_helicone=use_helicone)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)

    return {
        "message": response.choices[0].message,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
        "model": response.model,
    }


if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — LLM Client Test[/bold magenta]")
    console.print("="*60)

    result = call_llm(
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
    )

    console.print(f"\n[green]✅ Response:[/green] {result['message'].content}")
    console.print(f"[dim]Tokens used: {result['usage']['total_tokens']}[/dim]")