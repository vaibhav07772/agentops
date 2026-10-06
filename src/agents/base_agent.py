"""AgentOps — Base Agent (foundation for all agents)"""
from abc import ABC, abstractmethod
from typing import Dict, Any
from rich.console import Console
from src.core.llm_client import call_llm, DEFAULT_MODEL

console = Console()


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(
        self,
        name: str,
        description: str,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def call_llm(
        self,
        messages: list,
        tools: list = None,
        json_mode: bool = False,
        max_tokens: int = None,
    ) -> Dict:
        """Helper to call LLM with this agent's config"""
        return call_llm(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            tools=tools,
            json_mode=json_mode,
            max_tokens=max_tokens or self.max_tokens,
        )

    def log(self, message: str, style: str = "dim"):
        console.print(f"[{style}][{self.name}][/{style}] {message}")

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}')>"