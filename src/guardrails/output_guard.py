"""AgentOps — Output Guardrail (citation + leak + toxicity)"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import re
from typing import Dict, List
from dataclasses import dataclass

from rich.console import Console

console = Console()


# ─────────────────────────────────────
# Patterns
# ─────────────────────────────────────
CITATION_PATTERNS = [
    r"\[Source\s+\d+\]",
    r"Source\s+\d+",
]

TOXIC_PATTERNS = [
    r"\b(kill|murder|rape|terrorist|bomb)\s+(you|him|her|them|everyone)",
    r"\b(hate|stupid|idiot|moron)\s+(you|him|her)",
    r"\b(how\s+to\s+make\s+(a\s+)?bomb)",
]

PII_LEAK_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE_IN": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
}


@dataclass
class OutputGuardrailResult:
    """Output guardrail result"""
    valid: bool
    issues: List[str]
    citations: List[str]
    toxicity: List[str]
    pii_leaks: Dict[str, List[str]]


class OutputGuardrail:
    """Output validation"""

    def __init__(self):
        self.citation_compiled = [re.compile(p, re.IGNORECASE) for p in CITATION_PATTERNS]
        self.toxic_compiled = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]

    def extract_citations(self, text: str) -> List[str]:
        citations = []
        for pattern in self.citation_compiled:
            citations.extend(pattern.findall(text))
        # Dedupe preserving order
        seen = set()
        unique = []
        for c in citations:
            c_norm = c.strip()
            if c_norm.lower() not in seen:
                seen.add(c_norm.lower())
                unique.append(c_norm)
        return unique

    def check_toxicity(self, text: str) -> List[str]:
        detected = []
        for pattern in self.toxic_compiled:
            match = pattern.search(text)
            if match:
                detected.append(match.group(0)[:50])
        return detected

    def check_pii_leak(self, text: str) -> Dict[str, List[str]]:
        found = {}
        for pii_type, pattern in PII_LEAK_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                found[pii_type] = list(set(matches))
        return found

    def validate(
        self,
        text: str,
        requires_citation: bool = False,
    ) -> OutputGuardrailResult:
        """Full output validation"""
        issues = []
        valid = True

        # 1. Citation check (only if required)
        citations = self.extract_citations(text)
        if requires_citation and not citations:
            valid = False
            issues.append("No citations found (RAG response must cite sources)")

        # 2. Toxicity check
        toxicity = self.check_toxicity(text)
        if toxicity:
            valid = False
            issues.append(f"Toxic content detected: {toxicity}")

        # 3. PII leak check
        pii_leaks = self.check_pii_leak(text)
        if pii_leaks:
            valid = False
            issues.append(f"PII leaked in output: {list(pii_leaks.keys())}")

        return OutputGuardrailResult(
            valid=valid,
            issues=issues,
            citations=citations,
            toxicity=toxicity,
            pii_leaks=pii_leaks,
        )


# ─────────────────────────────────────
# Test
# ─────────────────────────────────────
if __name__ == "__main__":
    console.print("\n" + "="*60)
    console.print("[bold magenta]AgentOps — Output Guardrail Test[/bold magenta]")
    console.print("="*60)

    guard = OutputGuardrail()

    test_cases = [
        (
            "FastAPI is best for APIs [Source 1]. It has async support.",
            True,
            "Valid RAG with citation",
        ),
        (
            "FastAPI is best for APIs.",
            True,
            "Missing citation (RAG)",
        ),
        (
            "The answer is 42.",
            False,
            "Non-RAG (no citation needed)",
        ),
        (
            "Contact test@example.com for more info. [Source 1]",
            True,
            "PII leak",
        ),
        (
            "I will kill you [Source 1]",
            True,
            "Toxic content",
        ),
    ]

    for text, requires_citation, label in test_cases:
        console.print(f"\n[cyan]Test ({label}):[/cyan] {text[:80]}")
        result = guard.validate(text, requires_citation=requires_citation)
        console.print(f"  Valid: {result.valid}")

        if result.citations:
            console.print(f"  [green]Citations:[/green] {result.citations}")

        if result.issues:
            for issue in result.issues:
                console.print(f"  [red]⚠ {issue}[/red]")

        console.print("-" * 60)