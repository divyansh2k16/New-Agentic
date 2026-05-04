"""
Base class for all agents.

Every agent:
  - defines a NAME and SYSTEM_PROMPT (cached on first call)
  - defines a model (defaults to Haiku; Sonnet for heavy reasoning)
  - returns an AgentResult with a text finding + confidence [0,1]
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from config import HAIKU, TOKEN_BUDGETS
from llm_client import call
from monitoring.cost_tracker import CostTracker


@dataclass
class AgentResult:
    agent: str
    finding: str
    confidence: float   # 0.0 – 1.0
    metadata: dict


class BaseAgent(ABC):
    NAME: str = "base"
    MODEL: str = HAIKU

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the (potentially long) system prompt to be cached."""

    async def run(
        self,
        *,
        task: str,
        context: str,
        prior_findings: list[AgentResult],
        tracker: CostTracker,
    ) -> AgentResult:
        """
        Build the user message from task + context + prior round findings,
        call the LLM (cached system prompt), parse the JSON response.
        """
        prior_text = ""
        if prior_findings:
            prior_text = "\n\nPrior round findings:\n" + "\n".join(
                f"- [{r.agent}] {r.finding} (confidence={r.confidence:.2f})"
                for r in prior_findings
            )

        user_message = (
            f"TASK: {task}\n\n"
            f"CONTEXT:\n{context[:6000]}"   # hard cap to prevent runaway tokens
            f"{prior_text}\n\n"
            "Respond ONLY with valid JSON:\n"
            '{"finding": "<your finding>", "confidence": <0.0-1.0>, "metadata": {}}'
        )

        raw = await call(
            system_prompt=self.system_prompt,
            user_message=user_message,
            model=self.MODEL,
            max_tokens=TOKEN_BUDGETS.get(self.NAME, 512),
            tracker=tracker,
        )

        return self._parse(raw)

    def _parse(self, raw: str) -> AgentResult:
        """Extract JSON from the model response; fallback gracefully."""
        try:
            # strip markdown code fences if present
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(clean)
            return AgentResult(
                agent=self.NAME,
                finding=str(data.get("finding", raw[:200])),
                confidence=float(data.get("confidence", 0.5)),
                metadata=data.get("metadata", {}),
            )
        except (json.JSONDecodeError, ValueError):
            return AgentResult(
                agent=self.NAME,
                finding=raw[:200],
                confidence=0.4,
                metadata={"parse_error": True},
            )
