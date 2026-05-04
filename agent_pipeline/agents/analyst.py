from config import SONNET
from agents.base import BaseAgent


class AnalystAgent(BaseAgent):
    """Uses Sonnet for deeper pattern reasoning — escalate only when needed."""
    NAME = "analyst"
    MODEL = SONNET

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior data analyst. "
            "Identify patterns, correlations, and trends in the provided data and documents. "
            "Highlight statistically significant observations and business implications. "
            "Output JSON only."
        )
