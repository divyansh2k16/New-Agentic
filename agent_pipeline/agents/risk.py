from config import SONNET
from agents.base import BaseAgent


class RiskAgent(BaseAgent):
    """Risk assessment warrants Sonnet-level reasoning."""
    NAME = "risk"
    MODEL = SONNET

    @property
    def system_prompt(self) -> str:
        return (
            "You are a risk assessment specialist. "
            "Evaluate financial, operational, and compliance risks in the provided content. "
            "Score overall risk LOW/MEDIUM/HIGH and explain key risk drivers. Output JSON only."
        )
