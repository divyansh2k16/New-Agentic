from agents.base import BaseAgent


class ComparatorAgent(BaseAgent):
    NAME = "comparator"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a cross-document comparison specialist. "
            "Identify agreements, contradictions, and gaps across multiple documents. "
            "Flag where documents reference the same entities with different values. Output JSON only."
        )
