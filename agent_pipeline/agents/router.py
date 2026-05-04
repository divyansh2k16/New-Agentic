from agents.base import BaseAgent


class RouterAgent(BaseAgent):
    """Lightweight decision node — Haiku + tiny token budget."""
    NAME = "router"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a task routing agent. "
            "Based on the findings so far, decide the next processing priority: "
            "'continue', 'escalate', or 'terminate'. "
            "Explain your routing decision briefly. Output JSON only."
        )
