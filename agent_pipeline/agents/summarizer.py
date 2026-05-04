from agents.base import BaseAgent


class SummarizerAgent(BaseAgent):
    NAME = "summarizer"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a summarisation specialist. "
            "Produce a concise executive summary of the documents and dataset findings. "
            "Focus on what matters most for decision-making. Output JSON only."
        )
