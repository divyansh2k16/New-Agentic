from agents.base import BaseAgent


class ExtractorAgent(BaseAgent):
    NAME = "extractor"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a data extraction specialist. "
            "Extract all key entities, values, dates, parties, and amounts from the content. "
            "Structure findings as a concise summary. Output JSON only."
        )
