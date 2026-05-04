from agents.base import BaseAgent


class ClassifierAgent(BaseAgent):
    NAME = "classifier"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a document classification expert. "
            "Analyse the provided content and identify the document type, category, "
            "and key structural elements. "
            "Be precise and concise. Output JSON only."
        )
