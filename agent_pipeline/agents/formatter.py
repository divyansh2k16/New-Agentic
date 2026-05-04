from agents.base import BaseAgent


class FormatterAgent(BaseAgent):
    NAME = "formatter"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an output formatting specialist. "
            "Take all findings and produce a clean, structured final report. "
            "Use clear sections: Summary, Key Findings, Anomalies, Risk, Recommendations. "
            "Output JSON only."
        )
