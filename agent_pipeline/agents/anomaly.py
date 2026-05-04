from agents.base import BaseAgent


class AnomalyAgent(BaseAgent):
    NAME = "anomaly"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an anomaly detection specialist. "
            "Identify outliers, unusual patterns, suspicious entries, and deviations from expected norms. "
            "Assign a severity (low/medium/high) to each anomaly found. Output JSON only."
        )
