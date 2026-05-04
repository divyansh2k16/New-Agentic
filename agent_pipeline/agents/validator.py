from agents.base import BaseAgent


class ValidatorAgent(BaseAgent):
    NAME = "validator"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a data validation expert. "
            "Check for inconsistencies, missing values, constraint violations, and data quality issues. "
            "Report what passes and what fails. Output JSON only."
        )
