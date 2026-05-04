from agents.classifier  import ClassifierAgent
from agents.extractor   import ExtractorAgent
from agents.analyst     import AnalystAgent
from agents.validator   import ValidatorAgent
from agents.anomaly     import AnomalyAgent
from agents.router      import RouterAgent
from agents.summarizer  import SummarizerAgent
from agents.comparator  import ComparatorAgent
from agents.risk        import RiskAgent
from agents.formatter   import FormatterAgent

ALL_AGENTS = [
    ClassifierAgent,
    ExtractorAgent,
    AnalystAgent,
    ValidatorAgent,
    AnomalyAgent,
    RouterAgent,
    SummarizerAgent,
    ComparatorAgent,
    RiskAgent,
    FormatterAgent,
]
