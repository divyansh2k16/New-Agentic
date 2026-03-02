"""
Shared LangGraph state definition.

WHY THIS MATTERS (Interview concept):
LangGraph is a stateful graph framework. Every node (agent) reads from and
writes to this shared TypedDict. The graph router looks at this state to
decide which node to execute next — this is the backbone of multi-agent coordination.
"""
from typing import TypedDict, List, Optional, Dict, Any, Annotated
import operator


class DocumentMetadata(TypedDict):
    filename: str
    file_path: str
    page_count: int
    file_size_bytes: int
    upload_timestamp: str
    doc_id: str


class ClassificationResult(TypedDict):
    doc_type: str           # annual_report | income_statement | balance_sheet | cash_flow | earnings_release | trade_doc
    company_name: str
    fiscal_year: str
    fiscal_period: str      # Q1 | Q2 | Q3 | Q4 | FY
    confidence: float
    is_dual_use_material: bool          # Citi-specific: trade document flag
    dual_use_reasons: List[str]
    language: str


class ExtractedFinancials(TypedDict):
    # Income Statement
    revenue: Optional[float]
    gross_profit: Optional[float]
    operating_income: Optional[float]
    net_income: Optional[float]
    ebitda: Optional[float]
    eps_basic: Optional[float]
    eps_diluted: Optional[float]

    # Balance Sheet
    total_assets: Optional[float]
    total_liabilities: Optional[float]
    total_equity: Optional[float]
    cash_and_equivalents: Optional[float]
    total_debt: Optional[float]

    # Cash Flow
    operating_cash_flow: Optional[float]
    investing_cash_flow: Optional[float]
    financing_cash_flow: Optional[float]
    free_cash_flow: Optional[float]

    # Ratios (computed)
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    roe: Optional[float]                # Return on Equity
    roa: Optional[float]                # Return on Assets
    net_margin: Optional[float]

    # Metadata
    currency: str
    unit: str                           # millions | billions | thousands
    extraction_confidence: float
    raw_tables: List[Dict]


class ComparisonResult(TypedDict):
    documents_compared: List[str]
    yoy_changes: Dict[str, Any]         # metric -> {value, pct_change, trend}
    key_insights: List[str]
    risk_flags: List[str]


class FinancialPipelineState(TypedDict):
    """
    The central state object shared across ALL agents in the LangGraph.

    Annotated[list, operator.add] means each agent APPENDS to messages
    instead of overwriting — standard LangGraph pattern.
    """
    # Input
    task: str                           # classify | extract | compare | summarize | query
    user_id: str
    session_id: str
    query: Optional[str]                # For RAG query tasks

    # Documents being processed
    document_paths: List[str]
    current_doc_index: int
    documents_metadata: List[DocumentMetadata]

    # Agent outputs (accumulated per document)
    classifications: List[ClassificationResult]
    extractions: List[ExtractedFinancials]
    comparison: Optional[ComparisonResult]
    summary: Optional[str]
    query_response: Optional[str]
    retrieved_chunks: List[Dict]

    # Conversation history (append-only via operator.add)
    messages: Annotated[List[Dict], operator.add]

    # Pipeline control
    next_agent: str                     # Router writes this; LangGraph reads it
    errors: Annotated[List[str], operator.add]
    completed_steps: Annotated[List[str], operator.add]

    # Token tracking (for cost optimisation)
    total_input_tokens: int
    total_output_tokens: int

    # Final output
    final_report: Optional[str]
