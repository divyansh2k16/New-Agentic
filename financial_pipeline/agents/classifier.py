"""
Classifier Agent

CONCEPT: This agent is a specialised node in the LangGraph. Its only job is
to classify a financial document into a type and check for dual-use material
(trade finance specific to Citi use case).

Key patterns used:
- Structured output with Pydantic model
- Few-shot prompting for accuracy
- Confidence scoring
"""
import json
from loguru import logger
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List

from agents.state import FinancialPipelineState, ClassificationResult
from config.llm_factory import get_llm
from tools.pdf_loader import extract_text_from_pdf


class ClassificationOutput(BaseModel):
    doc_type: str = Field(
        description="One of: annual_report, income_statement, balance_sheet, cash_flow, earnings_release, trade_doc, unknown"
    )
    company_name: str = Field(description="Company or entity name found in the document")
    fiscal_year: str = Field(description="Fiscal year e.g. 2023, 2024")
    fiscal_period: str = Field(description="Q1, Q2, Q3, Q4, or FY (full year)")
    confidence: float = Field(description="0.0 to 1.0 confidence in classification")
    is_dual_use_material: bool = Field(
        description="True if document references goods/services with potential dual civilian-military use"
    )
    dual_use_reasons: List[str] = Field(
        description="List of specific reasons why the document is flagged as dual-use"
    )
    language: str = Field(description="Primary language of the document e.g. en, fr, de")


SYSTEM_PROMPT = """You are a financial document classification expert at a global investment bank.
Your task is to classify financial documents and identify trade compliance risks.

DOCUMENT TYPES:
- annual_report: Full year report with all financials, MD&A, and notes
- income_statement: P&L / profit & loss statement
- balance_sheet: Statement of financial position
- cash_flow: Cash flow statement
- earnings_release: Quarterly earnings press release
- trade_doc: Trade finance documents (letters of credit, invoices, bills of lading)
- unknown: Cannot be determined

DUAL-USE MATERIAL DEFINITION:
Dual-use goods are items that can be used for both civilian and military purposes.
Flag as True if the document mentions: weapons components, encryption technology,
chemical precursors, nuclear materials, aerospace technology, surveillance systems,
or any goods subject to export controls (EAR, ITAR, EU dual-use regulation).

INSTRUCTIONS:
- Analyse only the first 2000 characters of text (for efficiency)
- Return a JSON object matching the schema exactly
- Be conservative: if unsure about dual-use, flag it (false negatives are costly at a bank)
"""

FEW_SHOT_EXAMPLES = """
EXAMPLE 1:
Text: "Apple Inc. Annual Report 2023. For the fiscal year ended September 30, 2023.
Total net sales: $383.3 billion. Net income: $97.0 billion..."
Output: {"doc_type": "annual_report", "company_name": "Apple Inc.", "fiscal_year": "2023",
"fiscal_period": "FY", "confidence": 0.97, "is_dual_use_material": false, "dual_use_reasons": [], "language": "en"}

EXAMPLE 2:
Text: "COMMERCIAL INVOICE. Seller: XYZ Defense Corp. Goods: Night vision components,
Model NVG-7, Quantity: 500 units. Export License No: D/2023/001..."
Output: {"doc_type": "trade_doc", "company_name": "XYZ Defense Corp.", "fiscal_year": "2023",
"fiscal_period": "FY", "confidence": 0.95, "is_dual_use_material": true,
"dual_use_reasons": ["Night vision components are dual-use military/civilian equipment",
"Export license referenced indicating controlled goods"], "language": "en"}
"""


def classifier_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: Classifies each document in document_paths.
    Writes results to state['classifications'].
    """
    logger.info(f"[CLASSIFIER] Starting classification for session {state['session_id']}")
    llm = get_llm(model_tier="fast", max_tokens=512)

    classifications = []
    input_tokens_used = 0

    for doc_path in state["document_paths"]:
        try:
            # Extract first page text (cost optimisation: don't send full doc)
            text = extract_text_from_pdf(doc_path, max_chars=2000)
            logger.info(f"[CLASSIFIER] Processing: {doc_path}")

            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=f"""
{FEW_SHOT_EXAMPLES}

NOW CLASSIFY THIS DOCUMENT:
Text excerpt: {text}

Return ONLY valid JSON matching the ClassificationOutput schema.
""")
            ]

            response = llm.invoke(messages)
            input_tokens_used += response.usage_metadata.get("input_tokens", 0)

            # Parse JSON from response
            raw_text = response.content.strip()
            # Handle markdown code blocks if present
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            result: ClassificationResult = {
                "doc_type": parsed.get("doc_type", "unknown"),
                "company_name": parsed.get("company_name", ""),
                "fiscal_year": parsed.get("fiscal_year", ""),
                "fiscal_period": parsed.get("fiscal_period", "FY"),
                "confidence": parsed.get("confidence", 0.0),
                "is_dual_use_material": parsed.get("is_dual_use_material", False),
                "dual_use_reasons": parsed.get("dual_use_reasons", []),
                "language": parsed.get("language", "en"),
            }
            classifications.append(result)
            logger.info(
                f"[CLASSIFIER] {doc_path} -> {result['doc_type']} | "
                f"{result['company_name']} {result['fiscal_year']} | "
                f"Dual-use: {result['is_dual_use_material']}"
            )

        except Exception as e:
            logger.error(f"[CLASSIFIER] Failed on {doc_path}: {e}")
            classifications.append({
                "doc_type": "unknown", "company_name": "", "fiscal_year": "",
                "fiscal_period": "FY", "confidence": 0.0,
                "is_dual_use_material": False, "dual_use_reasons": [], "language": "en",
            })
            state["errors"].append(f"Classification failed for {doc_path}: {str(e)}")

    return {
        **state,
        "classifications": classifications,
        "total_input_tokens": state["total_input_tokens"] + input_tokens_used,
        "completed_steps": state["completed_steps"] + ["classification"],
        "next_agent": "extractor",
    }
