"""
Extractor Agent

CONCEPT: Uses a more powerful LLM model (Sonnet vs Haiku) because extraction
requires reading tables, understanding context, and precise value parsing.

Key patterns:
- Map-reduce: Process each page independently, then merge results
- Tool calling: LLM uses defined tools to store structured values
- Confidence-weighted extraction: flag low-confidence values
- Table-aware prompting
"""
import json
import re
from typing import Optional
from loguru import logger
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from agents.state import FinancialPipelineState, ExtractedFinancials
from config.settings import get_settings
from tools.pdf_loader import extract_text_from_pdf, extract_tables_from_pdf

settings = get_settings()


EXTRACTION_SYSTEM = """You are a financial data extraction specialist. Extract precise numerical values
from financial documents. All amounts should be in their raw unit (millions/billions as stated).

RULES:
1. Extract values EXACTLY as stated — do not calculate or infer
2. Note the currency (USD, EUR, GBP, etc.) and unit (millions, billions, thousands)
3. For negative values use negative numbers (e.g., -234.5)
4. If a value is NOT present, return null
5. Confidence: 1.0 = value clearly stated in a table; 0.7 = inferred from text; 0.5 = uncertain

Return ONLY valid JSON. No explanations.

SCHEMA:
{
  "revenue": number | null,
  "gross_profit": number | null,
  "operating_income": number | null,
  "net_income": number | null,
  "ebitda": number | null,
  "eps_basic": number | null,
  "eps_diluted": number | null,
  "total_assets": number | null,
  "total_liabilities": number | null,
  "total_equity": number | null,
  "cash_and_equivalents": number | null,
  "total_debt": number | null,
  "operating_cash_flow": number | null,
  "investing_cash_flow": number | null,
  "financing_cash_flow": number | null,
  "free_cash_flow": number | null,
  "currency": "USD",
  "unit": "millions",
  "extraction_confidence": 0.0-1.0
}
"""


def _compute_ratios(data: dict) -> dict:
    """Compute derived financial ratios from extracted values."""
    def safe_div(a, b):
        if a is not None and b is not None and b != 0:
            return round(a / b, 4)
        return None

    data["debt_to_equity"] = safe_div(data.get("total_debt"), data.get("total_equity"))
    data["net_margin"] = safe_div(data.get("net_income"), data.get("revenue"))
    data["roe"] = safe_div(data.get("net_income"), data.get("total_equity"))
    data["roa"] = safe_div(data.get("net_income"), data.get("total_assets"))
    data["current_ratio"] = None  # Needs current assets/liabilities (add if available)

    if data.get("free_cash_flow") is None:
        # FCF = Operating CF - Capex (if capex available in investing CF, approximate)
        op_cf = data.get("operating_cash_flow")
        if op_cf is not None:
            data["free_cash_flow"] = op_cf  # Approximation when capex not split out

    return data


def extractor_node(state: FinancialPipelineState) -> FinancialPipelineState:
    """
    LangGraph node: Extracts financial metrics from each classified document.
    Uses the primary (more capable) model since extraction is complex.
    """
    logger.info(f"[EXTRACTOR] Starting extraction for session {state['session_id']}")

    llm = ChatAnthropic(
        model=settings.primary_llm_model,
        api_key=settings.anthropic_api_key,
        max_tokens=1024,
    )

    extractions = []
    input_tokens_used = 0

    for i, doc_path in enumerate(state["document_paths"]):
        classification = state["classifications"][i] if i < len(state["classifications"]) else {}
        doc_type = classification.get("doc_type", "unknown")

        try:
            # Get full text and any structured tables
            text = extract_text_from_pdf(doc_path, max_chars=8000)
            tables = extract_tables_from_pdf(doc_path)
            table_text = "\n".join([f"TABLE {j+1}:\n{t}" for j, t in enumerate(tables[:5])])

            company = classification.get("company_name", "Unknown")
            fiscal_year = classification.get("fiscal_year", "Unknown")

            prompt = f"""Extract financial data from this {doc_type} for {company} ({fiscal_year}).

DOCUMENT TEXT:
{text}

{f'STRUCTURED TABLES FOUND:{chr(10)}{table_text}' if table_text else ''}

Extract ALL available financial metrics. Return ONLY valid JSON."""

            messages = [
                SystemMessage(content=EXTRACTION_SYSTEM),
                HumanMessage(content=prompt),
            ]

            response = llm.invoke(messages)
            input_tokens_used += response.usage_metadata.get("input_tokens", 0)

            raw = response.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw)
            parsed = _compute_ratios(parsed)
            parsed["raw_tables"] = tables[:3]  # Store first 3 tables for reference

            result: ExtractedFinancials = {
                "revenue": parsed.get("revenue"),
                "gross_profit": parsed.get("gross_profit"),
                "operating_income": parsed.get("operating_income"),
                "net_income": parsed.get("net_income"),
                "ebitda": parsed.get("ebitda"),
                "eps_basic": parsed.get("eps_basic"),
                "eps_diluted": parsed.get("eps_diluted"),
                "total_assets": parsed.get("total_assets"),
                "total_liabilities": parsed.get("total_liabilities"),
                "total_equity": parsed.get("total_equity"),
                "cash_and_equivalents": parsed.get("cash_and_equivalents"),
                "total_debt": parsed.get("total_debt"),
                "operating_cash_flow": parsed.get("operating_cash_flow"),
                "investing_cash_flow": parsed.get("investing_cash_flow"),
                "financing_cash_flow": parsed.get("financing_cash_flow"),
                "free_cash_flow": parsed.get("free_cash_flow"),
                "debt_to_equity": parsed.get("debt_to_equity"),
                "current_ratio": parsed.get("current_ratio"),
                "roe": parsed.get("roe"),
                "roa": parsed.get("roa"),
                "net_margin": parsed.get("net_margin"),
                "currency": parsed.get("currency", "USD"),
                "unit": parsed.get("unit", "millions"),
                "extraction_confidence": parsed.get("extraction_confidence", 0.7),
                "raw_tables": parsed.get("raw_tables", []),
            }
            extractions.append(result)
            logger.info(
                f"[EXTRACTOR] {doc_path} -> Revenue: {result.get('revenue')} "
                f"Net Income: {result.get('net_income')} "
                f"Confidence: {result.get('extraction_confidence')}"
            )

        except Exception as e:
            logger.error(f"[EXTRACTOR] Failed on {doc_path}: {e}")
            empty: ExtractedFinancials = {k: None for k in ExtractedFinancials.__annotations__}
            empty["currency"] = "USD"
            empty["unit"] = "millions"
            empty["extraction_confidence"] = 0.0
            empty["raw_tables"] = []
            extractions.append(empty)
            state["errors"].append(f"Extraction failed for {doc_path}: {str(e)}")

    return {
        **state,
        "extractions": extractions,
        "total_input_tokens": state["total_input_tokens"] + input_tokens_used,
        "completed_steps": state["completed_steps"] + ["extraction"],
        "next_agent": "comparator",
    }
