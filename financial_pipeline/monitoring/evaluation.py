"""
RAG Evaluation Module

CONCEPT: How do you know your AI is actually working well?
You need an evaluation framework. Key metrics:

1. RAGAS metrics (standard RAG evaluation framework):
   - Faithfulness: Is the answer grounded in the retrieved context? (no hallucination)
   - Answer Relevancy: Does the answer address the question?
   - Context Recall: Did retrieval find all relevant chunks?
   - Context Precision: Are the retrieved chunks actually relevant?

2. Extraction accuracy: Did we extract the right numbers?
   - Compare against ground truth (manually labelled set)
   - F1 score on numeric values (within 1% tolerance)

3. Classification accuracy:
   - Precision/Recall on document type labels

In production: run eval suite on every model/prompt change (CI/CD for AI).
"""
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import get_settings

settings = get_settings()


@dataclass
class RAGEvalResult:
    question: str
    answer: str
    context: List[str]
    faithfulness_score: float       # 0-1: answer supported by context?
    relevancy_score: float          # 0-1: answer relevant to question?
    context_precision: float        # 0-1: retrieved chunks useful?
    overall_score: float            # weighted average


@dataclass
class ExtractionEvalResult:
    document: str
    metric: str
    predicted_value: Optional[float]
    ground_truth_value: Optional[float]
    is_correct: bool
    error_pct: Optional[float]


class LLMJudge:
    """
    LLM-as-a-judge: uses Claude to evaluate another Claude's answers.
    This is the industry standard for evaluating open-ended text.

    Interview point: "How do you evaluate RAG quality without ground truth?"
    Answer: LLM-as-a-judge + human spot checks + automated metrics.
    """

    def __init__(self):
        self.llm = ChatAnthropic(
            model=settings.fast_llm_model,  # Use cheaper model for eval
            api_key=settings.anthropic_api_key,
            max_tokens=256,
        )

    def score_faithfulness(self, answer: str, context: List[str]) -> float:
        """
        Is every claim in the answer supported by the context?
        Score: 1.0 = fully grounded; 0.0 = complete hallucination.
        """
        context_text = "\n---\n".join(context[:3])
        prompt = f"""Rate how well the ANSWER is supported by the CONTEXT.
Score from 0.0 (completely unsupported) to 1.0 (fully supported).
Only consider what is explicitly stated in the context.

CONTEXT:
{context_text}

ANSWER:
{answer}

Return ONLY a JSON: {{"score": 0.85, "reason": "..."}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an objective evaluator. Return only JSON."),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            parsed = json.loads(raw)
            return float(parsed.get("score", 0.5))
        except Exception as e:
            logger.warning(f"[EVAL] Faithfulness scoring failed: {e}")
            return 0.5

    def score_relevancy(self, question: str, answer: str) -> float:
        """Does the answer actually address the question?"""
        prompt = f"""Rate how well the ANSWER addresses the QUESTION.
Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).

QUESTION: {question}
ANSWER: {answer}

Return ONLY JSON: {{"score": 0.9, "reason": "..."}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an objective evaluator. Return only JSON."),
                HumanMessage(content=prompt),
            ])
            raw = response.content.strip()
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            parsed = json.loads(raw)
            return float(parsed.get("score", 0.5))
        except Exception as e:
            logger.warning(f"[EVAL] Relevancy scoring failed: {e}")
            return 0.5


def evaluate_rag_response(
    question: str,
    answer: str,
    retrieved_chunks: List[Dict],
) -> RAGEvalResult:
    """
    Evaluate a single RAG response.

    Args:
        question: The user's question
        answer: The generated answer
        retrieved_chunks: Chunks used to generate the answer

    Returns:
        RAGEvalResult with component scores
    """
    judge = LLMJudge()
    context_texts = [c.get("text", "") for c in retrieved_chunks[:4]]

    faithfulness = judge.score_faithfulness(answer, context_texts)
    relevancy = judge.score_relevancy(question, answer)

    # Context precision: proportion of retrieved chunks that are actually relevant
    # (simplified: use retrieval scores as proxy)
    scores = [c.get("score", c.get("rrf_score", 0.5)) for c in retrieved_chunks[:4]]
    context_precision = sum(scores) / len(scores) if scores else 0.0

    overall = round(
        0.4 * faithfulness + 0.4 * relevancy + 0.2 * context_precision, 3
    )

    result = RAGEvalResult(
        question=question,
        answer=answer,
        context=context_texts,
        faithfulness_score=round(faithfulness, 3),
        relevancy_score=round(relevancy, 3),
        context_precision=round(context_precision, 3),
        overall_score=overall,
    )

    logger.info(
        f"[EVAL] Faithfulness: {result.faithfulness_score:.2f} | "
        f"Relevancy: {result.relevancy_score:.2f} | "
        f"Overall: {result.overall_score:.2f}"
    )
    return result


def evaluate_extraction_accuracy(
    predicted: Dict,
    ground_truth: Dict,
    tolerance_pct: float = 1.0,
) -> List[ExtractionEvalResult]:
    """
    Compare extracted financial values against ground truth.
    Used for regression testing after model/prompt changes.

    tolerance_pct: accept values within X% of ground truth as correct.
    """
    results = []
    numeric_metrics = [
        "revenue", "net_income", "ebitda", "total_assets",
        "total_liabilities", "total_equity", "eps_diluted",
    ]

    for metric in numeric_metrics:
        pred = predicted.get(metric)
        truth = ground_truth.get(metric)

        if truth is None:
            continue

        error_pct = None
        is_correct = False

        if pred is not None and truth != 0:
            error_pct = abs((pred - truth) / truth) * 100
            is_correct = error_pct <= tolerance_pct
        elif pred is None:
            is_correct = False

        results.append(ExtractionEvalResult(
            document=predicted.get("source", "unknown"),
            metric=metric,
            predicted_value=pred,
            ground_truth_value=truth,
            is_correct=is_correct,
            error_pct=round(error_pct, 2) if error_pct is not None else None,
        ))

    correct = sum(1 for r in results if r.is_correct)
    total = len(results)
    logger.info(f"[EVAL] Extraction accuracy: {correct}/{total} = {100*correct/max(total,1):.1f}%")
    return results


def run_evaluation_suite(test_cases: List[Dict]) -> Dict:
    """
    Run the full evaluation suite on a list of test cases.

    Each test case: {"question": str, "answer": str, "chunks": list}

    Returns aggregate metrics suitable for a dashboard or CI check.
    """
    all_results = []
    for case in test_cases:
        result = evaluate_rag_response(
            question=case["question"],
            answer=case["answer"],
            retrieved_chunks=case.get("chunks", []),
        )
        all_results.append(result)

    if not all_results:
        return {}

    avg = lambda key: round(sum(getattr(r, key) for r in all_results) / len(all_results), 3)

    return {
        "n_evaluated": len(all_results),
        "avg_faithfulness": avg("faithfulness_score"),
        "avg_relevancy": avg("relevancy_score"),
        "avg_context_precision": avg("context_precision"),
        "avg_overall": avg("overall_score"),
        "pass_rate": round(
            sum(1 for r in all_results if r.overall_score >= 0.7) / len(all_results), 3
        ),
    }
