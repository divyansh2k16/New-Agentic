"""
Async orchestrator: runs all 10 agents in parallel per round, recurses
until confidence converges or max_depth is reached.

Key latency fix: asyncio.gather fires all agents simultaneously.
Wall time per round ≈ slowest single agent (~1-2s), NOT sum of all agents (~15s).

Key cost fix: the system prompt of each agent is cached after round 1.
Rounds 2-N pay only 10% of the input cost for those cached bytes.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from agents import ALL_AGENTS
from agents.base import AgentResult
from config import CONFIDENCE_THRESHOLD, MAX_RECURSION_DEPTH
from monitoring.cost_tracker import CostTracker


@dataclass
class PipelineState:
    task: str
    context: str
    round_results: list[list[AgentResult]] = field(default_factory=list)
    depth: int = 0
    done: bool = False


async def _run_round(
    state: PipelineState,
    tracker: CostTracker,
    time_limit: float | None,
    start_time: float,
) -> list[AgentResult]:
    """
    Fire all agents concurrently.
    Each agent receives the task, the shared context, and the flattened
    findings from the previous round so it can refine its analysis.
    """
    prior: list[AgentResult] = (
        state.round_results[-1] if state.round_results else []
    )

    agents = [cls() for cls in ALL_AGENTS]

    async def _safe_run(agent) -> AgentResult:
        if time_limit and (time.time() - start_time) > time_limit:
            from agents.base import AgentResult
            return AgentResult(
                agent=agent.NAME,
                finding="Time limit reached — skipped.",
                confidence=0.0,
                metadata={"skipped": True},
            )
        return await agent.run(
            task=state.task,
            context=state.context,
            prior_findings=prior,
            tracker=tracker,
        )

    results = await asyncio.gather(*[_safe_run(a) for a in agents])
    return list(results)


def _overall_confidence(results: list[AgentResult]) -> float:
    if not results:
        return 0.0
    return sum(r.confidence for r in results) / len(results)


async def run_pipeline(
    task: str,
    context: str,
    max_depth: int = MAX_RECURSION_DEPTH,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    time_limit_seconds: float | None = 1800.0,  # 30-min wall clock guard
) -> tuple[PipelineState, CostTracker]:
    """
    Entry point. Returns (final state, cost tracker).

    Recursion terminates when EITHER:
      - overall confidence >= confidence_threshold   (quality gate)
      - depth >= max_depth                           (safety guard)
      - wall-clock time >= time_limit_seconds        (cost guard)
    """
    state = PipelineState(task=task, context=context)
    tracker = CostTracker()
    start = time.time()

    print(f"\nOrchestrator starting — max {max_depth} rounds, "
          f"confidence target {confidence_threshold:.0%}\n")

    while state.depth < max_depth:
        round_num = state.depth + 1
        t0 = time.time()
        print(f"  Round {round_num}: firing {len(ALL_AGENTS)} agents in parallel...", end=" ", flush=True)

        results = await _run_round(state, tracker, time_limit_seconds, start)
        state.round_results.append(results)
        state.depth += 1

        elapsed = time.time() - t0
        conf = _overall_confidence(results)
        print(f"done in {elapsed:.1f}s  |  avg confidence={conf:.2f}")

        if conf >= confidence_threshold:
            print(f"  Confidence threshold reached — stopping.\n")
            break

        if time_limit_seconds and (time.time() - start) >= time_limit_seconds:
            print(f"  Time limit reached — stopping.\n")
            break

    state.done = True
    return state, tracker


def final_report(state: PipelineState) -> str:
    """Flatten the last round's findings into a readable report."""
    if not state.round_results:
        return "No results."

    last = state.round_results[-1]
    lines = [
        f"=== Pipeline Report (completed in {state.depth} round(s)) ===\n"
    ]
    for r in last:
        lines.append(f"[{r.agent.upper()}]  confidence={r.confidence:.2f}")
        lines.append(f"  {r.finding}")
        lines.append("")
    return "\n".join(lines)
