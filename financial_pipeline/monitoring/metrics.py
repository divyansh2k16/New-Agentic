"""
Monitoring & Metrics

CONCEPT: Production AI systems need observability. Key metrics:
1. Latency: How long does each agent take? (P50, P95, P99)
2. Token usage: Cost tracking per user/session
3. Error rates: Which agents fail most?
4. Quality scores: Are answers good? (via evaluation)
5. Throughput: Requests per minute

Tools:
- Local: Prometheus + Grafana (via prometheus-client)
- Cloud: AWS CloudWatch, Azure Monitor, Datadog
- LLM-specific: LangSmith (traces every LangChain/LangGraph call)
"""
import time
import functools
from typing import Callable
from datetime import datetime
from collections import defaultdict
from loguru import logger

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ── Prometheus metrics ────────────────────────────────────────────────────────
if HAS_PROMETHEUS:
    PIPELINE_REQUESTS = Counter(
        "pipeline_requests_total",
        "Total pipeline invocations",
        ["task", "user_id", "status"]
    )
    AGENT_LATENCY = Histogram(
        "agent_latency_seconds",
        "Agent execution latency",
        ["agent_name"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
    )
    TOKEN_USAGE = Counter(
        "llm_tokens_total",
        "LLM tokens consumed",
        ["direction", "model"]  # direction: input | output
    )
    ACTIVE_SESSIONS = Gauge(
        "active_sessions",
        "Currently active processing sessions"
    )
    RETRIEVAL_SCORE = Histogram(
        "retrieval_relevance_score",
        "RAG chunk relevance scores",
        buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    )


# ── In-memory fallback metrics store ─────────────────────────────────────────
_metrics_store = {
    "requests": defaultdict(int),
    "latencies": defaultdict(list),
    "token_usage": defaultdict(int),
    "errors": defaultdict(int),
}


class MetricsCollector:
    """
    Lightweight metrics collector. Wraps both Prometheus and in-memory store.
    In production: emit to CloudWatch/Datadog directly.
    """

    @staticmethod
    def record_pipeline_start(task: str, user_id: str, session_id: str):
        _metrics_store["requests"][f"pipeline.{task}"] += 1
        if HAS_PROMETHEUS:
            ACTIVE_SESSIONS.inc()
        logger.info(f"[METRICS] Pipeline start | task={task} user={user_id} session={session_id}")

    @staticmethod
    def record_pipeline_end(task: str, user_id: str, success: bool, duration_s: float):
        status = "success" if success else "failure"
        _metrics_store["latencies"]["pipeline"].append(duration_s)
        if not success:
            _metrics_store["errors"]["pipeline"] += 1
        if HAS_PROMETHEUS:
            PIPELINE_REQUESTS.labels(task=task, user_id=user_id, status=status).inc()
            ACTIVE_SESSIONS.dec()

    @staticmethod
    def record_agent_latency(agent_name: str, duration_s: float):
        _metrics_store["latencies"][agent_name].append(duration_s)
        if HAS_PROMETHEUS:
            AGENT_LATENCY.labels(agent_name=agent_name).observe(duration_s)
        logger.debug(f"[METRICS] Agent {agent_name}: {duration_s:.2f}s")

    @staticmethod
    def record_token_usage(input_tokens: int, output_tokens: int, model: str):
        _metrics_store["token_usage"]["input"] += input_tokens
        _metrics_store["token_usage"]["output"] += output_tokens
        if HAS_PROMETHEUS:
            TOKEN_USAGE.labels(direction="input", model=model).inc(input_tokens)
            TOKEN_USAGE.labels(direction="output", model=model).inc(output_tokens)

    @staticmethod
    def record_retrieval_scores(scores: list):
        for score in scores:
            _metrics_store["latencies"]["retrieval_score"].append(score)
            if HAS_PROMETHEUS:
                RETRIEVAL_SCORE.observe(score)

    @staticmethod
    def get_summary() -> dict:
        """Return a summary of collected metrics."""
        def avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else 0

        def p95(lst):
            if not lst:
                return 0
            sorted_lst = sorted(lst)
            idx = int(0.95 * len(sorted_lst))
            return round(sorted_lst[idx], 3)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": sum(_metrics_store["requests"].values()),
            "total_errors": sum(_metrics_store["errors"].values()),
            "total_input_tokens": _metrics_store["token_usage"]["input"],
            "total_output_tokens": _metrics_store["token_usage"]["output"],
            "estimated_cost_usd": round(
                _metrics_store["token_usage"]["input"] * 3e-6 +
                _metrics_store["token_usage"]["output"] * 15e-6,
                4
            ),  # Claude Sonnet pricing
            "agent_latencies": {
                agent: {
                    "avg_s": avg(times),
                    "p95_s": p95(times),
                    "count": len(times),
                }
                for agent, times in _metrics_store["latencies"].items()
                if agent != "retrieval_score"
            },
        }


# ── Decorator for automatic timing ───────────────────────────────────────────
def timed_agent(agent_name: str):
    """
    Decorator to automatically time agent execution and record metrics.

    Usage:
        @timed_agent("classifier")
        def classifier_node(state): ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                MetricsCollector.record_agent_latency(agent_name, duration)
                return result
            except Exception as e:
                duration = time.perf_counter() - start
                _metrics_store["errors"][agent_name] += 1
                MetricsCollector.record_agent_latency(agent_name, duration)
                raise
        return wrapper
    return decorator


metrics = MetricsCollector()
