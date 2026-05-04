"""
Tracks token usage and running cost across all agents and rounds.
Thread-safe via asyncio.Lock so concurrent agents can all record safely.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from config import PRICING


@dataclass
class CostTracker:
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_write_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0

    async def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        pricing = PRICING.get(model, PRICING[list(PRICING)[0]])
        cost = (
            input_tokens        * pricing["input"]        / 1_000_000
            + output_tokens     * pricing["output"]       / 1_000_000
            + cache_creation_tokens * pricing["cache_write"] / 1_000_000
            + cache_read_tokens * pricing["cache_read"]   / 1_000_000
        )
        async with self._lock:
            self.total_input_tokens       += input_tokens
            self.total_output_tokens      += output_tokens
            self.total_cache_write_tokens += cache_creation_tokens
            self.total_cache_read_tokens  += cache_read_tokens
            self.total_cost_usd           += cost
            self.calls                    += 1

    def report(self) -> str:
        lines = [
            "─" * 44,
            " Cost Report",
            "─" * 44,
            f"  API calls          : {self.calls}",
            f"  Input tokens       : {self.total_input_tokens:,}",
            f"  Output tokens      : {self.total_output_tokens:,}",
            f"  Cache write tokens : {self.total_cache_write_tokens:,}",
            f"  Cache read tokens  : {self.total_cache_read_tokens:,}",
            f"  Total cost (USD)   : ${self.total_cost_usd:.4f}",
            "─" * 44,
        ]
        return "\n".join(lines)
