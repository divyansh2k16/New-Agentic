"""
Async Anthropic client with prompt caching.

The key frugality lever: mark the system-prompt block with
  cache_control={"type": "ephemeral"}
On the first call Anthropic writes the cache (1.25x input cost).
Every subsequent call reads from cache at 0.1x input cost — 90% savings
for the same system-prompt bytes re-sent across all agents in every round.
"""
from __future__ import annotations

import asyncio
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, HAIKU, PRICING
from monitoring.cost_tracker import CostTracker


_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


async def call(
    *,
    system_prompt: str,
    user_message: str,
    model: str = HAIKU,
    max_tokens: int = 512,
    tracker: CostTracker | None = None,
) -> str:
    """
    Single async LLM call with ephemeral prompt caching on the system block.

    Returns the text content of the first content block.
    Side-effect: updates tracker with token usage if provided.
    """
    response = await get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                # Pins this block to the prompt cache.
                # Cost: first call = cache_write rate, all subsequent = cache_read rate.
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )

    if tracker is not None:
        usage = response.usage
        tracker.record(
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
        )

    return response.content[0].text
