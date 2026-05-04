"""
Central config: models, pricing, recursion limits.
Haiku is the default; escalate to Sonnet only for high-complexity nodes.
"""
import os

# Model IDs
HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

# Per-million token pricing (input / cache_write / cache_read / output)
PRICING = {
    HAIKU:  {"input": 0.80, "cache_write": 1.00, "cache_read": 0.08, "output": 4.00},
    SONNET: {"input": 3.00, "cache_write": 3.75, "cache_read": 0.30, "output": 15.00},
}

# Orchestrator defaults
MAX_RECURSION_DEPTH = 5
CONFIDENCE_THRESHOLD = 0.80   # stop recursing once all agents exceed this

# Agent-level token budgets (keeps per-call cost predictable)
TOKEN_BUDGETS = {
    "classifier":  256,
    "extractor":   512,
    "analyst":     768,
    "validator":   256,
    "anomaly":     512,
    "router":      128,
    "summarizer":  512,
    "comparator":  512,
    "risk":        512,
    "formatter":   768,
}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
