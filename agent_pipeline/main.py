"""
Frugal Agentic Pipeline — entry point.

Usage:
  # Synthetic data (free — validates pipeline structure before spending tokens)
  python main.py --sample

  # Real data
  python main.py --task "Flag anomalies in trade finance documents" \
                 --docs ./my_docs --dataset ./data.csv

  # Tune cost/quality trade-off
  python main.py --sample --max-depth 3 --confidence 0.75
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow running from within the agent_pipeline/ directory
sys.path.insert(0, os.path.dirname(__file__))

from config import ANTHROPIC_API_KEY, MAX_RECURSION_DEPTH, CONFIDENCE_THRESHOLD
from data.sample_gen import generate_sample_context
from orchestrator import final_report, run_pipeline


def _load_real_context(docs_path: str | None, dataset_path: str | None) -> str:
    parts: list[str] = []

    if docs_path and os.path.isdir(docs_path):
        import glob
        files = glob.glob(os.path.join(docs_path, "**", "*.txt"), recursive=True)
        files += glob.glob(os.path.join(docs_path, "**", "*.md"),  recursive=True)
        for f in files[:10]:  # cap at 10 docs to match scenario
            try:
                text = open(f).read()[:2000]
                parts.append(f"=== {os.path.basename(f)} ===\n{text}")
            except OSError:
                pass

    if dataset_path and os.path.isfile(dataset_path):
        try:
            import csv
            rows = []
            with open(dataset_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for i, row in enumerate(reader):
                    if i >= 10_000:
                        break
                    rows.append(row)
            n = len(rows)
            parts.append(
                f"=== DATASET ({n:,} rows) ===\n"
                f"Headers: {list(rows[0].keys()) if rows else []}\n"
                f"Sample (first 5 rows): {rows[:5]}"
            )
        except Exception as e:
            parts.append(f"=== DATASET (load error: {e}) ===")

    return "\n\n".join(parts) if parts else ""


async def main() -> None:
    parser = argparse.ArgumentParser(description="Frugal Agentic Pipeline")
    parser.add_argument("--sample",     action="store_true",
                        help="Use synthetic data (no API key needed for structure validation)")
    parser.add_argument("--task",       default="Analyse the documents and dataset. "
                                                 "Extract key findings, identify anomalies, "
                                                 "and assess risk.",
                        help="Task description for all agents")
    parser.add_argument("--docs",       default=None,  help="Path to documents directory")
    parser.add_argument("--dataset",    default=None,  help="Path to CSV dataset")
    parser.add_argument("--max-depth",  type=int,   default=MAX_RECURSION_DEPTH)
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Wall-clock time limit in seconds (default 1800 = 30 min)")
    args = parser.parse_args()

    # --- Context assembly ---
    if args.sample:
        context = generate_sample_context(n_docs=10, n_rows=10_000)
        print("Using synthetic sample data (10 docs, 10k rows).\n")
    else:
        if not ANTHROPIC_API_KEY:
            print("ERROR: ANTHROPIC_API_KEY not set. Run with --sample to test without an API key.")
            sys.exit(1)
        context = _load_real_context(args.docs, args.dataset)
        if not context:
            print("ERROR: No context loaded. Provide --docs and/or --dataset, or use --sample.")
            sys.exit(1)

    # --- Dry-run structure check (no API calls) ---
    if args.sample and not ANTHROPIC_API_KEY:
        print("=== DRY RUN (no API key) — showing pipeline structure only ===\n")
        print(f"Task : {args.task}")
        print(f"Context length : {len(context)} chars")
        print(f"Agents : 10  |  Max rounds : {args.max_depth}  |  "
              f"Confidence target : {args.confidence:.0%}")
        print(f"Time limit : {args.time_limit}s\n")
        print("Context preview:")
        print(context[:800])
        print("\n[Set ANTHROPIC_API_KEY to run the full pipeline.]\n")
        return

    # --- Full pipeline run ---
    state, tracker = await run_pipeline(
        task=args.task,
        context=context,
        max_depth=args.max_depth,
        confidence_threshold=args.confidence,
        time_limit_seconds=args.time_limit,
    )

    print(final_report(state))
    print(tracker.report())


if __name__ == "__main__":
    asyncio.run(main())
