"""
CLI script to run the full pipeline against sample PDFs.
Perfect for testing without starting the UI/API.

Usage:
  python scripts/run_pipeline.py                           # Run on sample PDFs
  python scripts/run_pipeline.py --files doc1.pdf doc2.pdf # Run on specific files
  python scripts/run_pipeline.py --query "What was net income in 2023?"
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()


def print_results(result: dict):
    """Pretty-print pipeline results to the terminal."""

    # Classifications
    if result.get("classifications"):
        console.print("\n[bold cyan]DOCUMENT CLASSIFICATION[/bold cyan]")
        tbl = Table(show_header=True, header_style="bold magenta")
        tbl.add_column("Company")
        tbl.add_column("Type")
        tbl.add_column("Year")
        tbl.add_column("Confidence")
        tbl.add_column("Dual-Use")

        for clf in result["classifications"]:
            tbl.add_row(
                clf.get("company_name", "?"),
                clf.get("doc_type", "?"),
                f"{clf.get('fiscal_year', '?')} {clf.get('fiscal_period', '')}",
                f"{clf.get('confidence', 0):.0%}",
                "[red]YES[/red]" if clf.get("is_dual_use_material") else "[green]No[/green]",
            )
        console.print(tbl)

    # Extractions
    if result.get("extractions"):
        console.print("\n[bold cyan]FINANCIAL EXTRACTION[/bold cyan]")
        for i, ext in enumerate(result["extractions"]):
            clf = result["classifications"][i] if i < len(result.get("classifications", [])) else {}
            console.print(f"\n[bold]{clf.get('company_name', 'Doc ' + str(i+1))} {clf.get('fiscal_year', '')}[/bold]")
            currency = ext.get("currency", "USD")
            unit = ext.get("unit", "M")
            for key in ["revenue", "net_income", "ebitda", "eps_diluted", "total_assets", "total_debt", "roe", "net_margin"]:
                val = ext.get(key)
                if val is not None:
                    if key in ("roe", "roa", "net_margin"):
                        console.print(f"  {key.replace('_', ' ').title():<25} {val:.1%}")
                    else:
                        console.print(f"  {key.replace('_', ' ').title():<25} {val:>12,.1f} {currency} {unit}")

    # Comparison
    if result.get("comparison"):
        comp = result["comparison"]
        if comp.get("key_insights"):
            console.print("\n[bold cyan]KEY INSIGHTS[/bold cyan]")
            for insight in comp["key_insights"]:
                console.print(f"  • {insight}")
        if comp.get("risk_flags"):
            console.print("\n[bold red]RISK FLAGS[/bold red]")
            for flag in comp["risk_flags"]:
                console.print(f"  ⚠  {flag}")

    # Summary
    if result.get("summary"):
        console.print("\n[bold cyan]EXECUTIVE SUMMARY[/bold cyan]")
        console.print(Panel(Markdown(result["summary"]), border_style="cyan"))

    # Query response
    if result.get("query_response"):
        console.print("\n[bold cyan]QUERY RESPONSE[/bold cyan]")
        console.print(Panel(result["query_response"], border_style="green"))

    # Errors
    if result.get("errors"):
        console.print("\n[bold yellow]WARNINGS[/bold yellow]")
        for err in result["errors"]:
            console.print(f"  ⚠  {err}")

    # Token usage
    console.print(
        f"\n[dim]Token usage: {result.get('total_input_tokens', 0):,} in | "
        f"{result.get('total_output_tokens', 0):,} out | "
        f"Steps: {', '.join(result.get('completed_steps', []))}[/dim]"
    )


def main():
    parser = argparse.ArgumentParser(description="Run Financial PDF Pipeline")
    parser.add_argument("--files", nargs="+", help="PDF file paths to process")
    parser.add_argument("--query", type=str, help="Natural language query to run after analysis")
    parser.add_argument("--task", default="full_pipeline",
                        choices=["full_pipeline", "classify", "extract", "query"])
    parser.add_argument("--session", type=str, help="Session ID (optional)")
    args = parser.parse_args()

    # Determine files to process
    if args.files:
        file_paths = []
        for f in args.files:
            p = Path(f)
            if not p.exists():
                console.print(f"[red]File not found: {f}[/red]")
                sys.exit(1)
            file_paths.append(str(p.absolute()))
    else:
        # Use sample PDFs if they exist
        sample_dir = Path("./data/pdfs/samples")
        if not sample_dir.exists() or not list(sample_dir.glob("*.pdf")):
            console.print("[yellow]No sample PDFs found. Generating...[/yellow]")
            from scripts.generate_sample_pdfs import main as gen
            gen()

        # Take first 4 sample files for a quick demo
        file_paths = sorted([str(p) for p in sample_dir.glob("*.pdf")])[:4]
        console.print(f"[green]Using {len(file_paths)} sample PDF(s)[/green]")

    if not file_paths:
        console.print("[red]No PDF files to process[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Financial PDF Pipeline[/bold]")
    console.print(f"Files: {[Path(f).name for f in file_paths]}")
    console.print(f"Task: {args.task}")
    if args.query:
        console.print(f"Query: {args.query}")
    console.print()

    try:
        from agents.orchestrator import run_pipeline
        from rag.knowledge_base import ingest_documents_batch

        with console.status("Ingesting documents into knowledge base..."):
            ingest_results = ingest_documents_batch(file_paths, args.session or "cli-session")
            total_chunks = sum(r["chunks_added"] for r in ingest_results)
            console.print(f"[green]Ingested {total_chunks} chunks[/green]")

        with console.status("Running multi-agent pipeline..."):
            result = run_pipeline(
                document_paths=file_paths,
                task=args.task,
                query=args.query,
                user_id="cli_user",
                session_id=args.session or "cli-session",
            )

        print_results(result)

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Make sure you have installed requirements: pip install -r requirements.txt[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
