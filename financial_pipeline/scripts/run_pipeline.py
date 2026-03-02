"""
CLI script to run the full pipeline against your own PDF files.

DROP YOUR PDFs:  data/pdfs/my_documents/   ← copy files there, then just run this script

Usage:
  python scripts/run_pipeline.py                               # all PDFs in my_documents/
  python scripts/run_pipeline.py --folder path/to/folder       # different folder
  python scripts/run_pipeline.py --files report1.pdf rep2.pdf  # explicit file list
  python scripts/run_pipeline.py --query "What was net income in 2023?"
  python scripts/run_pipeline.py --limit 5                     # process first 5 only
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

MY_DOCUMENTS_DIR = Path("./data/pdfs/my_documents")


def discover_pdfs(folder: Path, limit: int = None) -> list:
    """Return sorted list of PDF paths from a folder, ignoring non-PDF files."""
    pdfs = sorted([p for p in folder.glob("*.pdf") if p.is_file()])
    if not pdfs:
        return []
    if limit:
        pdfs = pdfs[:limit]
    return [str(p.absolute()) for p in pdfs]


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
    parser = argparse.ArgumentParser(
        description="Financial PDF Pipeline — drop your PDFs in data/pdfs/my_documents/ and run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --limit 3
  python scripts/run_pipeline.py --query "What was the net income in 2023?"
  python scripts/run_pipeline.py --folder ~/Downloads/annual_reports
  python scripts/run_pipeline.py --files ~/docs/apple_2023.pdf ~/docs/apple_2022.pdf
        """,
    )
    parser.add_argument("--files", nargs="+", help="Explicit PDF file paths")
    parser.add_argument("--folder", type=str, help="Folder to scan for PDFs (default: data/pdfs/my_documents)")
    parser.add_argument("--limit", type=int, default=15, help="Max PDFs to process (default: 15)")
    parser.add_argument("--query", type=str, help="Natural language query to run after analysis")
    parser.add_argument("--task", default="full_pipeline",
                        choices=["full_pipeline", "classify", "extract", "query"])
    parser.add_argument("--session", type=str, help="Session ID (optional, auto-generated if omitted)")
    args = parser.parse_args()

    # ── Resolve file list ─────────────────────────────────────────────────────
    if args.files:
        # Explicit files passed on command line
        file_paths = []
        for f in args.files:
            p = Path(f).expanduser().resolve()
            if not p.exists():
                console.print(f"[red]File not found: {f}[/red]")
                sys.exit(1)
            if p.suffix.lower() != ".pdf":
                console.print(f"[yellow]Skipping non-PDF: {f}[/yellow]")
                continue
            file_paths.append(str(p))

    elif args.folder:
        # Explicit folder passed
        folder = Path(args.folder).expanduser().resolve()
        if not folder.exists():
            console.print(f"[red]Folder not found: {folder}[/red]")
            sys.exit(1)
        file_paths = discover_pdfs(folder, limit=args.limit)

    else:
        # Default: look in data/pdfs/my_documents/
        MY_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        file_paths = discover_pdfs(MY_DOCUMENTS_DIR, limit=args.limit)

        if not file_paths:
            console.print(Panel(
                "[bold yellow]No PDFs found in data/pdfs/my_documents/[/bold yellow]\n\n"
                "Copy your financial PDF files into:\n"
                f"  [cyan]{MY_DOCUMENTS_DIR.absolute()}[/cyan]\n\n"
                "Then re-run this script.\n\n"
                "Or pass files directly:\n"
                "  [dim]python scripts/run_pipeline.py --files report1.pdf report2.pdf[/dim]",
                title="No Documents Found",
                border_style="yellow",
            ))
            sys.exit(0)

    if not file_paths:
        console.print("[red]No valid PDF files found.[/red]")
        sys.exit(1)

    # ── Summary before running ────────────────────────────────────────────────
    console.print(f"\n[bold]Financial PDF Pipeline[/bold]")
    console.print(f"Found [cyan]{len(file_paths)}[/cyan] PDF(s) to process:\n")
    for p in file_paths:
        size_kb = Path(p).stat().st_size // 1024
        console.print(f"  [dim]•[/dim] {Path(p).name}  [dim]({size_kb} KB)[/dim]")
    if args.query:
        console.print(f"\nQuery: [green]{args.query}[/green]")
    console.print()

    # ── Run ───────────────────────────────────────────────────────────────────
    import uuid
    session_id = args.session or str(uuid.uuid4())[:8]

    try:
        from agents.orchestrator import run_pipeline
        from rag.knowledge_base import ingest_documents_batch

        with console.status("[cyan]Embedding documents into knowledge base...[/cyan]"):
            ingest_results = ingest_documents_batch(file_paths, session_id)
            new_chunks = sum(r["chunks_added"] for r in ingest_results)
            skipped = sum(1 for r in ingest_results if r.get("skipped"))

        if skipped:
            console.print(f"[dim]Skipped {skipped} already-indexed doc(s) (cached)[/dim]")
        console.print(f"[green]Ready: {new_chunks} new chunks embedded[/green]")

        with console.status("[cyan]Running multi-agent pipeline...[/cyan]"):
            result = run_pipeline(
                document_paths=file_paths,
                task=args.task,
                query=args.query,
                user_id="cli_user",
                session_id=session_id,
            )

        print_results(result)

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Run: pip install -r requirements.txt[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Pipeline failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
