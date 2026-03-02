"""
Streamlit Main App

CONCEPT: Streamlit gives you a Python-native UI without needing React/HTML.
Perfect for data science projects and internal tools.

For production at a bank: would use React + Next.js with proper auth (Azure AD SSO).
Streamlit works great for demos, prototypes, and internal analyst tools.

Run with: streamlit run ui/app.py --server.port 8501
"""
import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Financial PDF Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "conversation" not in st.session_state:
    st.session_state.conversation = []


def login_page():
    """Simple login form."""
    st.title("Financial PDF Intelligence Pipeline")
    st.markdown("##### Powered by Claude + LangGraph | Built for Citi")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("---")
        st.subheader("Sign In")

        email = st.text_input("Email", placeholder="analyst@citi.com")
        password = st.text_input("Password", type="password", placeholder="demo1234")

        if st.button("Login", type="primary", use_container_width=True):
            # Demo auth (replace with API call to /auth/login)
            DEMO_USERS = {
                "analyst@citi.com": {"password": "demo1234", "role": "analyst", "name": "Demo Analyst"},
                "admin@citi.com": {"password": "admin1234", "role": "admin", "name": "Admin User"},
            }
            user = DEMO_USERS.get(email)
            if user and user["password"] == password:
                st.session_state.authenticated = True
                st.session_state.user = {"email": email, "name": user["name"], "role": user["role"]}
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials. Try analyst@citi.com / demo1234")

        st.markdown("---")
        st.caption("Demo credentials: analyst@citi.com / demo1234")


def sidebar():
    """Navigation sidebar."""
    with st.sidebar:
        st.markdown(f"**{st.session_state.user['name']}**")
        st.caption(f"Role: {st.session_state.user['role']}")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Upload & Process", "Analysis Dashboard", "Query Engine", "Monitoring"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        if st.session_state.session_id:
            st.success(f"Session active")
            st.caption(f"ID: {st.session_state.session_id[:8]}...")

        if st.button("Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    return page


def upload_page():
    """Document upload and pipeline trigger."""
    st.title("Upload Financial Documents")
    st.markdown("Upload 1-15 annual reports, earnings releases, or trade documents.")

    uploaded_files = st.file_uploader(
        "Select PDF files (max 15, 50MB each)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        run_query_inline = st.checkbox("Also run a query after processing?")
    with col2:
        if run_query_inline:
            query = st.text_input("Enter your question", placeholder="What was the total revenue in 2023?")
        else:
            query = None

    if st.button("Process Documents", type="primary", disabled=not uploaded_files):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            return

        if len(uploaded_files) > 15:
            st.error("Maximum 15 files allowed.")
            return

        # Save files locally and run pipeline
        import uuid
        import tempfile
        from pathlib import Path

        session_id = str(uuid.uuid4())
        st.session_state.session_id = session_id

        # Save to temp directory
        temp_dir = Path("./data/pdfs") / session_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        file_paths = []
        for f in uploaded_files:
            path = temp_dir / f.name
            path.write_bytes(f.read())
            file_paths.append(str(path))

        st.info(f"Processing {len(file_paths)} document(s)...")

        progress = st.progress(0, text="Initialising pipeline...")

        try:
            # Import here to avoid circular imports
            from agents.orchestrator import run_pipeline
            from rag.knowledge_base import ingest_documents_batch

            progress.progress(10, text="Ingesting into knowledge base...")
            ingest_documents_batch(file_paths, session_id)

            progress.progress(25, text="Running classification agent...")
            result = run_pipeline(
                document_paths=file_paths,
                task="full_pipeline",
                query=query,
                user_id=st.session_state.user["email"],
                session_id=session_id,
            )
            progress.progress(100, text="Complete!")

            st.session_state.pipeline_result = result
            st.success(f"Pipeline complete! Steps: {', '.join(result.get('completed_steps', []))}")

            if result.get("errors"):
                st.warning(f"Warnings: {result['errors']}")

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.exception(e)


def analysis_page():
    """Display extracted financials and comparison."""
    st.title("Analysis Dashboard")

    result = st.session_state.get("pipeline_result")
    if not result:
        st.info("No analysis results yet. Upload documents on the Upload page first.")
        return

    classifications = result.get("classifications", [])
    extractions = result.get("extractions", [])
    comparison = result.get("comparison", {})
    summary = result.get("summary", "")

    # ── Summary ───────────────────────────────────────────────────────────────
    if summary:
        with st.expander("Executive Summary", expanded=True):
            st.markdown(summary)

    # ── Classification Results ────────────────────────────────────────────────
    if classifications:
        st.subheader("Document Classification")
        import pandas as pd
        clf_data = []
        for clf in classifications:
            clf_data.append({
                "Company": clf.get("company_name", "?"),
                "Type": clf.get("doc_type", "?"),
                "Year": clf.get("fiscal_year", "?"),
                "Period": clf.get("fiscal_period", "?"),
                "Confidence": f"{clf.get('confidence', 0):.0%}",
                "Dual-Use Flag": "🚨 YES" if clf.get("is_dual_use_material") else "✓ No",
            })
        st.dataframe(pd.DataFrame(clf_data), use_container_width=True)

    # ── Financial Metrics Table ───────────────────────────────────────────────
    if extractions:
        st.subheader("Extracted Financial Metrics")

        import pandas as pd
        metrics_rows = []
        metric_labels = {
            "revenue": "Revenue", "net_income": "Net Income", "ebitda": "EBITDA",
            "operating_income": "Operating Income", "eps_diluted": "EPS (Diluted)",
            "total_assets": "Total Assets", "total_debt": "Total Debt",
            "total_equity": "Total Equity", "operating_cash_flow": "Operating Cash Flow",
            "net_margin": "Net Margin", "roe": "ROE", "roa": "ROA",
        }

        for label, key in metric_labels.items():
            row = {"Metric": key}
            for i, ext in enumerate(extractions):
                clf = classifications[i] if i < len(classifications) else {}
                col_name = f"{clf.get('company_name', '?')} {clf.get('fiscal_year', '?')}"
                val = ext.get(label)
                if val is not None:
                    if key in ("Net Margin", "ROE", "ROA"):
                        row[col_name] = f"{val:.1%}"
                    else:
                        row[col_name] = f"{val:,.1f} {ext.get('currency', 'USD')} {ext.get('unit', 'M')}"
                else:
                    row[col_name] = "—"
            metrics_rows.append(row)

        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

    # ── Comparison Charts ─────────────────────────────────────────────────────
    if comparison and comparison.get("yoy_changes"):
        st.subheader("Year-over-Year Comparison")

        import plotly.graph_objects as go

        yoy = comparison["yoy_changes"]
        for metric in ["revenue", "net_income", "ebitda"]:
            if metric not in yoy:
                continue
            series = yoy[metric].get("series", [])
            if not series:
                continue

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{s['company']} {s['year']}" for s in series],
                y=[s["value"] for s in series],
                name=metric.replace("_", " ").title(),
            ))
            fig.update_layout(title=metric.replace("_", " ").title(), height=300)
            st.plotly_chart(fig, use_container_width=True)

        if comparison.get("key_insights"):
            st.subheader("Key Insights")
            for insight in comparison["key_insights"]:
                st.markdown(f"- {insight}")

        if comparison.get("risk_flags"):
            st.subheader("Risk Flags")
            for flag in comparison["risk_flags"]:
                st.error(f"⚠ {flag}")

    # ── Token Usage ───────────────────────────────────────────────────────────
    with st.expander("Token Usage & Cost"):
        in_tok = result.get("total_input_tokens", 0)
        out_tok = result.get("total_output_tokens", 0)
        est_cost = in_tok * 3e-6 + out_tok * 15e-6
        col1, col2, col3 = st.columns(3)
        col1.metric("Input Tokens", f"{in_tok:,}")
        col2.metric("Output Tokens", f"{out_tok:,}")
        col3.metric("Estimated Cost", f"${est_cost:.4f}")


def query_page():
    """RAG query interface."""
    st.title("Query Engine")
    st.markdown("Ask natural language questions about your uploaded documents.")

    if not st.session_state.get("session_id"):
        st.info("Please upload and process documents first.")
        return

    # Display conversation history
    for msg in st.session_state.conversation:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for src in msg["sources"]:
                        st.caption(f"📄 {src.get('filename', '?')} | Page {src.get('page', '?')} | Score: {src.get('relevance_score', 0):.2f}")

    # Input
    if prompt := st.chat_input("Ask about your documents..."):
        st.session_state.conversation.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                try:
                    from rag.retriever import FinancialRetriever
                    from langchain_anthropic import ChatAnthropic
                    from langchain_core.messages import SystemMessage, HumanMessage
                    from config.settings import get_settings

                    cfg = get_settings()
                    retriever = FinancialRetriever()
                    chunks = retriever.retrieve(prompt, st.session_state.session_id, k=5)

                    if not chunks:
                        answer = "No relevant information found in the uploaded documents."
                        sources = []
                    else:
                        context = "\n---\n".join(
                            f"[{c['source']}, p.{c['page']}]\n{c['text']}"
                            for c in chunks[:4]
                        )

                        llm = ChatAnthropic(
                            model=cfg.primary_llm_model,
                            api_key=cfg.anthropic_api_key,
                            max_tokens=600,
                        )
                        response = llm.invoke([
                            SystemMessage(content="Answer only from context. Cite sources [filename, page]."),
                            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"),
                        ])
                        answer = response.content
                        sources = [{"filename": c["source"], "page": c["page"], "relevance_score": c.get("score", 0)} for c in chunks[:4]]

                    st.markdown(answer)

                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.caption(f"📄 {src['filename']} | Page {src['page']} | Score: {src['relevance_score']:.2f}")

                except Exception as e:
                    error_msg = f"Query failed: {str(e)}"
                    st.error(error_msg)
                    st.session_state.conversation.append({
                        "role": "assistant",
                        "content": error_msg,
                    })


def monitoring_page():
    """Metrics dashboard."""
    st.title("Monitoring & Metrics")

    try:
        from monitoring.metrics import metrics
        summary = metrics.get_summary()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Requests", summary.get("total_requests", 0))
        col2.metric("Total Errors", summary.get("total_errors", 0))
        col3.metric("Input Tokens", f"{summary.get('total_input_tokens', 0):,}")
        col4.metric("Est. Cost (USD)", f"${summary.get('estimated_cost_usd', 0):.4f}")

        if summary.get("agent_latencies"):
            import pandas as pd
            latency_data = []
            for agent, stats in summary["agent_latencies"].items():
                latency_data.append({
                    "Agent": agent,
                    "Avg (s)": stats["avg_s"],
                    "P95 (s)": stats["p95_s"],
                    "Calls": stats["count"],
                })
            st.dataframe(pd.DataFrame(latency_data), use_container_width=True)

    except Exception as e:
        st.warning(f"Metrics unavailable: {e}")

    st.markdown("---")
    st.markdown("**LangSmith Tracing**: Enabled via `LANGCHAIN_TRACING_V2=true`")
    st.markdown("**Production**: Add Prometheus + Grafana or AWS CloudWatch")


# ── Main router ───────────────────────────────────────────────────────────────
def main():
    if not st.session_state.authenticated:
        login_page()
        return

    page = sidebar()

    if page == "Upload & Process":
        upload_page()
    elif page == "Analysis Dashboard":
        analysis_page()
    elif page == "Query Engine":
        query_page()
    elif page == "Monitoring":
        monitoring_page()


if __name__ == "__main__":
    main()
