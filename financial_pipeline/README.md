# Financial PDF Intelligence Pipeline

> Production-grade multi-agent AI system for financial document processing.
> Built with Claude + LangGraph + ChromaDB. Designed to demonstrate end-to-end
> agentic AI development for interview preparation.

---

## Quick Start (5 minutes)

### 1. Install dependencies

```bash
cd financial_pipeline
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Generate sample financial PDFs (12 documents, 4 companies x 3 years)

```bash
python scripts/generate_sample_pdfs.py
```

### 4. Run the pipeline CLI (no UI needed)

```bash
python scripts/run_pipeline.py
# or with a query:
python scripts/run_pipeline.py --query "What was GlobalBank Corp's net income in 2023?"
# or on your own PDFs:
python scripts/run_pipeline.py --files path/to/report1.pdf path/to/report2.pdf
```

### 5. Launch the Streamlit UI

```bash
streamlit run ui/app.py
# Open http://localhost:8501
# Login: analyst@citi.com / demo1234
```

### 6. Launch the API (optional)

```bash
uvicorn api.main:app --reload
# Open http://localhost:8000/docs for interactive API docs
```

---

## What This Pipeline Does

| Feature | Description |
|---------|-------------|
| **Classification** | Auto-detects doc type (annual report, earnings release, balance sheet), company name, fiscal year, dual-use flag |
| **Extraction** | Pulls structured financials: Revenue, Net Income, EBITDA, EPS, Balance Sheet, Cash Flows, Ratios |
| **Comparison** | Year-over-year analysis across multiple documents with trend detection |
| **Summarization** | Executive summary with insights and risk flags |
| **RAG Query Engine** | Ask natural language questions over all uploaded documents |
| **Dual-Use Detection** | Flags trade documents with potential export-control goods |

---

## Architecture Summary

```
PDF Upload → Guardrail → Classifier → Extractor → Comparator → Summarizer
                                                                     │
                                          ChromaDB ← Embed & Index ──┘
                                              │
                                    RAG Query Engine → Answer + Citations
```

Full architecture diagram: [docs/architecture.md](docs/architecture.md)

---

## Project Structure

```
financial_pipeline/
├── agents/                 # LangGraph multi-agent system
│   ├── state.py           # Shared TypedDict state
│   ├── orchestrator.py    # LangGraph graph definition + runner
│   ├── classifier.py      # Document classification agent
│   ├── extractor.py       # Financial data extraction agent
│   ├── comparator.py      # Year-over-year comparison agent
│   ├── summarizer.py      # Executive summary agent
│   └── query_agent.py     # RAG query agent
├── tools/
│   ├── pdf_loader.py      # PDF text + table extraction
│   └── guardrails.py      # Input/output validation
├── rag/
│   ├── embeddings.py      # Local sentence-transformers
│   ├── knowledge_base.py  # ChromaDB vector store management
│   └── retriever.py       # Hybrid retrieval (vector + BM25 + RRF)
├── api/                   # FastAPI backend
│   ├── main.py            # App, CORS, rate limiting, middleware
│   ├── auth.py            # JWT authentication
│   └── routes/            # Upload, query, auth endpoints
├── ui/
│   └── app.py             # Streamlit multi-page UI
├── monitoring/
│   ├── metrics.py         # Prometheus + in-memory metrics
│   └── evaluation.py      # RAGAS + LLM-as-a-judge evaluation
├── scripts/
│   ├── generate_sample_pdfs.py   # Create 12 test PDFs
│   └── run_pipeline.py           # CLI runner
└── docs/
    ├── architecture.md    # Full system architecture
    └── interview_qa.md    # Interview preparation Q&A
```

---

## Interview Preparation

See [docs/interview_qa.md](docs/interview_qa.md) for:
- 13 detailed Q&A covering all core concepts
- Your personal "project story" (90-second pitch)
- Technical flashcard table (30 concepts)
- Questions to ask the interviewer

---

## Deployment

### Local Development
```bash
# API only
uvicorn api.main:app --reload

# UI only
streamlit run ui/app.py

# Both via Docker
docker-compose up
```

### AWS Production
See [docs/architecture.md](docs/architecture.md) for the full AWS architecture.
Key services: Bedrock (Claude), OpenSearch (vector), ECS Fargate (compute), Cognito (auth).

---

## Demo Credentials (local only)

| Email | Password | Role |
|-------|----------|------|
| analyst@citi.com | demo1234 | analyst |
| admin@citi.com | admin1234 | admin |

---

## Key Technologies

- **LLM**: Claude (Anthropic) via `langchain-anthropic`
- **Agent Framework**: LangGraph (stateful multi-agent graphs)
- **Vector DB**: ChromaDB (local), Amazon OpenSearch (production)
- **Embeddings**: `all-MiniLM-L6-v2` via sentence-transformers (local, free)
- **PDF Processing**: pdfplumber + PyPDF
- **API**: FastAPI + Uvicorn
- **UI**: Streamlit
- **Auth**: JWT (python-jose + passlib)
- **Monitoring**: LangSmith + Prometheus + Loguru
- **Evaluation**: RAGAS-style metrics + LLM-as-a-judge
