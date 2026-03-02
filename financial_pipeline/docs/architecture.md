# Financial PDF Pipeline — Architecture Guide

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                            │
│  ┌──────────────────┐              ┌───────────────────────────────┐   │
│  │  Streamlit UI    │              │    FastAPI REST API            │   │
│  │  (localhost:8501)│              │    (localhost:8000)            │   │
│  │  - Login         │              │    - POST /auth/login          │   │
│  │  - Upload PDFs   │              │    - POST /documents/upload    │   │
│  │  - View Analysis │              │    - GET  /documents/status/id │   │
│  │  - Chat Q&A      │              │    - POST /query/ask           │   │
│  └────────┬─────────┘              └──────────────┬────────────────┘   │
└───────────┼──────────────────────────────────────┼─────────────────────┘
            │                                       │
            └─────────────────┬─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      AGENT ORCHESTRATION LAYER (LangGraph)              │
│                                                                         │
│   START → [GUARDRAIL] ──reject──► END                                  │
│                │                                                        │
│                ▼                                                        │
│         [CLASSIFIER]  ──── Haiku model (fast/cheap)                    │
│                │           • Document type                              │
│                │           • Company name                               │
│                │           • Fiscal year                                │
│                │           • Dual-use flag                              │
│                ▼                                                        │
│         [EXTRACTOR]   ──── Sonnet model (capable)                      │
│                │           • Revenue, Net Income, EBITDA                │
│                │           • Balance sheet items                        │
│                │           • Cash flows                                 │
│                │           • Computed ratios                            │
│                ▼                                                        │
│       ┌────────┴──────────┐                                            │
│    >1 doc              1 doc                                            │
│       ▼                   ▼                                            │
│  [COMPARATOR]      [SUMMARIZER]  ──── Sonnet model                    │
│       │                │               • Executive summary              │
│       └───────┬─────────┘               • Structured sections          │
│               ▼                                                        │
│         [QUERY AGENT]  ──── Sonnet model (RAG-powered)                │
│               │               • Hybrid retrieval                       │
│               │               • Context window mgmt                    │
│               ▼               • Citation                               │
│              END                                                       │
└─────────────────────────────────────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
┌───────────────────────┐            ┌──────────────────────────────────┐
│    RAG PIPELINE       │            │         LLM API LAYER            │
│                       │            │                                  │
│  ┌─────────────────┐  │            │  Claude Haiku  ──► Classification│
│  │ PDF Loader      │  │            │  Claude Sonnet ──► Extraction    │
│  │ (pdfplumber)    │  │            │                ──► Summarisation │
│  └────────┬────────┘  │            │                ──► Query         │
│           ▼           │            └──────────────────────────────────┘
│  ┌─────────────────┐  │
│  │ Text Chunker    │  │            ┌──────────────────────────────────┐
│  │ (500 chars,     │  │            │       MONITORING LAYER           │
│  │  50 overlap)    │  │            │                                  │
│  └────────┬────────┘  │            │  LangSmith ──► LLM traces        │
│           ▼           │            │  Prometheus ──► metrics          │
│  ┌─────────────────┐  │            │  RAGAS ──► RAG evaluation        │
│  │ Embeddings      │  │            │  Loguru ──► structured logs      │
│  │ (MiniLM local)  │  │            └──────────────────────────────────┘
│  └────────┬────────┘  │
│           ▼           │
│  ┌─────────────────┐  │
│  │ ChromaDB        │  │
│  │ Vector Store    │  │
│  │ (persistent)    │  │
│  └────────┬────────┘  │
│           │           │
│  Hybrid Retrieval:    │
│  Vector + BM25 + RRF  │
└───────────────────────┘
```

---

## Data Flow — Step by Step

### Step 1: Document Upload
1. User uploads 1-15 PDFs via Streamlit or API
2. Files saved to `./data/pdfs/{session_id}/`
3. Guardrail validation: file type, size, magic bytes, PII scan
4. Background job created (job_id returned immediately)

### Step 2: Knowledge Base Ingestion (for RAG)
1. `pdfplumber` extracts text page-by-page with table detection
2. Text chunked into 500-char segments with 50-char overlap
3. `sentence-transformers/all-MiniLM-L6-v2` generates embeddings locally
4. Chunks + embeddings + metadata stored in ChromaDB
5. Deduplication via SHA-256 file hash (skip if already indexed)

### Step 3: Classification Agent
- Model: `claude-haiku-4-5` (fast, cheap)
- Input: First 2,000 chars of document
- Output: doc_type, company_name, fiscal_year, dual_use_flag
- Technique: Few-shot prompting + structured JSON output

### Step 4: Extraction Agent
- Model: `claude-sonnet-4-6` (more capable)
- Input: 8,000 chars + extracted tables
- Output: Full financial metrics (revenue, net income, EBITDA, ratios, etc.)
- Technique: Table-aware prompting, computed ratios in Python (no LLM)

### Step 5: Comparison Agent
- YoY math: Pure Python (no LLM), saves tokens
- Narrative insights: claude-haiku from compact summary
- Output: YoY changes dict, key insights, risk flags

### Step 6: Summarizer Agent
- Model: claude-sonnet-4-6
- Input: Structured outputs from all prior agents
- Output: 400-600 word executive summary with structured sections

### Step 7: Query Agent (RAG)
- Retrieval: Hybrid (vector + BM25 + RRF fusion)
- Context: Top-5 chunks, max 6,000 chars
- Memory: Last 3 conversation turns
- Model: claude-sonnet-4-6 with strict grounding prompt

---

## AWS Production Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AWS CLOUD                                  │
│                                                                         │
│  ┌──────────┐    ┌────────────┐    ┌──────────┐    ┌────────────────┐ │
│  │CloudFront│───►│ API Gateway│───►│  Cognito  │    │ WAF + Shield   │ │
│  │  (CDN)   │    │ + Lambda   │    │  (Auth)   │    │ (Security)     │ │
│  └──────────┘    └─────┬──────┘    └──────────┘    └────────────────┘ │
│                        │                                               │
│                        ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    VPC (Private Subnet)                          │  │
│  │                                                                  │  │
│  │  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐   │  │
│  │  │  ALB         │───►│ ECS Fargate   │───►│ Amazon Bedrock │   │  │
│  │  │  (Load Bal.) │    │ (API + Agents)│    │ (Claude API)   │   │  │
│  │  └──────────────┘    └───────┬───────┘    └────────────────┘   │  │
│  │                              │                                   │  │
│  │  ┌─────────┐    ┌────────────▼──────────┐    ┌──────────────┐  │  │
│  │  │ S3      │    │    SQS Queue           │    │ OpenSearch   │  │  │
│  │  │ (Docs)  │───►│ (async job queue)      │    │ (Vector DB)  │  │  │
│  │  └─────────┘    └───────────────────────┘    └──────────────┘  │  │
│  │                                                                  │  │
│  │  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐   │  │
│  │  │ RDS Postgres │    │ ElastiCache     │    │ CloudWatch   │   │  │
│  │  │ (User/jobs)  │    │ Redis (cache)   │    │ + X-Ray      │   │  │
│  │  └──────────────┘    └─────────────────┘    └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key AWS Services Mapping
| Local Dev | AWS Production | Purpose |
|-----------|---------------|---------|
| ChromaDB | Amazon OpenSearch | Vector storage + search |
| sqlite | RDS PostgreSQL | User/session/job data |
| In-memory dict | ElastiCache Redis | Session state, caching |
| Local files | S3 + KMS | Encrypted document storage |
| BackgroundTasks | SQS + ECS | Async job processing |
| Direct API | Amazon Bedrock | Claude API (private endpoint) |
| Loguru | CloudWatch Logs | Centralised logging |
| Prometheus | CloudWatch Metrics | Monitoring + alerting |
| LangSmith | LangSmith (cloud) | LLM tracing |
| JWT (local) | Cognito | Auth + MFA + SSO |

---

## Technology Choices — Why Each Was Selected

| Technology | Why Chosen | Alternative |
|-----------|-----------|-------------|
| LangGraph | Stateful agent graphs with built-in checkpointing | LangChain LCEL (simpler but no state) |
| Claude API | Best reasoning, 200K context, structured output | GPT-4o (comparable) |
| ChromaDB | Zero-config local vector store with persistence | Qdrant, Weaviate, FAISS |
| sentence-transformers | Free local embeddings, no API cost, GDPR-friendly | Voyage AI, OpenAI embeddings |
| pdfplumber | Best table extraction from financial documents | PyPDF (simpler), PyMuPDF (faster) |
| FastAPI | Async, auto-docs, Pydantic integration | Flask (no async), Django (too heavy) |
| Streamlit | Python-native UI, no React needed | Gradio, React+NextJS |
| Hybrid RAG | Better than pure vector for specific numbers | Pure vector search |
| JWT | Stateless, scalable, standard | Sessions (stateful, harder to scale) |

---

## Key Design Patterns Used

1. **State machine pattern** (LangGraph): Each agent is a node; transitions are edges
2. **Repository pattern** (knowledge_base.py): Abstracts ChromaDB details from agents
3. **Factory pattern** (make_initial_state): Ensures consistent state initialisation
4. **Singleton pattern** (model loading, ChromaDB client): Load once, reuse
5. **Fail fast** (guardrail first): Validate inputs before spending on LLM calls
6. **Graceful degradation**: Single agent failure doesn't crash entire pipeline
7. **Separation of concerns**: Math in Python, language in LLM
8. **Deduplication**: Hash-based to prevent re-processing same document
