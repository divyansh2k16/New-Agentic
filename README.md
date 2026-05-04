# Agentic AI Systems — Learning Repository

Two production-grade agentic systems built to learn and demonstrate how multi-agent
pipelines work, how to control cost, and how to reason about design tradeoffs.

---

## Repository Structure

```
Agentic/
├── agent_pipeline/        # Frugal async system — 10 parallel agents, prompt caching
└── financial_pipeline/    # Production LangGraph system — trade finance document intelligence
```

---

## System 1: `agent_pipeline/` — Frugal Async Agentic System

### What It Does

A general-purpose multi-agent pipeline that analyses documents and datasets recursively.
Ten specialised agents run in parallel each round, share findings, and converge toward a
high-confidence collective result. Designed to be as cheap as possible while remaining
real-time.

### Architecture

```
                        ┌─────────────────────────────────────┐
                        │           ORCHESTRATOR               │
                        │  asyncio.gather → all 10 at once    │
                        │  confidence check → recurse/stop     │
                        │  max_depth + time_limit guards       │
                        └──────────────┬──────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │            parallel asyncio.gather                      │
    ┌─────▼─────┐  ┌────────▼───┐  ┌──▼──────┐  ┌────────▼──────┐ │
    │Classifier │  │ Extractor  │  │Validator│  │  Anomaly Det. │ │
    │  (Haiku)  │  │  (Haiku)   │  │ (Haiku) │  │   (Haiku)     │ │
    └───────────┘  └────────────┘  └─────────┘  └───────────────┘ │
    ┌───────────┐  ┌────────────┐  ┌─────────┐  ┌───────────────┐ │
    │  Router   │  │ Summarizer │  │Comparator│ │  Risk Assessor│ │
    │  (Haiku)  │  │  (Haiku)   │  │ (Haiku) │  │   (Sonnet)    │ │
    └───────────┘  └────────────┘  └─────────┘  └───────────────┘ │
    ┌───────────┐  ┌────────────┐                                   │
    │  Analyst  │  │ Formatter  │                                   │
    │  (Sonnet) │  │  (Haiku)   │◄──────────────────────────────────┘
    └───────────┘  └────────────┘

    Round N findings feed as context into Round N+1 (iterative refinement)
    Loop exits: confidence ≥ 80%  OR  depth ≥ 5  OR  wall time ≥ 30 min
```

### The Three Frugality Levers

#### 1. Model Routing — Haiku by Default
Most agents use `claude-haiku-4-5` ($0.80/$4.00 per million tokens). Only two nodes
escalate to `claude-sonnet-4-6` ($3.00/$15.00) where reasoning quality actually matters:

| Agent | Model | Why |
|---|---|---|
| Classifier, Extractor, Validator, Anomaly, Router, Summarizer, Comparator, Formatter | Haiku | Pattern matching, extraction, formatting — fast and cheap |
| Analyst | Sonnet | Cross-document pattern reasoning requires deeper inference |
| Risk | Sonnet | Compliance risk assessment needs higher accuracy |

**Rule of thumb:** escalate only when an incorrect output from the cheaper model would
propagate downstream and corrupt the final result.

#### 2. Prompt Caching — 90% off Repeated Input Tokens
Every agent's system prompt is marked with `cache_control: {"type": "ephemeral"}` in
`llm_client.py`. The mechanics:

```
Round 1, Call 1:  cache WRITE — 1.25× input rate  (one-time)
Round 1, Call 2+: cache READ  — 0.10× input rate  (all subsequent calls)
Rounds 2-5:       cache READ  — 0.10× input rate  (same cache, still warm)
```

In a 10-agent × 5-round loop = 50 total calls. The system prompt bytes are paid at
full rate exactly once. The other 49 calls pay 10% of that cost. For a 500-token
system prompt this alone saves ~$0.002 per run — trivial per run, significant at scale.

```python
# llm_client.py — how caching is applied
system=[{
    "type": "text",
    "text": system_prompt,
    "cache_control": {"type": "ephemeral"},   # ← this line does all the work
}]
```

#### 3. Parallel Execution — Latency = Slowest Agent, Not Sum
Without `asyncio.gather`: 10 agents × 1.5s each = **15s per round × 5 rounds = 75s**.
With `asyncio.gather`: all 10 fire simultaneously = **1.5s per round × 5 rounds = 7.5s**.

```python
# orchestrator.py
results = await asyncio.gather(*[_safe_run(a) for a in agents])
```

This is why Batch API was rejected for this use case — Batch gives 50% cost savings
but jobs complete asynchronously over up to 24 hours, making it incompatible with
real-time recursive loops.

### Cost Estimate (10 docs, 10k rows, 10 agents, 5 rounds)

| Approach | Cost |
|---|---|
| Sonnet, no optimisation | ~$4.00 |
| Haiku, no optimisation | ~$1.20 |
| Haiku + prompt caching | ~$0.40–0.80 |
| Batch API (not suitable — high latency) | ~$0.20 but 24h delay |

### Running It

```bash
cd agent_pipeline
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...

# Free dry-run — validates pipeline structure without spending tokens
python main.py --sample

# With real data
python main.py --task "Flag anomalies in trade finance documents" \
               --docs ./docs --dataset ./data.csv

# Tune the cost/quality tradeoff
python main.py --sample --max-depth 3 --confidence 0.75 --time-limit 600
```

### Key Files

| File | Purpose |
|---|---|
| `config.py` | Model names, pricing table, token budgets per agent |
| `llm_client.py` | Async Anthropic client — all caching logic lives here |
| `orchestrator.py` | Main loop — parallel gather, confidence check, recursion |
| `agents/base.py` | Abstract base — shared `run()`, JSON parsing, fallback |
| `monitoring/cost_tracker.py` | Thread-safe token/cost accumulator across all agents |
| `data/sample_gen.py` | Synthetic 10-doc + 10k-row data (no API key needed) |

---

## System 2: `financial_pipeline/` — Production LangGraph Trade Finance System

### What It Does

An enterprise-grade document intelligence pipeline for processing trade finance PDFs
(Letters of Credit, Bills of Lading, Invoices, etc.). Built with LangGraph for explicit
state management, conditional routing, and RAG-powered Q&A over uploaded documents.

### Architecture

```
PDF Documents
     │
     ▼
[GUARDRAIL CHECK] ── rejected ──► END
     │ passed
     ▼
[CLASSIFIER]          Identifies doc type, extracts metadata
     │
     ▼
[EXTRACTOR]           Pulls entities, amounts, dates, parties
     │
     ├── multiple docs ──► [COMPARATOR] ──► finds contradictions/gaps
     │                          │
     └── single doc ────────────┘
                                │
                                ▼
                          [SUMMARIZER]     Builds executive summary
                                │
                    ┌───────────┴──────────┐
                 query?                  no query
                    │                       │
                    ▼                       ▼
             [QUERY AGENT]               END
            RAG over docs
                    │
                    ▼
                   END
```

### Why LangGraph

LangGraph was chosen over plain function calls for four reasons:

1. **State persistence** — `MemorySaver` checkpointer stores pipeline state between
   steps. If the process crashes mid-run, it can resume from the last completed node.
2. **Conditional routing** — edges can branch based on state (e.g., single vs multiple
   documents take different paths) without if/else spaghetti.
3. **Typed state** — `FinancialPipelineState` is a typed dict. Every agent reads from
   and writes to a shared, validated state object.
4. **Observability** — LangSmith tracing is one env var away. Every node, token count,
   and routing decision is captured automatically.

### RAG (Retrieval-Augmented Generation)

The query agent uses ChromaDB as a vector store with `all-MiniLM-L6-v2` embeddings.
Documents are chunked, embedded, and stored on first ingestion. Subsequent queries
retrieve the top-k relevant chunks and pass them as context to the LLM — preventing
hallucination on document-specific questions.

### Provider Flexibility

The pipeline is provider-agnostic via `config/llm_factory.py`. Set `LLM_PROVIDER` to
switch between Anthropic and OpenAI without changing any agent code:

```bash
LLM_PROVIDER=anthropic  # claude-sonnet-4-6 / claude-haiku-4-5
LLM_PROVIDER=openai     # gpt-4o / gpt-4o-mini
```

### Running It

```bash
cd financial_pipeline
cp .env.example .env   # add your API key
pip install -r requirements.txt
python scripts/run_pipeline.py
# or with Docker:
docker-compose up
```

---

## Concepts Explained for Interviews

### What is an Agentic System?

A system where an LLM is given a task and the autonomy to decide *how* to complete it —
calling tools, routing to other agents, and iterating until a goal is met. The key
difference from a simple LLM call: **the model influences its own execution path**.

### Agent vs Chain vs Pipeline

| Pattern | Control flow | When to use |
|---|---|---|
| Chain | Fixed, sequential | Known, deterministic steps |
| Pipeline | Fixed with conditional branches | Multi-step with known decision points |
| Agent | Dynamic — model decides next step | Open-ended tasks, tool use |
| Multi-agent | Multiple models collaborating | Tasks that benefit from specialisation |

### Why Parallel Agents?

Each agent is a specialist. Running them concurrently means:
- No single agent's blind spot blocks the result
- Findings from one agent (e.g., anomaly detection) can be weighted against another
  (e.g., validation) in the aggregation step
- Wall time stays constant regardless of agent count

### The Confidence Loop — Why Recurse?

A single LLM pass may miss context or produce a low-confidence finding. Feeding all
agents' round-1 outputs back as context for round-2 lets agents revise their analysis
with the benefit of other agents' perspectives — similar to how a team debriefs and
refines conclusions. Recursion stops when average confidence ≥ 80%, avoiding wasted
compute once quality is sufficient.

### Why Not Batch API for Real-Time Loops?

Batch API offers 50% cost savings but processes jobs asynchronously — results arrive in
up to 24 hours. For a recursive agent loop that needs round-N outputs to construct
round-N+1 inputs, this is a hard blocker. Batch is appropriate for non-interactive
workloads: bulk document processing, nightly analytics, evaluation pipelines.

### Prompt Caching — The Most Underused Cost Lever

Every LLM call in a multi-agent system re-sends the system prompt. For a 10-agent ×
5-round loop that's 50 identical system-prompt transmissions. `cache_control: ephemeral`
pins that content server-side after the first call. The 49 subsequent reads cost 10% of
normal input pricing. At scale (50,000 docs/year) this can reduce annual LLM spend by
60-70% compared to uncached equivalents.

### How to Choose Model Size

Ask: "What is the cost of a wrong answer from the cheaper model?"

- **Classification, extraction, routing, formatting** → wrong answers are visible,
  easily caught, low propagation risk → **Haiku**
- **Risk assessment, pattern analysis, compliance reasoning** → wrong answers propagate
  silently into downstream decisions, high business impact → **Sonnet**
- **Opus** → reserved for multi-step reasoning where Sonnet demonstrably fails;
  at 6× Haiku's cost it needs explicit justification

---

## Interview Questions This Codebase Answers

**"How do you prevent runaway costs in an agentic loop?"**
Triple termination guard: confidence threshold, max recursion depth, wall-clock time
limit. Any one of the three stops the loop. See `orchestrator.py:run_pipeline`.

**"How do you handle failures in a multi-agent system?"**
Each agent wraps its LLM call in `_safe_run` with a time-check guard. `_parse` in
`base.py` falls back gracefully if the model returns malformed JSON — returns a
low-confidence result rather than crashing the pipeline.

**"Why use asyncio instead of threading for parallel agents?"**
LLM calls are I/O-bound (waiting on HTTP responses), not CPU-bound. `asyncio` handles
thousands of concurrent I/O waits on a single thread with no GIL contention, lower
memory overhead, and simpler shared-state reasoning than threading.

**"How would you scale this to 50,000 documents?"**
Switch the RAG vector store from ChromaDB (in-process) to Pinecone or Weaviate
(managed). Replace MemorySaver with a Redis or PostgreSQL checkpointer. Add a task
queue (Celery/ARQ) in front of the pipeline. Use Batch API for the non-interactive
classification and extraction passes, real-time only for user-facing query responses.

**"What's the difference between your two systems?"**
`financial_pipeline` uses LangGraph — appropriate for a fixed domain (trade finance)
with known document types and predictable routing. `agent_pipeline` uses raw asyncio —
appropriate for a general-purpose task where the set of analysis steps is the same every
time and LangGraph's graph overhead isn't needed.

---

## Tech Stack

| Component | `agent_pipeline` | `financial_pipeline` |
|---|---|---|
| Orchestration | asyncio + custom | LangGraph |
| LLM | Anthropic SDK direct | LangChain (Anthropic / OpenAI) |
| Caching | `cache_control: ephemeral` | Not implemented |
| Vector store | — | ChromaDB |
| Embeddings | — | sentence-transformers |
| API | — | FastAPI |
| Observability | CostTracker (custom) | LangSmith |
| State | Python dataclass | TypedDict (LangGraph) |

---

## Setup

```bash
git clone https://github.com/divyansh2k16/New-Agentic.git
cd New-Agentic

# Frugal pipeline (free dry-run)
cd agent_pipeline && pip install -r requirements.txt
python main.py --sample

# Financial pipeline
cd ../financial_pipeline && cp .env.example .env
pip install -r requirements.txt && python scripts/run_pipeline.py
```
