# Interview Preparation Guide
## Financial Document AI Pipeline — Citi Data Scientist Role

---

## PART 1: HOW TO TALK ABOUT YOUR PROJECT (Your Story)

When asked "Tell me about your project" — use this 90-second structure:

> "At Citi, I engineered a production-grade, multi-agent AI system for processing
> trade and financial documents. The system handles classification, data extraction,
> year-over-year comparison, and natural language Q&A over a knowledge base of 1000+
> documents. The architecture uses LangGraph for agent orchestration, a hybrid RAG
> pipeline combining semantic and keyword search for retrieval, and Claude as the
> primary LLM. I built it with production concerns in mind — authentication, rate
> limiting, guardrails for compliance, monitoring with LangSmith, and a clear AWS
> deployment path. The dual-use material detection is particularly critical for trade
> finance compliance under export control regulations."

---

## PART 2: CORE CONCEPTS — UNDERSTAND THESE DEEPLY

---

### Q1: What is Retrieval-Augmented Generation (RAG) and why do you use it?

**Your answer:**
> RAG is a pattern where instead of relying solely on an LLM's training knowledge,
> you first retrieve relevant documents from a knowledge base and inject them into
> the LLM's context window. This solves two fundamental problems:
>
> 1. **Knowledge cutoff**: LLMs don't know about your specific documents. A financial
>    report from 2024 wasn't in the training data.
> 2. **Hallucination**: LLMs can confidently make up numbers. With RAG, you constrain
>    the model to only answer from retrieved context, and you can verify every claim.
>
> In my system, when a user asks "What was Meridian Energy's net income in 2022?",
> we don't send the entire document to the LLM. Instead, we embed the query, find
> the 5-6 most semantically relevant chunks from our ChromaDB vector store, and only
> send those chunks to Claude. This saves 95% of the tokens and gives a grounded answer.

**Follow-up: What are the weaknesses of RAG?**
> - Retrieval quality is a hard ceiling — garbage in, garbage out
> - Chunking strategy dramatically affects quality (too small = no context, too large = diluted relevance)
> - Complex multi-hop questions need chain-of-thought + multiple retrieval rounds
> - Evaluation is harder than standard ML (no single ground truth)

---

### Q2: What is a multi-agent system and why use LangGraph?

**Your answer:**
> A multi-agent system decomposes a complex task into specialised sub-agents.
> Instead of one LLM prompt doing everything (which overwhelms the context window
> and leads to poor quality), each agent has a single responsibility:
>
> - **Classifier**: Is this an annual report or earnings release? What company? What year?
> - **Extractor**: Pull specific financial numbers from the document.
> - **Comparator**: Calculate year-over-year changes across multiple documents.
> - **Summarizer**: Write the executive summary from all the above.
> - **Query Agent**: Answer natural language questions using RAG.
>
> LangGraph is the framework I use to wire these agents together. It's a directed
> graph where:
> - **Nodes** are agents (Python functions)
> - **Edges** are the data flow between agents
> - **State** is a shared TypedDict that every agent reads from and writes to
> - **Conditional edges** allow routing — if only 1 document, skip the comparator
>
> The key advantage over plain function calls: LangGraph has built-in checkpointing
> (resume interrupted pipelines), human-in-the-loop support, and visual graph
> rendering for debugging.

**Follow-up: When would you NOT use multi-agent?**
> For simple, single-document tasks, a single well-crafted prompt is cheaper and faster.
> Multi-agent adds latency and complexity. The crossover point is when tasks require
> different expertise levels, different models (cheap for classification, expensive for
> extraction), or parallel processing of many documents.

---

### Q3: What is the difference between semantic search and keyword (BM25) search?

**Your answer:**
> **Semantic search** converts text into dense vectors using an embedding model.
> Similarity is computed in vector space using cosine similarity. It understands
> meaning and paraphrases: "profit" and "net income" will be close in vector space.
>
> **BM25** (Best Match 25) is a classic statistical algorithm that counts term
> frequency weighted by inverse document frequency. It finds exact or near-exact
> keyword matches. It's great for specific numbers: if you search "34.5 billion",
> BM25 will find exactly "34.5 billion" even if the vector model doesn't.
>
> In my system I use **hybrid retrieval** combining both:
> - Over-retrieve with both methods (top-20 from each)
> - Merge using **Reciprocal Rank Fusion (RRF)** — a mathematical technique
>   that assigns each result a score of 1/(k+rank) and sums across both lists
> - Return the top-6 after fusion
>
> Financial documents heavily use specific numbers, so pure semantic search misses
> exact figures. Hybrid retrieval consistently outperforms either alone by 10-20% on F1.

---

### Q4: How do you handle the context window limitation?

**Your answer:**
> Claude Sonnet has a 200K token context window, but sending 15 full annual reports
> (each 100+ pages) would be ~3M tokens — impossible and prohibitively expensive.
>
> My strategies:
> 1. **Chunking**: Documents are split into 500-character chunks with 50-char overlap.
>    Only the 5-6 most relevant chunks are sent per query. (~1,500 tokens)
> 2. **Tiered model selection**: Classification uses Haiku (cheap, fast). Extraction
>    uses Sonnet (more capable). This is "model routing" — match model capability
>    to task complexity.
> 3. **Summarisation for classification**: Only the first 2,000 characters are sent
>    to the classifier — enough to identify document type without reading everything.
> 4. **Conversation windowing**: Multi-turn conversations keep only the last 3 Q&A
>    pairs in context. Older history is compressed or dropped.
> 5. **Pre-computed results**: After extraction, structured numbers are stored in the
>    state dict. The summarizer uses these numbers, not raw PDF text.
>
> Together these reduce token usage by ~80% vs naive full-document prompting.

---

### Q5: What are guardrails in an AI system?

**Your answer:**
> Guardrails are safety mechanisms that sit before (input) and after (output) the LLM.
> At a bank like Citi, these are mandatory — not optional.
>
> **Input guardrails:**
> - File type and size validation (only PDFs, max 50MB)
> - PDF magic byte verification (prevent file extension spoofing)
> - Query length limits (prevent DoS via huge prompts)
> - Banned content patterns (prompt injection detection)
> - PII scanning (SSN, credit card numbers in uploaded docs)
>
> **Output guardrails:**
> - PII detection in LLM responses (LLMs sometimes repeat PII from context)
> - Hallucination detection (answers not grounded in retrieved context)
> - Extreme value flagging (if the model outputs 1000% revenue growth, flag it)
>
> **Compliance guardrails (banking-specific):**
> - Dual-use material detection for trade documents (export control regulations)
> - Audit logging (every LLM call is logged with inputs/outputs for FINRA compliance)
> - Data residency (European documents must stay in EU regions)
>
> In my architecture, the guardrail node is the FIRST node in the LangGraph — it runs
> before any LLM calls. This is "fail fast" design — no wasted API credits on bad inputs.

---

### Q6: How do you evaluate an AI system? What metrics do you use?

**Your answer:**
> This is crucial — you can't improve what you can't measure. I use a three-layer evaluation:
>
> **Layer 1: RAG-specific metrics (RAGAS framework)**
> - **Faithfulness** (0-1): Is every claim in the answer supported by the retrieved context?
>   Measures hallucination. A score of 1.0 means zero hallucination.
> - **Answer Relevancy** (0-1): Does the answer actually address the question?
> - **Context Precision** (0-1): Are the retrieved chunks actually useful for the answer?
> - **Context Recall** (0-1): Did we retrieve ALL the information needed?
>
> **Layer 2: Task-specific metrics**
> - **Extraction accuracy**: Compare extracted financial values against manually-labelled
>   ground truth. We use 1% tolerance — values within 1% of ground truth are correct.
>   F1 score over all numeric fields.
> - **Classification accuracy**: Precision/Recall on document type, company name, fiscal year.
>
> **Layer 3: LLM-as-a-judge**
> For open-ended evaluation (summary quality, insight quality), I use Claude-Haiku
> to score another Claude-Sonnet's outputs. This is the industry standard when
> there's no single correct answer. Cheaper than human eval; scales to CI/CD pipelines.
>
> **Production metrics:**
> - Latency: P50, P95, P99 per agent
> - Token cost: tracked per user, per session
> - Error rates: per agent, per document type
> - Active sessions: for capacity planning

---

### Q7: How would you deploy this on AWS?

**Your answer:**
> The production AWS architecture has 5 layers:
>
> **1. Ingestion Layer**
> - S3 bucket for document storage (server-side encryption with KMS)
> - Lambda trigger on S3 upload → sends to SQS queue
>
> **2. Processing Layer**
> - SQS queue with dead-letter queue for failed jobs
> - ECS Fargate tasks (auto-scaling) that consume from SQS
> - Each Fargate task runs the LangGraph pipeline
> - Amazon Bedrock for LLM inference (uses Claude models via Bedrock API)
>
> **3. Storage Layer**
> - Amazon OpenSearch for vector search (replaces ChromaDB)
> - RDS PostgreSQL for structured data and user management
> - ElastiCache Redis for session state and rate limiting
>
> **4. API/Auth Layer**
> - API Gateway + Lambda (or ALB + ECS) for FastAPI
> - Cognito for user authentication (supports MFA, SSO with Active Directory)
> - WAF (Web Application Firewall) for DDoS protection and rate limiting
>
> **5. Observability Layer**
> - CloudWatch Logs for all application logs
> - X-Ray for distributed tracing across Lambda/ECS
> - CloudWatch Metrics + Alarms for SLO monitoring
> - LangSmith for LLM-specific trace analysis
>
> **For Citi specifically**: would also need AWS PrivateLink (no public internet),
> VPC endpoints, CloudTrail for audit, and compliance with OCC/Federal Reserve guidelines.

---

### Q8: What is the dual-use material classification in your Citi project?

**Your answer:**
> In trade finance, banks like Citi process thousands of commercial documents daily —
> letters of credit, commercial invoices, bills of lading. Some of these reference
> goods that have both civilian and military applications: encryption technology,
> aircraft parts, chemical precursors, night-vision equipment.
>
> Dual-use goods are controlled under:
> - US: Export Administration Regulations (EAR), ITAR (military)
> - EU: Council Regulation 428/2009
> - UK: Export Control Order
>
> My classifier agent checks every trade document against a set of patterns and
> uses the LLM to reason about whether the goods described could have military
> applications. If flagged, the document is routed to a compliance analyst.
>
> The key design principle: **false negatives are far more costly than false positives**.
> A missed dual-use document can result in sanctions violations (millions in fines,
> regulatory action). So I set a conservative threshold — "when in doubt, flag it."
>
> This is implemented as a structured output field (is_dual_use_material: bool +
> dual_use_reasons: list) with few-shot examples of known dual-use document patterns.

---

### Q9: What is LangSmith and why do you use it?

**Your answer:**
> LangSmith is the observability platform for LangChain/LangGraph applications.
> When you set `LANGCHAIN_TRACING_V2=true`, every LLM call, retrieval step, and
> agent transition is automatically traced and sent to LangSmith.
>
> What you can see in LangSmith:
> - **Full trace**: Exact prompt sent, response received, latency, tokens used
> - **Agent path**: Which nodes executed, in what order, with what inputs/outputs
> - **Error diagnosis**: See exactly which prompt caused a JSON parsing failure
> - **Dataset management**: Create evaluation datasets from production traces
> - **Comparison**: A/B test different prompts or models on the same trace set
>
> For interview purposes, say: "LangSmith is to LLMs what Datadog is to microservices —
> it gives you the visibility you need to diagnose and improve production AI systems."

---

### Q10: How do you optimise for cost when many users run the pipeline simultaneously?

**Your answer:**
> Cost optimisation is a first-class concern. My strategies:
>
> **1. Model routing**: Use Haiku for classification (~$0.25/1M tokens) vs Sonnet
>    for extraction (~$3/1M tokens). Classification only needs to identify doc type.
>    Only complex reasoning steps use the expensive model.
>
> **2. Truncation at ingestion**: For classification, I only send 2,000 characters
>    (first page essentially). For extraction, I send 8,000 chars. Never the whole doc.
>
> **3. Pre-computed maths**: YoY calculations (percentages, ratios) are done in pure
>    Python, not by the LLM. LLMs are expensive calculators.
>
> **4. Caching**: Embed once, store in ChromaDB. Re-uploads of the same document
>    (detected by SHA-256 hash) are skipped — no re-embedding, no re-extraction.
>
> **5. Batch embedding**: sentence-transformers processes 32 chunks at once on GPU
>    rather than one-by-one.
>
> **6. Rate limiting**: 100 requests/minute per IP prevents cost runaway from abuse.
>
> **7. Async processing**: Background tasks (FastAPI BackgroundTasks / SQS on AWS)
>    allow the system to process many documents concurrently without blocking.
>
> For a 15-document session, estimated cost: ~$0.05-0.15 (classification cheap,
> extraction expensive, summary moderate).

---

## PART 3: BEHAVIOURAL / SITUATIONAL QUESTIONS

---

### Q11: Tell me about a time you had to improve an AI system's accuracy.

**Framework: STAR (Situation, Task, Action, Result)**

> **Situation**: Our initial document extraction had only 60% accuracy on financial tables —
> the LLM was guessing values when tables were presented in unusual formats.
>
> **Task**: Improve accuracy to above 85% while keeping latency acceptable.
>
> **Action**:
> - Added pdfplumber as primary PDF parser (better table detection than PyPDF)
> - Added explicit table extraction step before LLM call — tables sent as structured text
> - Changed prompt to explicitly reference table numbers ("See Table 2, row Revenue")
> - Added few-shot examples of correct extractions for different table formats
> - Created an evaluation dataset of 50 manually-labelled documents
>
> **Result**: Accuracy improved from 60% to 91% on the held-out test set.
> Latency increased by 2 seconds (table extraction step) but accuracy improvement justified it.

---

### Q12: How do you handle a case where the LLM returns incorrect/malformed output?

**Your answer:**
> In my code, every LLM call is wrapped in try/except with a fallback.
>
> For structured outputs (classification, extraction), I:
> 1. Ask the LLM to return JSON in the prompt
> 2. Strip markdown code blocks (```json ... ```) that LLMs often add
> 3. Use json.loads() — if it fails, log and return an empty/default result
> 4. Never let a single document failure crash the entire pipeline
>
> For more robust production systems, I'd use:
> - Pydantic models with `model.with_structured_output()` in LangChain
> - Retry with exponential backoff (ask LLM to fix its output)
> - Constitutional AI prompts ("If your output is not valid JSON, start over")
>
> The key principle: **degrade gracefully**. Return partial results rather than
> a complete failure. Log everything for debugging.

---

### Q13: What would you do differently if you were building this at true Citi scale (1M+ documents)?

**Your answer:**
> At 1M+ documents, everything changes:
>
> 1. **Vector DB**: ChromaDB is single-machine. At scale: Amazon OpenSearch with
>    HNSW index, or Pinecone. These handle billions of vectors with millisecond latency.
>
> 2. **Async processing**: Every document goes through an SQS queue. Worker pods
>    (ECS/EKS) auto-scale based on queue depth. DLQ (dead-letter queue) for failed docs.
>
> 3. **Embedding at scale**: Pre-compute embeddings in batch using SageMaker batch
>    transform — process 1M documents overnight rather than real-time.
>
> 4. **Model fine-tuning**: At this scale, it's worth fine-tuning a smaller model
>    (Mistral-7B) specifically on financial documents. Cheaper than Claude at scale,
>    and higher accuracy on the narrow domain.
>
> 5. **Caching layer**: Redis for embedding cache (same sentence = same vector),
>    query cache (popular questions answered from cache).
>
> 6. **Data governance**: Document lineage tracking, access controls per document,
>    retention policies, GDPR right-to-deletion.
>
> 7. **A/B testing infrastructure**: Test new prompts against old prompts on 5% traffic
>    before full rollout. LangSmith has this capability.

---

## PART 4: TECHNICAL FLASHCARD CONCEPTS

| Concept | One-Line Definition |
|---------|---------------------|
| RAG | Retrieve relevant context, inject into LLM prompt |
| Vector embedding | Dense numerical representation of text meaning |
| Cosine similarity | Angle-based similarity between two vectors (1=identical, 0=orthogonal) |
| BM25 | Statistical keyword search (TF-IDF variant) |
| Hybrid retrieval | Combine vector + keyword search using RRF |
| LangGraph | Stateful agent orchestration as a directed graph |
| LangChain | Modular LLM application framework |
| ChromaDB | Local-first vector database with persistence |
| Chunking | Splitting documents into smaller overlapping segments |
| Guardrails | Input/output safety and validation layer |
| LangSmith | LLM observability and tracing platform |
| RAGAS | RAG evaluation framework (faithfulness, relevancy, recall, precision) |
| HumanMessage / AIMessage | LangChain chat message types |
| TypedDict | Python typed dictionary — used for LangGraph state |
| Conditional edges | LangGraph routing based on state values |
| MemorySaver | In-memory checkpointing for LangGraph |
| Token | Smallest unit of LLM input/output (~0.75 words) |
| JWT | Stateless authentication token (header.payload.signature) |
| FastAPI | Modern async Python web framework with auto OpenAPI docs |
| Streamlit | Python-native UI framework for data apps |
| P95 latency | 95th percentile response time (worst 5% excluded) |
| Few-shot prompting | Include examples in the prompt to guide LLM output format |
| LLM-as-a-judge | Use one LLM to evaluate another's output |
| RRF | Reciprocal Rank Fusion — merges multiple ranked lists |
| Dual-use goods | Items with both civilian and military applications |
| HNSW | Hierarchical Navigable Small World — vector index algorithm |
| Sentence transformers | Local embedding models (MiniLM, MPNet, etc.) |
| Pydantic | Python data validation and serialisation library |
| BackgroundTasks | FastAPI async task runner |

---

## PART 5: QUESTIONS TO ASK THE INTERVIEWER

These show you think at a systems level:

1. "How does the team currently handle document processing for trade finance — is it mostly rule-based or ML-driven?"
2. "What are the biggest compliance constraints on LLM deployment at Citi — is there an approved list of models/providers?"
3. "How does the team approach evaluation — do you have a labelled dataset for financial document extraction?"
4. "Is the inference happening on-premise, in a private cloud, or via external APIs like Anthropic/OpenAI?"
5. "What's the biggest technical challenge you'd want me to tackle in the first 90 days?"
