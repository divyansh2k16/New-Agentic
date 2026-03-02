"""
FastAPI Application Entry Point

CONCEPT: The API layer is the production interface between the UI/clients and the agent system.
Production concerns addressed here:
1. CORS: allow frontend to call the API
2. Rate limiting: prevent abuse (slowapi)
3. Request logging: every request is logged
4. Error handling: structured error responses
5. Health check: for load balancer / k8s probes
6. OpenAPI docs: auto-generated at /docs
"""
import time
import uuid
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes.auth_routes import router as auth_router
from api.routes.documents import router as docs_router
from api.routes.query import router as query_router
from config.settings import get_settings

settings = get_settings()

# ── Rate limiter setup ────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown lifecycle."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"LLM: {settings.primary_llm_model} (primary) / {settings.fast_llm_model} (fast)")
    settings.ensure_dirs()
    yield
    logger.info("Shutting down...")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Financial PDF Intelligence Pipeline

Production-grade multi-agent AI system for financial document processing.

### Features
- **Classification**: Auto-detect document type, company, fiscal year
- **Extraction**: Pull structured financial metrics from any PDF
- **Comparison**: Year-over-year analysis across multiple documents
- **Summarization**: Executive summaries with key insights
- **Query Engine**: Natural language Q&A over your documents (RAG)
- **Compliance**: Dual-use material detection for trade documents

### Authentication
Use `/auth/login` with form data. Demo users:
- `analyst@citi.com` / `demo1234`
- `admin@citi.com` / `admin1234`
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],  # Streamlit + React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log every request with timing. Essential for production debugging."""
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info(f"[{request_id}] → {request.method} {request.url.path}")

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"[{request_id}] ← {response.status_code} "
        f"({duration_ms:.1f}ms) {request.url.path}"
    )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(docs_router)
app.include_router(query_router)


# ── Health & utility endpoints ────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """Kubernetes / load balancer health probe."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "model": settings.primary_llm_model,
    }


@app.get("/", tags=["System"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health",
    }


# ── Startup (run directly) ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
