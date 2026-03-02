"""
Document Upload & Management Routes

CONCEPT: Documents are the raw input. This API handles:
- Multi-file upload (up to 15 PDFs at once)
- Async processing (don't make the user wait for full pipeline)
- Background task queue (FastAPI BackgroundTasks for local; SQS on AWS)
- Status tracking (poll for completion)
"""
import os
import uuid
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import aiofiles
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from loguru import logger

from api.auth import get_current_user, User, require_analyst
from config.settings import get_settings
from tools.guardrails import validate_file

settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])

# In-memory job tracker (replace with Redis in production)
_jobs: dict = {}


class UploadResponse(BaseModel):
    session_id: str
    uploaded_files: List[str]
    rejected_files: List[dict]
    job_id: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    session_id: str
    status: str          # pending | running | completed | failed
    progress: int        # 0-100
    completed_steps: List[str]
    errors: List[str]
    result: Optional[dict] = None
    created_at: str
    updated_at: str


def _run_pipeline_background(job_id: str, session_id: str, file_paths: List[str], user_id: str):
    """Background task: runs the full agent pipeline."""
    try:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()

        from agents.orchestrator import run_pipeline
        from rag.knowledge_base import ingest_documents_batch

        # Step 1: Ingest into knowledge base (for RAG)
        _jobs[job_id]["progress"] = 10
        ingest_results = ingest_documents_batch(file_paths, session_id)
        logger.info(f"[JOB {job_id}] Ingested {sum(r['chunks_added'] for r in ingest_results)} chunks")

        _jobs[job_id]["progress"] = 20

        # Step 2: Run multi-agent pipeline
        final_state = run_pipeline(
            document_paths=file_paths,
            task="full_pipeline",
            user_id=user_id,
            session_id=session_id,
        )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 100
        _jobs[job_id]["completed_steps"] = final_state.get("completed_steps", [])
        _jobs[job_id]["errors"] = final_state.get("errors", [])
        _jobs[job_id]["result"] = {
            "classifications": final_state.get("classifications", []),
            "extractions": _serialize_extractions(final_state.get("extractions", [])),
            "comparison": final_state.get("comparison"),
            "summary": final_state.get("summary"),
            "token_usage": {
                "input": final_state.get("total_input_tokens", 0),
                "output": final_state.get("total_output_tokens", 0),
            },
        }
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
        logger.info(f"[JOB {job_id}] Completed successfully")

    except Exception as e:
        logger.error(f"[JOB {job_id}] Failed: {e}")
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["errors"] = [str(e)]
        _jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()


def _serialize_extractions(extractions: list) -> list:
    """Convert ExtractedFinancials TypedDicts to JSON-serialisable dicts."""
    result = []
    for ext in extractions:
        clean = {k: v for k, v in ext.items() if k != "raw_tables"}
        result.append(clean)
    return result


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: User = Depends(require_analyst),
):
    """
    Upload 1-15 PDF financial documents for processing.

    - Files are saved to disk
    - Validation runs immediately (fast)
    - Full pipeline runs as a background task
    - Returns a job_id to poll for status
    """
    if not files:
        raise HTTPException(400, "No files uploaded")
    if len(files) > 15:
        raise HTTPException(400, "Maximum 15 files per upload")

    session_id = str(uuid.uuid4())
    session_dir = Path(settings.upload_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    uploaded = []
    rejected = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            rejected.append({"filename": file.filename, "reason": "Not a PDF"})
            continue

        # Check file size before saving
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        if size > settings.max_file_size_mb * 1024 * 1024:
            rejected.append({
                "filename": file.filename,
                "reason": f"Exceeds {settings.max_file_size_mb}MB limit"
            })
            continue

        save_path = session_dir / file.filename
        async with aiofiles.open(save_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        # Validate saved file
        is_valid, reason = validate_file(str(save_path))
        if not is_valid:
            save_path.unlink(missing_ok=True)
            rejected.append({"filename": file.filename, "reason": reason})
        else:
            uploaded.append(str(save_path))

    if not uploaded:
        raise HTTPException(400, f"No valid PDFs. Issues: {rejected}")

    # Create job
    job_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    _jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "status": "pending",
        "progress": 0,
        "completed_steps": [],
        "errors": [],
        "result": None,
        "created_at": now,
        "updated_at": now,
    }

    # Schedule background pipeline
    background_tasks.add_task(
        _run_pipeline_background,
        job_id=job_id,
        session_id=session_id,
        file_paths=uploaded,
        user_id=current_user.user_id,
    )

    logger.info(
        f"[UPLOAD] User {current_user.email} | "
        f"Session: {session_id} | Files: {len(uploaded)} | Job: {job_id}"
    )

    return UploadResponse(
        session_id=session_id,
        uploaded_files=[Path(p).name for p in uploaded],
        rejected_files=rejected,
        job_id=job_id,
        message=f"Processing {len(uploaded)} file(s). Poll /documents/status/{job_id} for results.",
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll the status of a background processing job."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")

    job = _jobs[job_id]
    return JobStatus(**job)


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Return the executive summary for a completed session."""
    # Find job by session_id
    job = next((j for j in _jobs.values() if j["session_id"] == session_id), None)
    if not job:
        raise HTTPException(404, "Session not found")
    if job["status"] != "completed":
        raise HTTPException(400, f"Pipeline not complete yet. Status: {job['status']}")

    result = job.get("result", {})
    return {
        "session_id": session_id,
        "summary": result.get("summary", ""),
        "classifications": result.get("classifications", []),
        "token_usage": result.get("token_usage", {}),
    }
