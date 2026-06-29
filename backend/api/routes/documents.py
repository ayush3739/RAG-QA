"""documents API — Upload, list, delete documents."""

import asyncio,os,json
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form, HTTPException, status, Depends
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse


from backend.db.base import get_db, AsyncSessionLocal
from backend.api.deps import get_current_user
from backend.core.config import settings
from backend.core.indexer import Indexer
from backend.core.retriever import Retriever
from backend.models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    IndexJobStatusResponse,
    DocumentItem
)
from backend.models import models
from backend.services.session_service import SessionService

router = APIRouter()

# Lightweight in-memory job tracker (replace with Redis/DB for production)
_INDEX_JOBS: dict[str, dict] = {}


def _safe_filename(name: str) -> str:
    return Path(name).name


# ── Background job ────────────────────────────────────────────────────────────

async def _run_index_job_async(job_id: str, document_id: int,document_public_id :int, file_path: str) -> None:
    """
    Runs as an asyncio task on the main event loop.
    Gets its own AsyncSession — never shares the request session.
    CPU-bound steps inside Indexer.index() are offloaded via asyncio.to_thread.
    """
    try:
        _INDEX_JOBS[job_id]["status"] = "running"

        async with AsyncSessionLocal() as db:
            await Indexer(
                file_path=file_path,
                db_session=db,
                document_id=document_id,
                document_public_id = document_public_id
            ).index()

            # Update document status + chunk count
            result = await db.execute(
                select(models.Document).where(models.Document.id == document_id)
            )
            doc = result.scalars().first()
            if doc:
                from sqlalchemy import func
                count_result = await db.execute(
                    select(func.count()).where(models.Chunk.document_id == document_id)
                )
                doc.chunk_count = count_result.scalar()
                doc.status = "indexed"
                await db.commit()

        _INDEX_JOBS[job_id]["status"] = "completed"

    except Exception as exc:
        _INDEX_JOBS[job_id]["status"] = "failed"
        _INDEX_JOBS[job_id]["error"] = str(exc)
        # Mark document as failed in DB
        try:
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    select(models.Document).where(models.Document.id == document_id)
                )
                doc = result.scalars().first()
                if doc:
                    doc.status = "failed"
                    await db.commit()
        except Exception:
            pass


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: models.User = Depends(get_current_user),
    file: UploadFile = File(...),
    session_id: UUID | None = Form(None),
):
    """Upload and index a PDF document."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")
    if not settings.github_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing GITHUB_TOKEN; required for embeddings during indexing",
        )

    safe_name = _safe_filename(file.filename)
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    contents = await file.read()
    public_id = uuid4().hex
    saved_path = upload_dir / f"{public_id}_{safe_name}"
    saved_path.write_bytes(contents)

    # Create Document row first so Chunk FK constraint is satisfied
    doc = models.Document(
        user_id=current_user.id,
        public_id = public_id,
        name=safe_name,
        file_path=str(saved_path),                           # filled in after we know doc.id
        bm25_path = f"data/bm25/{public_id}_bm25.pkl",
        status="queued",
        mime_type=file.content_type,
        file_size_kb=len(contents) // 1024,
    )
    db.add(doc)                               
    await db.commit()
    await db.refresh(doc)

    if session_id:
        session_service = SessionService()
        
        # Verify the session exists and belongs to the user
        session = await session_service.get_session(session_id, db)
        if not session or session.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, 
                detail="Invalid session_id or unauthorized"
            )
            
        await session_service.link_document_to_session(session_id=session_id, document_id=doc.id, db=db)

    job_id = uuid4().hex
    _INDEX_JOBS[job_id] = {
        "status": "queued",
        "document_id": doc.id,
        "filename": safe_name,
        "path": str(saved_path),
    }

    # Schedule as a real async task — does not block the response
    asyncio.create_task(
        _run_index_job_async(job_id, doc.id,doc.public_id, str(saved_path))
    )

    return {
        "status": "queued",
        "job_id": job_id,
        "document_id": doc.public_id,
        "filename": safe_name,
    }


@router.get("/documents/all", response_model=DocumentListResponse)
async def list_documents(
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: models.User = Depends(get_current_user)
):
    """List all indexed documents."""
    try:
        result = await db.execute(
            select(models.Document).where(models.Document.user_id == current_user.id)
        )
        docs = result.scalars().all()
        return DocumentListResponse(documents=[
            DocumentItem(name=doc.name, 
            public_id=doc.public_id, 
            chunk_count=doc.chunk_count,
            status=doc.status, 
            mime_type=doc.mime_type, 
            file_size_kb=doc.file_size_kb
            ) for doc in docs
        ])

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Postgres unavailable: {exc}",
        )


@router.delete("/document/{public_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    public_id: str, 
    db: Annotated[AsyncSession, Depends(get_db)],
    current_user: models.User = Depends(get_current_user)
):
    """Delete a document and its chunks."""
    result = await db.execute(
        select(models.Document).where(
            models.Document.public_id == public_id,
            models.Document.user_id == current_user.id
        )
    )
    doc = result.scalars().first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )
        
    
    # Store paths before deleting the DB row
    file_path = doc.file_path
    bm25_path = doc.bm25_path
    
    await db.delete(doc)
    await db.commit()
    
    # Clean up physical files
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass
            
    if bm25_path and os.path.exists(bm25_path):
        try:
            os.remove(bm25_path)
        except Exception:
            pass

    return DocumentDeleteResponse(status="deleted", document=doc.name)

@router.post("/test-retrieval")
async def test_retrieval(
    document_id: int,
    query: str,
    db: AsyncSession = Depends(get_db)
):
    retriever = Retriever(
        document_ids=[document_id],
        db=db
    )

    result = await retriever.similarity_search(query)

    return result

@router.post("/test-answer")
async def test_answer(
    document_id: int,
    query: str,
    db: AsyncSession = Depends(get_db)
):
    retriever = Retriever(
        document_ids=[document_id],
        db=db
    )

    return await retriever.answer(query)

@router.get("/documents/status/{job_id}")
async def get_indexing_status(job_id: str):
    """Stream indexing job progress via Server-Sent Events."""
    async def event_generator():
        while True:
            job = _INDEX_JOBS.get(job_id)
            if not job:
                yield f"data: {json.dumps({'status': 'failed', 'error': 'Job not found'})}\n\n"
                break
            
            # Yield current status
            yield f"data: {json.dumps({'job_id': job_id, 'status': job['status'], 'document_id': job.get('document_id'), 'filename': job.get('filename'), 'error': job.get('error')})}\n\n"
            
            if job["status"] in ["completed", "failed"]:
                break
                
            await asyncio.sleep(1)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")