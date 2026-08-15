"""documents API — Upload, list, delete documents."""

import asyncio,os,json
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Form, HTTPException, status, Depends
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.responses import StreamingResponse
import docx
from pypdf import PdfReader


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

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
REJECTED_EXTENSIONS_WITH_HELP = {
    ".doc": "Older Word format (.doc) is not supported. Please convert it to .docx before uploading.",
    ".rtf": "Rich Text Format (.rtf) is not supported. Please convert it to .pdf, .docx, .md, or .txt before uploading."
}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/plain",
    "text/markdown",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}



def _safe_filename(name: str) -> str:
    return Path(name).name


def verify_magic_bytes(contents: bytes, ext: str) -> bool:
    if ext == ".pdf":
        return contents.startswith(b"%PDF")
    if ext == ".docx":
        # DOCX files are zip files starting with PK
        return contents.startswith(b"PK\x03\x04")
    if ext in {".txt", ".md"}:
        try:
            # Plain text files must be decodeable as UTF-8 or ASCII
            contents[:2048].decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    return False



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

            # Redundant status and chunk count update is now handled within Indexer.index()
            pass

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
    """Upload and index a PDF, Word, Markdown, or Text document."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")
    
    # 1. Extension check
    ext = Path(file.filename).suffix.lower()
    if ext in REJECTED_EXTENSIONS_WITH_HELP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=REJECTED_EXTENSIONS_WITH_HELP[ext]
        )
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # 2. MIME check
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported MIME type '{file.content_type}'. Allowed MIME types: {', '.join(sorted(ALLOWED_MIME_TYPES))}"
        )

    # 3. Check Content-Length header (if available) as a fast reject
    max_size_bytes = 30 * 1024 * 1024  # 30 MB
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > max_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="File size exceeds the 30 MB limit (max 30 MB)"
                )
        except ValueError:
            pass

    if not settings.jina_key:
        raise HTTPException(
            status_code=500,
            detail="Missing JINA_KEY; required for embeddings during indexing",
        )

    safe_name = _safe_filename(file.filename)
    # Ensure name length fits in database column constraints
    if len(safe_name) > 200:
        name_path = Path(safe_name)
        stem = name_path.stem
        suffix = name_path.suffix
        safe_name = stem[:200 - len(suffix)] + suffix

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Read contents once for verification
    contents = await file.read()

    # 4. Empty file check
    if not contents or len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    # 5. File size check
    if len(contents) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds the 30 MB limit (max 30 MB)"
        )

    # 6. Magic bytes check
    if not verify_magic_bytes(contents, ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File contents do not match the declared file extension."
        )

    # 7. Generate safe filename (UUID/public_id based, never original filename on disk)
    public_id = uuid4().hex
    saved_path = upload_dir / f"{public_id}{ext}"
    saved_path.write_bytes(contents)

    # 8. Dry-run parser check to identify unreadable/corrupted files upfront
    if ext == ".docx":
        try:
            # Attempt to parse document structure
            _ = docx.Document(saved_path)
        except Exception as e:
            if saved_path.exists():
                saved_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or invalid Word document (.docx). Please make sure it is a valid document. Error: {e}"
            )
    elif ext == ".pdf":
        try:
            reader = PdfReader(saved_path)
            # Ensure we can read pages metadata without error
            _ = len(reader.pages)
        except Exception as e:
            if saved_path.exists():
                saved_path.unlink()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupted or unreadable PDF file. Please make sure it is valid. Error: {e}"
            )

    # Create Document row first so Chunk FK constraint is satisfied
    doc = models.Document(
        user_id=current_user.id,
        public_id = public_id,
        name=safe_name,
        file_path=str(saved_path),
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