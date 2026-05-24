"""Collections API — Upload, list, delete documents."""

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException, status
from qdrant_client import QdrantClient
from pathlib import Path
from uuid import uuid4

from backend.core.config import settings
from backend.core.indexer import Indexer
from backend.models.schemas import (
    CollectionUploadResponse,
    CollectionListResponse,
    CollectionDeleteResponse,
    IndexJobStatusResponse,
)

router = APIRouter()

# Lightweight in-memory job tracker (replace with Redis/DB for production)
_INDEX_JOBS: dict[str, dict] = {}


def _safe_filename(name: str) -> str:
    return Path(name).name


def _make_collection_id() -> str:
    return uuid4().hex


def _run_index_job(job_id: str, collection_id: str, file_path: str) -> None:
    try:
        _INDEX_JOBS[job_id]["status"] = "running"
        Indexer(file_path, collection_name=collection_id).index()
        _INDEX_JOBS[job_id]["status"] = "completed"
    except Exception as exc:
        _INDEX_JOBS[job_id]["status"] = "failed"
        _INDEX_JOBS[job_id]["error"] = str(exc)


@router.post("/collections/upload", response_model=CollectionUploadResponse)
async def upload_collection(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")
    if not settings.github_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing GITHUB_TOKEN; required for embeddings during indexing",
        )

    collection_id = _make_collection_id()
    job_id = uuid4().hex
    safe_name = _safe_filename(file.filename)
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{collection_id}_{safe_name}"

    contents = await file.read()
    saved_path.write_bytes(contents)

    _INDEX_JOBS[job_id] = {
        "status": "queued",
        "collection_id": collection_id,
        "filename": safe_name,
        "path": str(saved_path),
    }
    background_tasks.add_task(_run_index_job, job_id, collection_id, str(saved_path))

    return {
        "status": "queued",
        "job_id": job_id,
        "collection_id": collection_id,
        "filename": safe_name,
        
    }


@router.get("/collections", response_model=CollectionListResponse)
async def list_collections():
    """List all indexed collections."""   
    try:
        client = QdrantClient(url=settings.qdrant_url)
        collections = [c.name for c in client.get_collections().collections]
        return {"collections": collections }
    
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Qdrant unavailable: {exc}")


@router.delete("/collections/{name}", response_model=CollectionDeleteResponse)
async def delete_collection(name: str):
    """Delete a collection."""
    try:
        client = QdrantClient(url=settings.qdrant_url)
        existing = [c.name for c in client.get_collections().collections]
        if name not in existing:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Collection not found")
        client.delete_collection(collection_name=name)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Qdrant unavailable: {exc}",
        )

    bm25_path = Path("data/bm25") / f"{name}_bm25.pkl"
    if bm25_path.exists():
        bm25_path.unlink()

    upload_dir = Path("data/uploads")
    if upload_dir.exists():
        for upload_file in upload_dir.glob(f"{name}_*"):
            try:
                upload_file.unlink()
            except Exception:
                pass

    return CollectionDeleteResponse(status="deleted", collection=name)


@router.get("/collections/status/{job_id}", response_model=IndexJobStatusResponse)
async def get_indexing_status(job_id: str):
    """Get indexing progress."""
    job = _INDEX_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return IndexJobStatusResponse(job_id=job_id, **job)
