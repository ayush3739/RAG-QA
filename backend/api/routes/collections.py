"""Collections API — Upload, list, delete documents."""

from fastapi import APIRouter, UploadFile, File
from backend.models.schemas import CollectionResponse

router = APIRouter()


@router.post("/collections/upload")
async def upload_collection(file: UploadFile = File(...)):
    """Upload and index a PDF document."""
    # TODO: Implement async indexing
    return {"status": "success", "collection": file.filename}


@router.get("/collections")
async def list_collections():
    """List all indexed collections."""
    # TODO: Connect to Qdrant
    return {"collections": []}


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection."""
    # TODO: Implement Qdrant collection deletion
    return {"status": "deleted", "collection": name}


@router.get("/collections/status/{job_id}")
async def get_indexing_status(job_id: str):
    """Get indexing progress."""
    # TODO: Track async job status
    return {"job_id": job_id, "progress": 100}
