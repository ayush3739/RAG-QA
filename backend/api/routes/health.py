"""Health Check API."""

from fastapi import APIRouter
from backend.core.utils import vector_db_health_check
from backend.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint.""" 
    ok, err = vector_db_health_check(settings.qdrant_url)
    status = "ok" if ok else "degraded"
    return {
        "status": status,
        "service": "DocuMind API v2.0",
        "vector_db": {"healthy": ok, "error": err},
    }

