from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from backend.core.config import settings
from backend.db.base import engine, get_db

# Import routers
from backend.api.routes import chat, documents, research, feedback,auth,user,sessions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT_DIR / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI):

    logger.info("🚀 Starting DocuMind API...")
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))

            await conn.execute(
                text("CREATE EXTENSION IF NOT EXISTS vector")
            )
        logger.info("✅ Database and pgvector verified")
        yield
    except Exception as e:
        logger.exception(f"❌ Startup failed: {e}")
        raise
     
    finally:
        await engine.dispose()
        logger.info("🛑 Shutting down DocuMind API...")


app = FastAPI(title="DocuMind API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=FRONTEND_DIR),
        name="frontend",
    )


@app.get("/test-chat", include_in_schema=False)
async def test_chat_page():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/test-history", include_in_schema=False)
async def test_history_page():
    return FileResponse(FRONTEND_DIR / "history.html")


# Include routers
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat_sse"])
app.include_router(research.router, prefix="/api/v1", tags=["Research"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(sessions.router, prefix="/api/v1/sessions", tags=["Sessions"])

@app.get("/health")
async def health_check(db: Annotated[AsyncSession, Depends(get_db)]):
    """Health check endpoint."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "error",
                "service": "DocuMind API v2.0",
                "database": {"healthy": False, "error": "Database unavailable"},
                "vector_db": {"healthy": False, "error": str(exc) },
            },
        ) from exc
    return {
        "status": "ok",
        "service": "DocuMind API v2.0",
        "database": {"healthy": True, "error": None},
        "vector_db": {"healthy":True , "error": None},
    }


@app.get("/api/v1/health/db", tags=["Health"], summary="Postgres DB health")
async def db_health_check(db: Annotated[AsyncSession, Depends(get_db)]):
    """Lightweight Postgres connectivity check (runs `SELECT 1`)."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"database": {"healthy": False, "error": str(exc)}},
        ) from exc
    return {"database": {"healthy": True, "error": None}}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
