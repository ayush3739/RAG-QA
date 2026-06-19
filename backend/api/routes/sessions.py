"""
backend/api/routes/sessions.py

Session management API.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from backend.api.deps import get_current_user, get_db
from backend.models.models import User
from backend.services.session_service import SessionService
from backend.models.schemas import (
    SessionCreate,
    SessionList,
    MessageList,
    SessionDeleteResponse
)

router = APIRouter()
_session_service = SessionService()

@router.post("/", response_model=SessionCreate, status_code=status.HTTP_201_CREATED)
async def create_session(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new chat session."""
    return await _session_service.create_session(user_id=current_user.id, db=db)


@router.get("/", response_model=SessionList)
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all sessions for the current user."""
    return await _session_service.list_sessions(user_id=current_user.id, db=db)


@router.get("/{session_id}/history", response_model=MessageList)
async def get_session_history(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get the full message history for a specific session."""
    # Note: In a production app, verify that the session belongs to current_user
    session = await _session_service.get_session(session_id=session_id, db=db)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    return await _session_service.get_messages(session_id=session_id, db=db)


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a specific session."""
    session = await _session_service.get_session(session_id=session_id, db=db)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    return await _session_service.delete_session(session_id=session_id, db=db)


@router.post("/{session_id}/documents/{document_id}", status_code=status.HTTP_200_OK)
async def link_document_to_session(
    session_id: UUID,
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Explicitly link an existing document to a session."""
    session = await _session_service.get_session(session_id=session_id, db=db)
    if not session or session.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
        
    # Link the document
    await _session_service.link_document_to_session(session_id=session_id, document_id=document_id, db=db)
    return {"status": "linked", "session_id": session_id, "document_id": document_id}
