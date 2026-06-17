"""Session Service — Post Gres Chat history storage."""
from sqlalchemy import select
from backend.db.base import AsyncSession
from backend.models.models import Session,Message,SessionDocument
from backend.models.schemas import (SessionCreate,SessionList,
SessionItem,MessageItem,MessageList, SessionDeleteResponse,MessageCreateResponse,SessionDocumentList)
from uuid import UUID
from datetime import datetime

class SessionService:
    """Manage chat sessions and history."""
    
    async def create_session(self,user_id: int,db: AsyncSession) -> SessionCreate:
        session = Session(
            title=f"Chat {datetime.now():%Y-%m-%d}",
            user_id=user_id,
        )

        db.add(session)

        await db.commit()
        await db.refresh(session)

        return SessionCreate(session_id=session.id)
        
    
    async def add_message(
    self,
    session_id: UUID,
    role: str,
    content: str,
    db: AsyncSession,
    sources: list | None = None,
    confidence: float = 0.0,
    tool_used: str | None = None,
) -> MessageCreateResponse:
        """Add message to session history."""
        mess = Message(
            session_id = session_id,
            role = role,
            content = content,
            tool_used = tool_used,
            chunks = None,
            confidence = confidence,
            citations = {"sources": sources or []},
        )
        db.add(mess)
        await db.commit()
        await db.refresh(mess)

        return MessageCreateResponse(status="message added",message_id= mess.id)
        
    async def get_messages(self,session_id: UUID, db: AsyncSession, ) -> MessageList:
        """Get all messages in chronological order."""

        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )

        messages = result.scalars().all()

        return MessageList(
            messages=[
                MessageItem(
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    citations=m.citations,
                    chunks=m.chunks,
                    confidence=m.confidence,
                    tool_used=m.tool_used,
                    used_vector_db=m.used_vector_db,
                    debug=m.debug,
                    created_at=m.created_at,
                )
                for m in messages
            ]
        )
    
    async def get_recent_history(self, session_id: UUID, db: AsyncSession, limit: int = 10,) -> MessageList:
        """Get latest N messages for LLM context."""

        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )

        messages = list(reversed(result.scalars().all()))

        return MessageList(
            messages=[
                MessageItem(
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    citations=m.citations,
                    chunks=m.chunks,
                    confidence=m.confidence,
                    tool_used=m.tool_used,
                    used_vector_db=m.used_vector_db,
                    debug=m.debug,
                    created_at=m.created_at,
                )
                for m in messages
            ]
        )
    
    async def list_sessions(self,user_id :int, db: AsyncSession) -> SessionList:
        """List all sessions for a user."""
        result = await db.execute(select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc()
        ))

        sessions = result.scalars().all()
        return SessionList(sessions=[SessionItem(session_id=s.id,title=s.title) for s in sessions])    
    
    async def get_session_documents(self, session_id: UUID, db: AsyncSession,) -> list[int]:
        result = await db.execute(
            select(SessionDocument.document_id)
            .where(SessionDocument.session_id == session_id)
        )
        return result.scalars().all()
        
    async def delete_session(self, session_id: UUID,db: AsyncSession) ->SessionDeleteResponse:
        """Delete a session."""
        result  = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalars().first()
        if not session:
            return SessionDeleteResponse(
                status="Not Deleted",
                error="Session not found"
        )
        await db.delete(session)
        await db.commit()     

        return SessionDeleteResponse(status="deleted",session_id=session.id,session_name=session.title)
