"""Session Service — Postgres Chat history storage."""
from sqlalchemy import select
from backend.db.base import AsyncSession
from backend.models.models import Session, Message, SessionDocument, Document
from backend.models.schemas import (
    SessionCreate,
    SessionList,
    SessionItem,
    MessageItem,
    MessageList,
    SessionDeleteResponse,
    MessageCreateResponse,
    DocumentItem,
    DocumentListResponse
)
from uuid import UUID
from datetime import datetime

class SessionService:
    """Manage chat sessions and history."""
    
    async def create_session(self, user_id: int, db: AsyncSession) -> SessionCreate:
        session = Session(
            title=f"Chat {datetime.now():%Y-%m-%d}",
            user_id=user_id,
        )

        db.add(session)
        await db.commit()
        await db.refresh(session)

        return SessionCreate(session_id=session.id)
        
    async def get_session(
        self,
        session_id: UUID,
        db: AsyncSession,
    ) -> Session | None:

        result = await db.execute(
            select(Session)
            .where(Session.id == session_id)
        )

        return result.scalars().first()

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
        if tool_used and len(tool_used) > 100:
            tool_used = tool_used[:97] + "..."

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

        return MessageCreateResponse(status="message added", message_id=mess.id)
        
    async def get_messages(self, session_id: UUID, db: AsyncSession) -> MessageList:
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
                    message_id=m.id,
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
    
    async def get_recent_history(self, session_id: UUID, db: AsyncSession, limit: int = 10) -> MessageList:
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
                    message_id=m.id,
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
    
    async def list_sessions(self, user_id: int, db: AsyncSession) -> SessionList:
        """List all sessions for a user."""
        result = await db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.updated_at.desc())
        )

        sessions = result.scalars().all()
        return SessionList(sessions=[
            SessionItem(session_id=s.id, title=s.title, updated_at=s.updated_at) 
            for s in sessions
        ])    
    
    async def get_session_documents(self, session_id: UUID, db: AsyncSession) -> list[int]:
        result = await db.execute(
            select(SessionDocument.document_id)
            .where(SessionDocument.session_id == session_id)
        )
        return result.scalars().all()

    async def get_session_documents_full(self, session_id: UUID, db: AsyncSession) -> DocumentListResponse:
        """Get the full documents linked to this session."""
        result = await db.execute(
            select(Document)
            .join(SessionDocument, Document.id == SessionDocument.document_id)
            .where(SessionDocument.session_id == session_id)
        )
        docs = result.scalars().all()
        return DocumentListResponse(documents=[
            DocumentItem(
                name=doc.name, 
                public_id=doc.public_id, 
                chunk_count=doc.chunk_count,
                status=doc.status, 
                mime_type=doc.mime_type, 
                file_size_kb=doc.file_size_kb
            ) for doc in docs
        ])

    async def link_document_to_session(self, session_id: UUID, document_id: int, db: AsyncSession) -> None:
        """Link an uploaded document to a session."""
        session_doc = SessionDocument(session_id=session_id, document_id=document_id)
        db.add(session_doc)
        await db.commit()
        
    async def unlink_document_from_session(self, session_id: UUID, document_id: int, db: AsyncSession) -> None:
        """Unlink an uploaded document from a session."""
        result = await db.execute(
            select(SessionDocument).where(
                SessionDocument.session_id == session_id,
                SessionDocument.document_id == document_id
            )
        )
        session_doc = result.scalars().first()
        if session_doc:
            await db.delete(session_doc)
            await db.commit()
        
    async def delete_session(self, session_id: UUID, db: AsyncSession) -> SessionDeleteResponse:
        """Delete a session."""
        result = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalars().first()
        if not session:
            return SessionDeleteResponse(
                status="Not Deleted",
                error="Session not found"
            )
        await db.delete(session)
        await db.commit()     

        return SessionDeleteResponse(status="deleted", session_id=session.id, session_name=session.title)

    async def rename_session(self, session_id: UUID, new_name: str, db: AsyncSession) -> Session | None:
        """Rename a session's title."""
        result = await db.execute(select(Session).where(Session.id == session_id))
        session = result.scalars().first()
        if not session:
            return None
        session.title = new_name
        await db.commit()
        await db.refresh(session)
        return session
