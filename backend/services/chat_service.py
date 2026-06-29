from uuid import UUID
from typing import AsyncGenerator
import json

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.session_service import SessionService
from backend.agent.router import answer_query


class ChatService:

    def __init__(self):
        self.session_service = SessionService()

    async def stream_chat(
        self,
        session_id: UUID,
        question: str,
        user_id: int,
        db: AsyncSession,
    ) -> AsyncGenerator[tuple[str, object], None]:

        # 1. Validate session

        session = await self.session_service.get_session(
            session_id=session_id,
            db=db,
        )

        if not session:
            yield ("error", "Session not found")
            return

        if session.user_id != user_id:
            yield ("error", "Unauthorized")
            return

        # 2. Save user message

        await self.session_service.add_message(
            session_id=session_id,
            role="user",
            content=question,
            db=db,
        )

        # 3. Load history
        # Already includes the user message saved above — no need to append it again.

        history = await self.session_service.get_recent_history(
            session_id=session_id,
            db=db,
            limit=6,
        )

        # 4. Load attached documents

        document_ids = await self.session_service.get_session_documents(
            session_id=session_id,
            db=db,
        )

        # 4b. Get document names for context injection
        document_names: list[str] = []
        if document_ids:
            doc_list = await self.session_service.get_session_documents_full(
                session_id=session_id,
                db=db,
            )
            document_names = [doc.name for doc in doc_list.documents]

        # 5. Route through the agent layer

        try:
            result = await answer_query(
                query=question,
                document_ids=document_ids,
                db=db,
                history=[
                    {"role": msg.role, "content": msg.content}
                    for msg in history.messages[:-1]
                ],
                include_web=True,
                document_names=document_names,
            )
        except Exception as exc:
            error_message = (
                "Sorry, I couldn't complete this response because the agent "
                f"failed: {exc}"
            )
            await self.session_service.add_message(
                session_id=session_id,
                role="assistant",
                content=error_message,
                db=db,
                confidence=0.0,
                sources=[],
                tool_used="error",
            )
            yield ("error", error_message)
            yield ("done", "[DONE]")
            return

        full_response = result["answer"]

        # The agent currently returns a full synthesized answer. Keep the SSE
        # contract stable by emitting it as token events.
        for token in full_response.split():
            yield ("token", token + " ")

        # 9. Save assistant message

        sources = result.get("sources", [])
        confidence = result.get("confidence")
        stored_confidence = confidence if confidence is not None else 0.0
        tool_trace = result.get("tool_trace", [])
        routing_reason = result.get("routing_reason", "")
        used_vector_db = result.get("used_vector_db", False)
        chunks = result.get("chunks", [])
        retrieved_chunks = result.get("retrieved_chunks", [])

        message_id = None
        try:
            msg_res = await self.session_service.add_message(
                session_id=session_id,
                role="assistant",
                content=full_response,
                db=db,
                confidence=stored_confidence,
                sources=sources,
                tool_used=" -> ".join(tool_trace) if tool_trace else None,
            )
            message_id = msg_res.message_id
        except Exception as exc:
            yield ("error", f"Answer was generated but history save failed: {exc}")
            yield ("done", "[DONE]")
            return

        # 10. Metadata

        yield (
            "metadata",
            json.dumps({
                "message_id": message_id,
                "confidence": confidence,
                "documents": document_ids,
                "chunks_found": len(chunks),
                "used_vector_db": used_vector_db,
                "sources": sources,
                "chunks": retrieved_chunks,
                "tool_trace": tool_trace,
                "routing_reason": routing_reason,
            }),
        )

        # 11. Done

        yield ("done", "[DONE]")
