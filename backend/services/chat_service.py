from uuid import UUID
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.session_service import SessionService
from backend.core.retriever import Retriever
from backend.services.llm_provider import LLMProvider


class ChatService:

    def __init__(self):
        self.session_service = SessionService()
        self.llm = LLMProvider()

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

        # 5. Retrieval

        all_chunks = []
        confidence = 0.0
        used_vector_db = False

        for document_id in document_ids:

            retriever = Retriever(
                document_id=document_id,
                db=db,
            )

            result = await retriever.similarity_search(question)

            all_chunks.extend(result["chunks"])

            confidence = max(
                confidence,
                result["confidence"] or 0.0,
            )

            used_vector_db = (
                used_vector_db
                or result["used_vector_db"]
            )

        # 6. Build context

        context = "\n\n".join(
            chunk["text"]
            for chunk in all_chunks[:10]
        )

        # 7. Build messages
        # System prompt first, then history (which already contains the user question).

        messages = [
            {
                "role": "system",
                "content": f"""You are DocuMind, a document question-answering assistant.

Answer using the provided document context.
If the answer is not present in the context, say so clearly.

Context:
{context}""",
            }
        ]

        for msg in history.messages:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        # 8. LLM stream

        full_response = ""

        async for token in self.llm.stream(messages):
            full_response += token
            yield ("token", token)

        # 9. Save assistant message

        sources = [
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page_label"],
                "source": chunk["source"],
            }
            for chunk in all_chunks[:10]
        ]

        await self.session_service.add_message(
            session_id=session_id,
            role="assistant",
            content=full_response,
            db=db,
            confidence=confidence,
            sources=sources,
        )

        # 10. Metadata

        yield (
            "metadata",
            {
                "confidence": confidence,
                "documents": document_ids,
                "chunks_found": len(all_chunks),
                "used_vector_db": used_vector_db,
            },
        )

        # 11. Done

        yield ("done", {})