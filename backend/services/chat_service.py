from uuid import UUID
from typing import AsyncGenerator
import json
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.session_service import SessionService
from backend.services.analytics_service import AnalyticsService
from backend.services.llm_provider import LLMProvider
from backend.agent.graph_agent import stream_answer_query_with_graph


logger = logging.getLogger(__name__)


class ChatService:

    def __init__(self):
        self.session_service = SessionService()
        self.analytics_service = AnalyticsService()

    async def stream_chat(
        self,
        session_id: UUID,
        question: str,
        user_id: int,
        db: AsyncSession,
        mode :str = "auto",
    ) -> AsyncGenerator[tuple[str, object], None]:

        # 1. Validate session and initialize chat
        try:
            session = await self.session_service.get_session(
                session_id=session_id,
                db=db,
            )

            if not session:
                yield ("graph_status", json.dumps({
                    "node": "session_validation",
                    "label": "Session Not Found",
                    "status": "failed",
                    "message": "Session not found",
                }))
                yield ("error", "Session not found")
                yield ("done", "[DONE]")
                return

            if session.user_id != user_id:
                yield ("graph_status", json.dumps({
                    "node": "session_validation",
                    "label": "Unauthorized Access",
                    "status": "failed",
                    "message": "Unauthorized",
                }))
                yield ("error", "Unauthorized")
                yield ("done", "[DONE]")
                return

            # 2. Save user message
            await self.session_service.add_message(
                session_id=session_id,
                role="user",
                content=question,
                db=db,
            )

            # 2b. Track query usage
            # it gets as a deps at the chat api level 

            # 3. Load history
            # Already includes the user message saved above — no need to append it again.
            history = await self.session_service.get_recent_history(
                session_id=session_id,
                db=db,
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

        except Exception as prep_exc:
            logger.exception(
                "[chat:prep_error] session_id=%s user_id=%s",
                session_id,
                user_id,
            )
            prep_error = f"Failed to initialize chat: {prep_exc}"
            yield ("graph_status", json.dumps({
                "node": "initialization",
                "label": "Initialization Error",
                "status": "failed",
                "message": prep_error,
            }))
            yield ("error", prep_error)
            yield ("done", "[DONE]")
            return

        # 5. Route through the agent layer

        try:
            result = None
            async for event_type, data in stream_answer_query_with_graph(
                query=question,
                document_ids=document_ids,
                db=db,
                history=[
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "citations": msg.citations,
                    }
                    for msg in history.messages[:-1]
                ],
                include_web=True,
                document_names=document_names,
                mode = mode,
            ):
                if event_type == "agent_result":
                    result = data
                    continue
                yield (event_type, data)

            if result is None:
                raise RuntimeError("Graph agent did not return a result")
        except Exception as exc:
            logger.exception(
                "[chat:error] session_id=%s user_id=%s",
                session_id,
                user_id,
            )
            exc_str = str(exc).lower()
            is_rate_limited = (
                "rate limit" in exc_str
                or "ratelimit" in exc_str
                or "resource_exhausted" in exc_str
                or "resourceexhausted" in exc_str
                or "quota" in exc_str
                or "429" in exc_str
                or "too many requests" in exc_str
                or "exhausted" in exc_str
            )

            if is_rate_limited:
                error_message = (
                    "The AI provider is currently busy or rate-limited. "
                    "Please wait a moment and try again."
                )
                event_type_to_yield = "rate_limit"
                node_label = "Provider Rate Limited"
            else:
                error_message = (
                    "Sorry, I couldn't complete this response because the agent "
                    f"encountered an error: {exc}"
                )
                event_type_to_yield = "error"
                node_label = "Agent Execution Error"

            # 1. Immediately yield graph status update marking the failure so UI stops spinning
            yield ("graph_status", json.dumps({
                "node": "error",
                "label": node_label,
                "status": "failed",
                "message": error_message,
            }))

            # 2. Persist error message safely without letting DB transaction failures block SSE
            try:
                await self.session_service.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=error_message,
                    db=db,
                    confidence=0.0,
                    sources=[],
                    tool_used="error",
                )
            except Exception as db_exc:
                logger.error("[chat:error_db_save_failed] %s", db_exc)
                try:
                    await db.rollback()
                except Exception:
                    pass

            # 3. Always yield error event and done marker to guarantee stream contract
            yield (event_type_to_yield, error_message)
            yield ("done", "[DONE]")
            return

        full_response = result["answer"]

        yield ("answer", json.dumps(full_response))

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
                "llm_telemetry": LLMProvider.get_last_telemetry(),
            }),
        )

        # 11. Done

        yield ("done", "[DONE]")
