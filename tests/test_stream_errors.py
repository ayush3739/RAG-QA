import asyncio
import json
import sys
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services.chat_service import ChatService


def test_stream_session_not_found_emits_error_and_done():
    """When a session is not found, stream must yield graph_status (failed), error, and done."""
    async def run():
        service = ChatService()
        service.session_service = MagicMock()
        session_id = uuid4()
        mock_db = AsyncMock()
        service.session_service.get_session = AsyncMock(return_value=None)

        events = []
        async for event_type, data in service.stream_chat(
            session_id=session_id,
            question="Hello",
            user_id=1,
            db=mock_db,
        ):
            events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "graph_status" in event_types
        assert "error" in event_types
        assert "done" in event_types

        # Verify graph_status payload
        status_event = next(e[1] for e in events if e[0] == "graph_status")
        status_payload = json.loads(status_event)
        assert status_payload["status"] == "failed"
        assert status_payload["node"] == "session_validation"

        # Verify error message and done marker
        error_event = next(e[1] for e in events if e[0] == "error")
        assert error_event == "Session not found"
        done_event = next(e[1] for e in events if e[0] == "done")
        assert done_event == "[DONE]"

    asyncio.run(run())


def test_stream_unauthorized_user_emits_error_and_done():
    """When session user_id does not match, stream must yield graph_status (failed), error, and done."""
    async def run():
        service = ChatService()
        service.session_service = MagicMock()
        session_id = uuid4()
        mock_db = AsyncMock()
        mock_session = MagicMock()
        mock_session.user_id = 999  # Different from requesting user 1
        service.session_service.get_session = AsyncMock(return_value=mock_session)

        events = []
        async for event_type, data in service.stream_chat(
            session_id=session_id,
            question="Hello",
            user_id=1,
            db=mock_db,
        ):
            events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "graph_status" in event_types
        assert "error" in event_types
        assert "done" in event_types

        status_event = next(e[1] for e in events if e[0] == "graph_status")
        status_payload = json.loads(status_event)
        assert status_payload["status"] == "failed"
        assert status_payload["node"] == "session_validation"

        error_event = next(e[1] for e in events if e[0] == "error")
        assert error_event == "Unauthorized"
        done_event = next(e[1] for e in events if e[0] == "done")
        assert done_event == "[DONE]"

    asyncio.run(run())


def test_stream_rate_limit_error_emits_rate_limit_and_done():
    """When Groq or LLM provider raises 429 / rate limit, stream must yield rate_limit and done events."""
    async def run():
        service = ChatService()
        service.session_service = MagicMock()
        session_id = uuid4()
        mock_db = AsyncMock()
        mock_session = MagicMock()
        mock_session.user_id = 1
        service.session_service.get_session = AsyncMock(return_value=mock_session)
        service.session_service.add_message = AsyncMock(return_value=MagicMock(message_id=101))
        
        mock_history = MagicMock()
        mock_history.messages = []
        service.session_service.get_recent_history = AsyncMock(return_value=mock_history)
        service.session_service.get_session_documents = AsyncMock(return_value=[])

        # Simulate Groq 429 RateLimitError during graph execution
        async def fake_graph_stream_rate_limited(*args, **kwargs):
            raise RuntimeError("Error code: 429 - {'error': {'message': 'Rate limit reached for model groq/llama-3.3-70b-versatile', 'type': 'tokens'}}")
            yield

        with patch("backend.services.chat_service.stream_answer_query_with_graph", side_effect=fake_graph_stream_rate_limited):
            events = []
            async for event_type, data in service.stream_chat(
                session_id=session_id,
                question="Summarize this",
                user_id=1,
                db=mock_db,
            ):
                events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "graph_status" in event_types
        assert "rate_limit" in event_types
        assert "done" in event_types

        # Verify graph_status payload marks the node failed
        status_event = next(e[1] for e in events if e[0] == "graph_status")
        status_payload = json.loads(status_event)
        assert status_payload["status"] == "failed"
        assert status_payload["node"] == "error"
        assert "rate-limited" in status_payload["message"].lower()

        # Verify rate_limit event text
        rate_limit_data = next(e[1] for e in events if e[0] == "rate_limit")
        assert "rate-limited" in rate_limit_data.lower()

        # Verify done event terminates stream
        done_data = next(e[1] for e in events if e[0] == "done")
        assert done_data == "[DONE]"

    asyncio.run(run())


def test_stream_generic_error_emits_error_and_done():
    """When an unexpected agent exception occurs, stream must yield graph_status (failed), error, and done."""
    async def run():
        service = ChatService()
        service.session_service = MagicMock()
        session_id = uuid4()
        mock_db = AsyncMock()
        mock_session = MagicMock()
        mock_session.user_id = 1
        service.session_service.get_session = AsyncMock(return_value=mock_session)
        service.session_service.add_message = AsyncMock(return_value=MagicMock(message_id=102))

        mock_history = MagicMock()
        mock_history.messages = []
        service.session_service.get_recent_history = AsyncMock(return_value=mock_history)
        service.session_service.get_session_documents = AsyncMock(return_value=[])

        async def fake_graph_stream_error(*args, **kwargs):
            raise ValueError("Qdrant embedding dimension mismatch: expected 768, got 1024")
            yield

        with patch("backend.services.chat_service.stream_answer_query_with_graph", side_effect=fake_graph_stream_error):
            events = []
            async for event_type, data in service.stream_chat(
                session_id=session_id,
                question="Explain neural search",
                user_id=1,
                db=mock_db,
            ):
                events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "graph_status" in event_types
        assert "error" in event_types
        assert "done" in event_types

        status_event = next(e[1] for e in events if e[0] == "graph_status")
        status_payload = json.loads(status_event)
        assert status_payload["status"] == "failed"

        error_data = next(e[1] for e in events if e[0] == "error")
        assert "dimension mismatch" in error_data

        done_data = next(e[1] for e in events if e[0] == "done")
        assert done_data == "[DONE]"

    asyncio.run(run())


def test_stream_db_failure_during_error_still_emits_done():
    """If database save fails during error handling (e.g. broken transaction), stream MUST still yield error and done."""
    async def run():
        service = ChatService()
        service.session_service = MagicMock()
        session_id = uuid4()
        mock_db = AsyncMock()
        mock_session = MagicMock()
        mock_session.user_id = 1
        service.session_service.get_session = AsyncMock(return_value=mock_session)
        
        # First add_message succeeds (for user message)
        # Second add_message fails (when trying to save error message)
        service.session_service.add_message = AsyncMock(side_effect=[
            MagicMock(message_id=1),
            Exception("Database transaction rolled back")
        ])

        mock_history = MagicMock()
        mock_history.messages = []
        service.session_service.get_recent_history = AsyncMock(return_value=mock_history)
        service.session_service.get_session_documents = AsyncMock(return_value=[])

        async def fake_graph_stream_crash(*args, **kwargs):
            raise RuntimeError("LLM connection timeout")
            yield

        with patch("backend.services.chat_service.stream_answer_query_with_graph", side_effect=fake_graph_stream_crash):
            events = []
            async for event_type, data in service.stream_chat(
                session_id=session_id,
                question="Test question",
                user_id=1,
                db=mock_db,
            ):
                events.append((event_type, data))

        event_types = [e[0] for e in events]
        assert "graph_status" in event_types
        assert "error" in event_types
        assert "done" in event_types
        assert events[-1] == ("done", "[DONE]")

    asyncio.run(run())


if __name__ == "__main__":
    print("Running test_stream_session_not_found_emits_error_and_done...")
    test_stream_session_not_found_emits_error_and_done()
    print("PASS")

    print("Running test_stream_unauthorized_user_emits_error_and_done...")
    test_stream_unauthorized_user_emits_error_and_done()
    print("PASS")

    print("Running test_stream_rate_limit_error_emits_rate_limit_and_done...")
    test_stream_rate_limit_error_emits_rate_limit_and_done()
    print("PASS")

    print("Running test_stream_generic_error_emits_error_and_done...")
    test_stream_generic_error_emits_error_and_done()
    print("PASS")

    print("Running test_stream_db_failure_during_error_still_emits_done...")
    test_stream_db_failure_during_error_still_emits_done()
    print("PASS")

    print("\nALL 5 STREAMING ERROR TESTS PASSED SUCCESSFULLY!")
