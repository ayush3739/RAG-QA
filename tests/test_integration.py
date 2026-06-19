import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
import uuid

from backend.main import app

DUMMY_EMAIL_A = "userA@example.com"
DUMMY_EMAIL_B = "userB@example.com"
DUMMY_PASSWORD = "securepassword"

@pytest.fixture
def anyio_backend():
    return 'asyncio'

# Helper to register and login
async def register_and_login(client: AsyncClient, email: str) -> str:
    rand_email = f"test_{uuid.uuid4().hex}@example.com"
    
    # Register
    res = await client.post(
        "/api/v1/auth/register",
        data={
            "name": "Integration User",
            "email": rand_email,
            "password": DUMMY_PASSWORD
        }
    )
    assert res.status_code == 201
    
    # Login
    res = await client.post(
        "/api/v1/auth/login",
        data={
            "username": rand_email,
            "password": DUMMY_PASSWORD
        }
    )
    assert res.status_code == 200
    return res.json()["access_token"]


@pytest.mark.anyio
async def test_full_message_flow_and_ownership():
  async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
    # 1. Register & Login User A and User B
    token_a = await register_and_login(client, DUMMY_EMAIL_A)
    token_b = await register_and_login(client, DUMMY_EMAIL_B)
    
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    # 2. Upload Document (User A)
    with patch("backend.api.routes.documents._run_index_job_async", new_callable=AsyncMock):
        res = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", b"%PDF-1.4 dummy content", "application/pdf")},
            headers=headers_a
        )
        assert res.status_code == 200
        doc_public_id = res.json()["document_id"]
        
    # 3. Create Session (User A)
    res = await client.post("/api/v1/sessions/", headers=headers_a)
    assert res.status_code == 201
    session_id = res.json()["session_id"]
    
    # Let's do another upload and auto-link to session_id to test that!
    with patch("backend.api.routes.documents._run_index_job_async", new_callable=AsyncMock):
        res = await client.post(
            "/api/v1/documents/upload",
            files={"file": ("test2.pdf", b"%PDF-1.4 dummy content", "application/pdf")},
            data={"session_id": session_id},
            headers=headers_a
        )
        assert res.status_code == 200

    # 4. Check Ownership (User B tries to access User A's session)
    res = await client.get(f"/api/v1/sessions/{session_id}/history", headers=headers_b)
    assert res.status_code == 404  # Not found or unauthorized
    
    # User B tries to delete User A's session
    res = await client.delete(f"/api/v1/sessions/{session_id}", headers=headers_b)
    assert res.status_code == 404

    # 5. Chat via SSE (User A)
    # We must mock LLM stream and Retriever to avoid actual API calls
    async def fake_stream(*args, **kwargs):
        yield "Hello"
        yield " World"

    with patch("backend.services.llm_provider.LLMProvider.stream", side_effect=fake_stream), \
         patch("backend.core.retriever.Retriever.similarity_search", new_callable=AsyncMock) as mock_search:
             
        mock_search.return_value = {"chunks": [], "confidence": 0.9, "used_vector_db": True}
        
        res = await client.post(
            f"/api/v1/chat/{session_id}",
            json={"question": "What is in the document?"},
            headers=headers_a
        )
        assert res.status_code == 200
        content = res.text
        assert "data: Hello" in content
        assert "data:  World" in content
        assert "event: done" in content

    # 6. History Persists (User A)
    res = await client.get(f"/api/v1/sessions/{session_id}/history", headers=headers_a)
    assert res.status_code == 200
    history = res.json()["messages"]
    
    # We should have exactly 2 messages: user's question, and assistant's response
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is in the document?"
    
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello World" # Stream chunks combined
