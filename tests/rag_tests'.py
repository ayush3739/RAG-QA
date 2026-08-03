import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend.core.retriever import Retriever


pytestmark = pytest.mark.skipif(
    not os.getenv("GITHUB_TOKEN"), reason="GITHUB_TOKEN not set — integration tests skipped"
)


def test_retriever_initializes():
    """Retriever should initialize (with fallbacks) when credentials exist."""
    r = Retriever("test.pdf")
    assert hasattr(r, "similarity_search")
    assert hasattr(r, "answer")


def test_similarity_search_returns_structured_payload():
    """similarity_search should return a dict with expected keys and structured chunks."""
    r = Retriever("test.pdf")
    res = r.similarity_search("who is albert einstein", k=5)
    assert isinstance(res, dict)
    assert set(["chunks", "used_vector_db", "debug", "confidence"]).issubset(set(res.keys()))
    assert isinstance(res["chunks"], list)
    if len(res["chunks"]) > 0:
        c = res["chunks"][0]
        assert set(["chunk_id", "text", "source", "page_label"]).issubset(set(c.keys()))


def test_answer_integration_runs_if_enabled():
    """Run full answer flow — skipped unless RUN_RAG_INTEGRATION is set to '1'."""
    if os.getenv("RUN_RAG_INTEGRATION") != "1":
        pytest.skip("Integration run disabled — set RUN_RAG_INTEGRATION=1 to enable")
    r = Retriever("test.pdf")
    out = r.answer("who is albert einstein", k=5)
    # output may be a string (LLM raw) or structured dict depending on configuration
    assert isinstance(out, (str, dict))

