import os
from types import SimpleNamespace


def _ensure_token():
    # tests create Retriever which checks GITHUB_TOKEN; provide a dummy one for unit tests
    os.environ.setdefault("GITHUB_TOKEN", "test-token")


def test_similarity_search_uses_bm25(monkeypatch):
    """When vector DB is unavailable, similarity_search should use BM25 results."""
    _ensure_token()
    from backend.core.retriever import Retriever

    r = Retriever("test.pdf")
    # simulate Qdrant unavailable
    r.vector_db = None

    class FakeBM25:
        def get_scores(self, tokens):
            return [0.1, 0.9, 0.2]

    r.bm25 = FakeBM25()
    r.bm25_texts = ["first chunk", "second chunk", "third chunk"]
    r.bm25_meta = [
        {"chunk_id": "cid1", "page_label": "1", "source": "doc"},
        {"chunk_id": "cid2", "page_label": "2", "source": "doc"},
        {"chunk_id": "cid3", "page_label": "3", "source": "doc"},
    ]

    res = r.similarity_search("einstein", k=3)

    assert isinstance(res, dict)
    assert "chunks" in res
    # highest bm25 score should correspond to cid2
    chunk_ids = [c.get("chunk_id") for c in res["chunks"]]
    assert "cid2" in chunk_ids


def test_rerank_attaches_scores(monkeypatch):
    _ensure_token()
    from backend.core.retriever import Retriever

    r = Retriever("test.pdf")

    class Doc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    chunks = [Doc("a"), Doc("b"), Doc("c")]

    # monkeypatch the reranker's predict to return deterministic scores
    monkeypatch.setattr(r.reranker, "predict", lambda pairs: [0.2, 0.9, 0.5])

    ranked, max_score = r.rerank_("query", chunks, top_n=3)
    assert max_score == 0.9
    # ensure metadata attached
    assert all(hasattr(c, "metadata") and "reranker_score" in c.metadata for c in ranked)
    # ensure order is descending by score
    scores = [c.metadata["reranker_score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)


def test_generate_response_calls_llm(monkeypatch):
    _ensure_token()
    from backend.core.retriever import Retriever

    r = Retriever("test.pdf")

    # replace openai_client with a fake that returns a predictable structure
    fake_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="FAKE ANSWER"
                )
            )
        ]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda *a, **k: fake_resp)))
    r.openai_client = fake_client

    out = r.generate_response(
        "who is einstein",
        {
            "chunks": [{"chunk_id": "cid1", "source": "doc", "page_label": "1", "text": "some context"}],
            "used_vector_db": True,
            "debug": {},
            "confidence": 0.5,
        },
    )
    assert isinstance(out, dict)
    assert out["answer"] == "FAKE ANSWER"
    assert isinstance(out["citations"], list)
    assert out["citations"][0]["chunk_id"] == "cid1"
    assert out["used_vector_db"] is True
