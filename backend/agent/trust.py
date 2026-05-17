"""Trust Layer — Confidence Scoring & Citations — Phase 3."""


def compute_confidence(reranker_score: float) -> float:
    """Compute confidence from reranker score (0-1)."""
    return min(max(reranker_score, 0.0), 1.0)


def extract_citations(chunks: list) -> list:
    """Extract page numbers and section info from retrieved chunks."""
    citations = []
    for chunk in chunks:
        citations.append({
            "page": chunk.get("page_label", "N/A"),
            "source": chunk.get("source", "N/A"),
            "excerpt": chunk.get("page_content", "")[:200]  # first 200 chars
        })
    return citations


def should_refuse_answer(confidence: float, threshold: float = 0.3) -> bool:
    """Decide if confidence is too low to answer."""
    return confidence < threshold
