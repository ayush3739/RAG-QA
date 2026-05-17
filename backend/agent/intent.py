"""Intent Classifier Node — Phase 3."""


INTENT_PROMPT = """Classify this query into exactly one category:
- factual    → specific question answerable from a document
- summarize  → asking for an overview or summary of document content
- web        → requires current/live information not in any document
- direct     → general knowledge question, no document needed
- quiz       → asking to generate questions or test knowledge

Query: {query}
Collection available: {has_collection}

Return ONLY the category word, nothing else."""


def classify_intent(query: str, has_collection: bool) -> str:
    """Classify user intent. Phase 3 placeholder."""
    # TODO: Implement intent classification via fast LLM call
    return "factual"  # default for now
