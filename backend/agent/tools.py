"""Agent Tools — Phase 3."""


async def retrieve_from_document(query: str, collection: str):
    """Tool: Search document via RAG."""
    # TODO: Implement document retrieval
    pass


async def web_search(query: str):
    """Tool: Search the web via Tavily."""
    # TODO: Implement web search
    pass


async def direct_answer(query: str):
    """Tool: Answer directly via LLM (no retrieval)."""
    # TODO: Implement direct LLM answer
    pass


async def summarize_document(collection: str):
    """Tool: Generate document summary."""
    # TODO: Implement document summarization
    pass


async def generate_quiz(collection: str, num_questions: int = 5):
    """Tool: Generate quiz from document."""
    # TODO: Implement quiz generation
    pass
