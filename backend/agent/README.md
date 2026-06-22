# Agent Layer

This folder contains DocuMind's tool-routed RAG layer. The API routes should not decide how to answer a question directly; they call this agent layer, and the agent chooses the right tool path.

## Files

### `tools.py`

Tool schemas and backend implementations used by the router.

The file intentionally separates two concerns:

- LLM-visible LangChain tool schemas expose only safe arguments such as `query` and `num_questions`.
- Backend implementation functions receive private runtime context such as `db`, `document_ids`, `user_id`, session state, and API keys from Python code.

The model never sees `db`, `document_ids`, `user_id`, `session_id`, JWTs, or secrets. It only chooses a tool and safe arguments. The backend merges that with trusted context before execution.

### LLM-visible tool schemas

These are decorated with `@tool` and can be passed to LangChain/OpenAI-style tool binding:

- `retrieve_from_document(query)`
  - Schema: search uploaded documents for the user's query.
  - LLM-visible args: `query`.

- `web_search(query)`
  - Schema: search web/current information.
  - LLM-visible args: `query`.

- `summarize_document(query)`
  - Schema: summarize or explain the uploaded document.
  - LLM-visible args: `query`.

- `generate_quiz(query, num_questions=5)`
  - Schema: generate quiz/practice questions from the uploaded document.
  - LLM-visible args: `query`, `num_questions`.

- `direct_answer(query)`
  - Schema: answer directly without document retrieval or live web.
  - LLM-visible args: `query`.

These schema functions deliberately raise if called directly. They exist so the LLM can choose tools safely; real execution happens through the implementation functions below.

`TOOLS` contains these LangChain tool schema objects:

```python
TOOLS = [
    retrieve_from_document,
    web_search,
    summarize_document,
    generate_quiz,
    direct_answer,
]
```

`OPENAI_TOOL_SCHEMAS` contains the same public tool contract in OpenAI-compatible function-calling format. The router passes this list to `LLMProvider.tool_call(...)` for providers that support native tool calls.

### Backend implementation functions

- `retrieve_from_document_impl(query, document_ids, db)`
  - Runs the existing `Retriever` over the documents linked to the session.
  - Returns retrieved chunks, retrieval confidence, and vector-search debug fields.
  - Used when the query is about uploaded/indexed document content.

- `web_search_impl(query)`
  - Calls Tavily directly through `httpx`.
  - Returns web results, web source metadata, and a fixed web confidence.
  - Respects `settings.enable_web_search` and `settings.tavily_api_key`.

- `direct_answer_impl(query)`
  - Calls `LLMProvider` without retrieval.
  - Used for simple math, general knowledge, greetings, coding/help questions, or reasoning that does not need the uploaded document or live web.

- `summarize_document_impl(query, document_ids, db)`
  - Loads indexed chunks from Postgres and asks the LLM for a concise document summary.
  - Used when the user asks for overview, key points, or summary.

- `generate_quiz_impl(query, document_ids, db, num_questions=5)`
  - Loads indexed chunks and asks the LLM to create MCQs as JSON.
  - Used when the user asks for quiz/test/practice questions.

Helper functions:

- `_load_document_chunks(...)`
  - Reads ordered chunks for one or more documents from Postgres.

- `_format_context(...)`
  - Converts chunks into prompt context with chunk/page/source metadata.

- `_sources_from_chunks(...)`
  - Converts chunks into frontend/API source objects.

- `_parse_json_array(...)`
  - Best-effort parser for quiz JSON returned by the LLM.

Execution pattern:

```python
# LLM chooses:
{"name": "retrieve_from_document", "args": {"query": "What is ACID?"}}

# Backend executes with private context:
await retrieve_from_document_impl(
    query=tool_call["args"]["query"],
    document_ids=session_document_ids,
    db=db,
)
```

This is the production pattern used by most tool-calling systems: the model decides what to do, while backend code decides how and where to do it.

### `router.py`

The orchestration layer for one user query.

Main entry point:

```python
await answer_query(
    query=question,
    document_ids=document_ids,
    db=db,
    history=history,
    include_web=True,
)
```

Responsibilities:

- Ask an LLM routing prompt which tool or tools to use.
- Support multi-tool compound queries, for example:
  - document + web: "What is this document about and today's weather?"
  - document + direct answer: "What is this document about and Newton's third law?"
- Execute the selected tools by calling `*_impl` functions with backend context.
- Normalize raw reranker confidence into `0.0-1.0` with a sigmoid.
- Escalate low-confidence document retrieval to web search once when web is enabled.
- Synthesize the final answer from document/web/direct-answer context.
- Return answer text plus metadata for SSE/frontend display.

Important functions:

- `_select_tool(...)`
  - Uses native provider tool calls for OpenAI-compatible providers (`github`, `groq`).
  - The LLM returns only public tool names and args, for example:
    ```json
    {"name": "retrieve_from_document", "args": {"query": "What is ACID?"}}
    ```
  - Falls back to the strict JSON routing prompt for providers that do not expose native tool calls through `LLMProvider`.
  - The routing prompt explicitly says not to use document retrieval just because documents are attached.

- `_fallback_select_tools(...)`
  - Heuristic fallback if router output is not valid JSON.

- `_synthesize(...)`
  - Builds the final answer prompt and calls `LLMProvider.invoke`.
  - Keeps document answers grounded and page-cited.
  - Allows longer answers up to 1000 words when the user asks for detail/summary.

- `_direct_general_context(...)`
  - For compound document + general-knowledge questions, answers only the general-knowledge portion so the final synthesis can combine it with document-grounded content without claiming it came from the document.

- `_normalize_confidence(...)`
  - Converts CrossEncoder raw scores/logits into a frontend-friendly `0.0-1.0` confidence value.

Return shape:

```python
{
    "answer": str,
    "sources": list[dict],
    "confidence": float | None,
    "tool_trace": list[str],
    "used_vector_db": bool,
    "chunks": list[dict],
    "retrieved_chunks": list[dict],
    "routing_reason": str,
}
```

## Runtime Flow

The current chat path is:

1. `backend/api/routes/chat.py`
   - Handles HTTP/SSE.
   - Authenticates the user.
   - Delegates to `ChatService`.

2. `backend/services/chat_service.py`
   - Validates session ownership.
   - Saves the user message.
   - Loads recent history and linked document IDs.
   - Calls `answer_query(...)`.
   - Emits token events, metadata event, then done event.
   - Saves assistant response or an assistant error message.

3. `backend/agent/router.py`
   - Asks the LLM for native tool calls when supported.
   - Validates tool names against allowed tools.
   - Injects backend-only context such as `document_ids` and `db`.
   - Executes implementation functions and synthesizes the response.

4. `backend/agent/tools.py`
   - Performs retrieval, web search, direct answer, summary, or quiz generation.

## SSE Metadata

The frontend receives streamed answer text first, then metadata:

```json
{
  "confidence": 0.82,
  "documents": [1, 2],
  "chunks_found": 5,
  "used_vector_db": true,
  "sources": [],
  "chunks": [],
  "tool_trace": ["retrieve_from_document"],
  "routing_reason": "asks about uploaded document content"
}
```

This lets the UI show the answer immediately while still displaying citations/chunks after the model finishes.

## Current Limitations

- The final answer is generated with `LLMProvider.invoke()` and then emitted as word-like SSE token events. It is not true provider-token streaming yet.
- `research.py` is still a placeholder and does not call this router yet.
- Native tool calls are currently enabled for OpenAI-compatible providers handled by `LLMProvider` (`github`, `groq`). Ollama still uses the JSON routing fallback unless a compatible tool-call adapter is added for the selected local model.
- `tool_used` in the `messages` table is a compact string. Full traces are returned in SSE metadata; for production, a separate JSONB `tool_trace` column would be cleaner.
- Retrieval confidence is based on reranker scores, normalized with sigmoid. This is useful for display/routing but is not a full groundedness check.
