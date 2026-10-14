# 📋 DocuMind — Project Master TODO & Implementation Checklist

> Grounded directly in [`idea/PRD2.0.md`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/idea/PRD2.0.md), [`PRD2_0_Updated.md`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/PRD2_0_Updated.md), [`idea/implementation_plan.md`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/idea/implementation_plan.md), and [`idea/frontend.md`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/idea/frontend.md).

---

## 🚀 1. Flagship Research Workspace (PRD F3.2 & `frontend.md` §33.7)
*Backend endpoint `POST /api/v1/research` is implemented, but the frontend workspace and client wiring are completely missing.*

- [ ] **1.1 Frontend API Client Method**:
  - Add `api.research(payload: { topic: string, collection?: string, include_web?: boolean, output_format?: 'structured' | 'bullet' | 'prose' })` to [`frontend/src/lib/api.ts`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/lib/api.ts).
- [ ] **1.2 Dedicated Research Workspace Component (`ResearchWorkspace.tsx`)**:
  - Build UI layout matching `frontend.md` §33.7:
    - **Topic Input Header**: Topic search/query bar with template presets (*Executive Brief*, *Deep Dive*, *Comparative Analysis*).
    - **Document Filter**: Multi-select document scope pill bar.
    - **Web Toggle**: Switch to include/exclude live web research.
    - **Agent Progress Timeline**: Visual progress steps (Planner $\to$ Subagent Retrieval $\to$ Evidence Fusion $\to$ Critic).
    - **Report Viewer**: Structured layout rendering `summary`, `key_findings` bullet list, `sources` evidence blocks, and `confidence` meter.
- [ ] **1.3 Sidebar & Route Integration**:
  - Add `Research` navigation item to [`frontend/src/components/Sidebar.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/Sidebar.tsx).
  - Wire `currentTab === "research"` in [`frontend/src/App.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/App.tsx) and connect `handleNewResearch` to launch the research workspace instead of regular chat.

---

## 🧠 2. RAG Core Quality & Chunking Fundamentals (PRD F1.1 - F1.5)

- [ ] **2.1 Dual-Tier Parent-Child Chunking (F1.1)**:
  - *Current state:* Uses single-tier `RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)`.
  - Update [`backend/core/indexer.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/core/indexer.py) to split documents into:
    - **Child chunks (~150 tokens):** Embedded and indexed in pgvector + BM25 for pinpoint retrieval.
    - **Parent chunks (~512 tokens):** Mapped to child chunks in the database.
  - Update [`backend/models/models.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/models/models.py) `Chunk` model with `parent_content` / `parent_id`.
  - Update [`backend/core/retriever.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/core/retriever.py) so search matches on child vectors, but returns parent text to the LLM context.
- [ ] **2.2 Configurable `alpha` in Hybrid Search RRF (F1.2)**:
  - Add configurable `alpha` parameter (`0.0` = pure BM25, `1.0` = pure vector, `0.5` = balanced) in `reciprocal_rank_fusion` inside [`backend/core/retriever.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/core/retriever.py).
- [ ] **2.3 Deterministic Web Escalation in Main Graph (F3.1 & PRD2_0_Updated)**:
  - *Current state:* `router.py` has escalation logic, but `main_graph.py` (which powers `chat_service.py`) routes `document_qa` directly to `generator` without checking confidence.
  - In [`backend/agent/graphs/main_graph.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/agent/graphs/main_graph.py), add conditional branch after `document_qa`: if reranker confidence < 0.4 and web is enabled and query is not document-scoped, escalate to `web_search` once before answer synthesis.

---

## 📄 3. Document Intelligence & Ingestion (PRD F4.1, F4.5 & `frontend.md` §33.5)

- [ ] **3.1 Document Auto-Summary on Upload (F4.1 & `frontend.md` §33.5.24)**:
  - Add `summary` and `key_topics` columns to `models.Document`.
  - In [`backend/api/routes/documents.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/api/routes/documents.py), trigger background LLM summarization immediately after chunk indexing completes (5 bullet points, page count, estimated read time).
  - Display the generated AI Summary card in [`DocumentDetailsView.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/DocumentDetailsView.tsx) and before chat starts in `ChatWorkspace.tsx`.
- [ ] **3.2 Web Page / URL Ingestion (F4.5)**:
  - Add `trafilatura` scraping pipeline to `backend/core/indexer.py`.
  - Add `POST /api/v1/documents/url` endpoint in `backend/api/routes/documents.py` to ingest and index any public article or documentation URL.
  - Add URL input tab in the upload modal of [`DocumentsView.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/DocumentsView.tsx).
- [ ] **3.3 Document Previewer & Chunk Browser (`frontend.md` §33.5.22 - §33.5.27)**:
  - Add side-by-side PDF preview or chunk inspector drawer in [`DocumentDetailsView.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/DocumentDetailsView.tsx) so users can examine the exact chunks indexed for that document.

---

## 💬 4. Conversational UX & Features (PRD F4.2 - F4.4, F5.1 & `frontend.md` §33.6)

- [ ] **4.1 Suggested Follow-Up Questions (F4.2 & `frontend.md` §33.6.21)**:
  - In [`backend/services/chat_service.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/services/chat_service.py), prompt LLM for 3 relevant follow-ups and emit via SSE `metadata` event.
  - In [`backend/api/routes/research.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/api/routes/research.py), populate `follow_up_questions` rather than returning an empty array.
  - In [`frontend/src/components/ChatWorkspace.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/ChatWorkspace.tsx), render clickable follow-up prompt chips below assistant responses that submit the question upon click.
- [ ] **4.2 1-Click Conversation & Report Export (F4.4 & `frontend.md` §33.6.30)**:
  - Add an "Export Chat" button in [`ChatWorkspace.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/ChatWorkspace.tsx) and [`SessionsView.tsx`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/frontend/src/components/SessionsView.tsx).
  - Download formatted Markdown (`# Research Session: {title}\n- **You:** ...\n- **DocuMind:** ...\n_Sources: Page X_`).
- [ ] **4.3 Dynamic Client-Side Model Selection (F5.1)**:
  - Add `model` / `llm_mode` field to `ChatRequest` in [`backend/models/schemas.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/models/schemas.py).
  - Wire the model selector in `ChatWorkspace.tsx` and `SettingsView.tsx` to send the requested provider with each chat message instead of relying on local browser storage alone.

---

## 📊 5. Observability, Metrics & Tracing (PRD Section 7 & `todo.md` #22)

- [ ] **5.1 Structured File Logging (`logs/agent.log`)**:
  - Configure file handler in [`backend/main.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/main.py) or `backend/core/logger.py` writing to `logs/agent.log`.
  - Log: timestamp, query, session_id, routing decision, tools executed, retrieval duration, reranker score, and errors.
- [ ] **5.2 System Metrics Endpoint (`GET /api/v1/metrics`)**:
  - Implement `GET /api/v1/metrics` in [`backend/api/routes/analytics.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/api/routes/analytics.py):
    - Total documents and chunks indexed.
    - Average query response latency.
    - Daily query breakdown by tool (`document_retrieval`, `web_search`, `direct_llm`, `research`).
    - Total thumbs up / thumbs down feedback count.
- [ ] **5.3 LangSmith Tracing Configuration**:
  - Add `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` settings in [`backend/core/config.py`](file:///c:/Users/Ayush%20Maurya/Desktop/hexon/rag-qa/backend/core/config.py) so tracing operates automatically when keys are set in `.env` (matching landing page claim).

---

## ❓ 6. Quiz Generation Mode (PRD F5.2)

- [ ] **6.1 Quiz API Endpoint**:
  - Implement `POST /api/v1/quiz/{document_id}` in `backend/api/routes/documents.py` (or a dedicated `quiz.py` route) invoking `generate_quiz_impl`.
  - Return structured JSON: `[{"question": "...", "options": ["A", "B", "C", "D"], "answer": "A", "source_page": 3}]`.
- [ ] **6.2 Quiz Frontend Interface**:
  - Add a "Test My Knowledge" / "Generate Quiz" button in `DocumentsView.tsx` or `ChatWorkspace.tsx`.
  - Render an interactive MCQ modal with immediate answer verification and source page citations.

---

## 🛡️ 7. Production Hardening & Evaluation (Phase 6 — Planned Quality Gate)

- [ ] **7.1 Automated RAGAS Evaluation Suite (F6.1)**:
  - Create `eval/golden_set.json` (20–30 curated `(query, expected_answer, expected_source)` triples).
  - Create `eval/run_eval.py` assessing `faithfulness`, `context_precision`, `context_recall`, and `answer_relevancy`.
- [ ] **7.2 CI Quality Gate (F6.1)**:
  - Add GitHub Actions workflow running `eval/run_eval.py` on PRs modifying retriever, agent, or prompts.
- [ ] **7.3 Input & Log Guardrails (F6.2)**:
  - Input prompt-injection filter & PII redaction (redacting email/phone/SSN patterns before writing to query logs).
- [ ] **7.4 Retrieval-Layer Guardrail (F6.3)**:
  - Scan retrieved document chunks for indirect prompt injection payloads before sending to the synthesis prompt.
- [ ] **7.5 Output Groundedness Check (F6.4)**:
  - Fast inline groundedness heuristic at request time emitting `grounded: boolean` in SSE metadata.

---

## ✅ Completed Fundamentals (Already Working)

- [x] Hybrid Search (BM25 + pgvector cosine similarity via Reciprocal Rank Fusion)
- [x] CrossEncoder Reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) top-15 $\to$ top-5
- [x] Calibrated confidence scoring (sigmoid normalized 0.0 to 1.0)
- [x] Sliding-window conversational memory (last 6 messages persisted in PostgreSQL)
- [x] Multi-document chat querying within sessions
- [x] Complete JWT Auth, OAuth (Google/GitHub), Refresh Token (HttpOnly cookies), Password reset & Email verification
- [x] Async background ingestion with SSE status polling (`/api/v1/documents/status/{job_id}`)
- [x] Multi-tier LLM fallback chain (Bedrock $\to$ Gemini $\to$ NIM $\to$ Groq $\to$ OpenRouter $\to$ GitHub $\to$ Ollama)
- [x] LangGraph routing architecture (Router, RetrieverSubagent, ResearchGraph, Tavily web search)
- [x] User-facing Inspector Panel (retrieved chunks, vector/BM25/reranker scores, tool trace, routing reasons)
- [x] Thumbs up/down feedback dialog with PostgreSQL persistence (`/api/v1/feedback`)
- [x] User query activity analytics (`/api/v1/analytics/activity`)
- [x] Clean light/dark adaptive design system (no hardcoded dark hexes, refined typography & horizon glows)