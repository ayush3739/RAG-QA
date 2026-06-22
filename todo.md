Based on the PRD + implementation plan and the current codebase, these are the main things still left.

**Highest Priority**
1. `/api/v1/research` needs to be completed as the flagship structured endpoint:
   - Always return structured JSON.
   - Support `include_web=false`.
   - Support `output_format=structured | bullet | prose`.
   - Return `summary`, `key_findings`, `sources`, `confidence`, `tool_trace`, `follow_up_questions`.

2. Align tool-calling exactly with PRD:
   - PRD says general questions should be zero tool calls, not `direct_answer` as a tool.
   - Current code has `direct_answer` as a tool. It works, but it is a PRD deviation.
   - Decide whether to keep it pragmatically or remove it for strict PRD alignment.

3. Add real follow-up questions:
   - API should optionally return 3 suggested follow-ups.
   - Frontend should show clickable follow-up pills.

4. Add document auto-summary after upload:
   - Generate summary/key topics after indexing.
   - Store it.
   - Show it before chat starts.

5. Finish source/citation quality:
   - Frontend confidence badge: high/medium/low.
   - Better page/source display like `Page 4 - Section 2.1`.
   - Ensure low-confidence doc answers say “not found” instead of over-answering.

**Backend/API Gaps**
6. Metrics endpoint:
   - `GET /api/v1/metrics`.

7. Feedback endpoint needs production completion:
   - `POST /api/v1/feedback`.
   - Store thumbs up/down with query, answer, tool trace, confidence.

8. Session polish:
   - Named sessions.
   - Rename sessions.
   - Clear/delete session.
   - Restore full chat history cleanly.
   - Store full `tool_trace`, ideally JSONB, not just compact `tool_used`.

9. Consistent error response schema:
   - PRD wants all errors as structured JSON like:
   ```json
   {"error": "...", "code": "..."}
   ```

**Retrieval/RAG Quality**
10. Verify parent-child chunking:
   - Child chunks retrieved.
   - Parent chunks sent to LLM.
   - Re-index removes old chunks cleanly.

11. Confirm hybrid search config:
   - `alpha` configurable.
   - No duplicate chunks after BM25/vector merge.

12. Confirm reranking config:
   - Reranker toggle via env.
   - Latency measured under target.

13. Better “I don’t know” behavior:
   - If doc confidence is low and web is disabled or not useful, answer honestly.

**Frontend/UX**
14. Build real chat UI beyond testing pages:
   - Sidebar sessions.
   - New chat.
   - Restore session.
   - Source cards.
   - Retrieved chunks view.
   - Follow-up pills.

15. Export conversation:
   - Markdown/plain text export with Q&A and citations.

16. Multi-format ingestion:
   - `.docx`
   - `.txt`
   - `.md`
   - URL ingestion with `trafilatura`.

17. LLM toggle:
   - Local/cloud switch without restart.

18. Quiz endpoint/UI:
   - `POST /api/v1/quiz/{collection}` or equivalent.
   - Return valid quiz JSON with source pages.

**Production Hardening**
19. RAGAS eval suite:
   - `eval/golden_set.json`.
   - `eval/run_eval.py`.
   - Metrics: faithfulness, context precision, context recall, answer relevancy.

20. CI quality gate:
   - GitHub Actions workflow.
   - Fail PRs when retrieval/agent quality regresses.

21. Guardrails:
   - Input prompt-injection detection.
   - PII redaction for logs.
   - Retrieved-chunk injection scanning.
   - Output groundedness check with `grounded` and `groundedness_score`.

22. Observability:
   - Log agent decisions/tool calls/errors to `logs/agent.log`.

In short: the core chat + tool-routed agent path is mostly there now. The biggest remaining PRD items are `/research`, follow-ups, production session/source polish, multi-format ingestion, feedback/metrics, and the Phase 6 eval/guardrail layer.