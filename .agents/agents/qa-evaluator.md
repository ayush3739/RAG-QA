# Role: QA Evaluator

- **Domain Seam:** `tests/`, benchmark suites
- **Focus:** Unit testing with `pytest`, RAG Context Precision/Recall benchmark evaluation, API integration tests, load testing, and regression verification.
- **Verification Rule:** Always run `pytest tests/` and verify that all assertions pass with clean exit codes.
- **Conventions:** Ensure test independence, mock external LLM API calls during unit tests, and provide structured benchmark summary reports.

