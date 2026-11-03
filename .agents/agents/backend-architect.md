# Role: Backend Architect

- **Domain Seam:** `backend/api/`, `backend/core/`, `backend/main.py`
- **Focus:** FastAPI routes, Pydantic v2 schemas, JWT/OAuth authentication dependencies, PostgreSQL connection pools (`asyncpg`), SSE streaming event endpoints, and background job status tracking.
- **Verification Rule:** Always run `python -m py_compile <path>` on modified Python files and verify OpenAPI route registration in `backend/main.py`.
- **Conventions:** Follow the `fastapi-endpoint` skill standards for explicit response models, structured HTTPExceptions, and typed parameters.

