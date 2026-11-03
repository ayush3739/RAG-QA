---
name: fastapi-endpoint
description: Best practices, patterns, and conventions for creating and extending FastAPI routes in KnowVex. Use whenever adding or refactoring an API endpoint, Pydantic schema, authentication dependency, or background task.
---

# FastAPI Endpoint Engineering in KnowVex

This skill defines the canonical pattern for building and extending production endpoints in the KnowVex backend (`backend/api/routes/`).

---

## 1. Core Principles

1. **Strict Type Safety & Schemas:** Every endpoint must have explicit `response_model` and input models defined via Pydantic v2. Never accept raw unvalidated dictionaries.
2. **Explicit Dependency Injection (`Depends`):**
   - Use `get_current_user` for authenticated routes.
   - Use `get_db` or session dependencies for database interactions.
3. **Structured HTTP Exceptions:**
   - Never raise generic `Exception`. Use `HTTPException(status_code=..., detail=...)`.
   - `401 Unauthorized` for missing/invalid JWT.
   - `403 Forbidden` for permission or unverified email issues.
   - `404 Not Found` for missing resources.
   - `422 Unprocessable Entity` for validation issues.
   - `429 Too Many Requests` for rate limits.
4. **SSE Streaming Standards:** For AI token streaming, always use `StreamingResponse(stream_generator(), media_type="text/event-stream")` with standardized event names (`token`, `answer`, `metadata`, `error`, `[DONE]`).

---

## 2. Standard Endpoint Template

```python
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional
from backend.api.deps import get_current_user

router = APIRouter(prefix="/example", tags=["Example"])

# 1. Pydantic Models
class ExampleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    document_ids: List[str] = Field(default_factory=list)

class ExampleResponse(BaseModel):
    id: str
    title: str
    status: str

# 2. Route Implementation
@router.post("/", response_model=ExampleResponse, status_code=status.HTTP_201_CREATED)
async def create_example(
    payload: ExampleRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Creates a new resource scoped to the authenticated user.
    """
    try:
        user_id = current_user.get("id")
        # Domain logic execution
        return ExampleResponse(
            id="res_123",
            title=payload.title,
            status="active"
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request."
        )
```

---

## 3. Verification Checklist

Whenever an agent builds or alters an endpoint:
- [ ] Run `python -m py_compile <path-to-route-file>` to catch syntax and import errors.
- [ ] Ensure the route is registered in `backend/main.py` (`app.include_router(...)`).
- [ ] Ensure OpenAPI docs at `/docs` reflect the correct response models and error responses.

