"""Research API - structured agent-routed research endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agent.router import answer_query
from backend.api.deps import get_current_user
from backend.db.base import get_db
from backend.models import models
from backend.models.schemas import ResearchRequest, ResearchResponse

router = APIRouter()


def _extract_key_findings(answer: str, max_items: int = 5) -> list[str]:
    lines = []
    for raw_line in answer.splitlines():
        line = raw_line.strip().lstrip("-*0123456789. ").strip()
        if line:
            lines.append(line)

    if len(lines) >= 2:
        return lines[:max_items]

    sentences = [part.strip() for part in answer.replace("\n", " ").split(".") if part.strip()]
    return [sentence + "." for sentence in sentences[:max_items]]


def _format_summary(answer: str, output_format: str) -> str:
    if output_format == "bullet":
        findings = _extract_key_findings(answer)
        return "\n".join(f"- {finding}" for finding in findings) or answer
    return answer


async def _document_ids_from_collection(
    collection: str | None,
    user_id: int,
    db: AsyncSession,
) -> list[int]:
    if not collection:
        return []

    filters = [models.Document.user_id == user_id]
    if collection.isdigit():
        filters.append(models.Document.id == int(collection))
    else:
        filters.append(models.Document.public_id == collection)

    result = await db.execute(select(models.Document).where(*filters))
    document = result.scalars().first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Collection/document not found for this user.",
        )

    return [document.id]


@router.post("/research", response_model=ResearchResponse)
async def research(
    request: ResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user),
):
    """Run the same agent router as chat and return a structured research report."""
    document_ids = await _document_ids_from_collection(
        collection=request.collection,
        user_id=current_user.id,
        db=db,
    )

    result = await answer_query(
        query=request.topic,
        document_ids=document_ids,
        db=db,
        history=[],
        include_web=request.include_web,
    )

    summary = _format_summary(result["answer"], request.output_format)

    return ResearchResponse(
        summary=summary,
        key_findings=_extract_key_findings(result["answer"]),
        sources=result.get("sources", []),
        confidence=result.get("confidence"),
        tool_trace=result.get("tool_trace", []),
        follow_up_questions=[],
    )
