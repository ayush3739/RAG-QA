"""Research API — Autonomous research endpoint."""

from fastapi import APIRouter
from backend.models.schemas import ResearchRequest, ResearchResponse

router = APIRouter()


@router.post("/research")
async def research(request: ResearchRequest):
    """Autonomous research endpoint — structured JSON report."""
    # TODO: Implement LangGraph agent
    return ResearchResponse(
        summary="Research summary here",
        key_findings=[],
        sources=[],
        confidence=0.8,
        tool_trace=["retrieve_from_document"],
        follow_up_questions=[]
    )
