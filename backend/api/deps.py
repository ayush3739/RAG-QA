"""Dependency Injection for FastAPI."""

from backend.core.config import settings
from backend.core.retriever import Retriever


async def get_settings():
    """Provide app settings."""
    return settings


async def get_retriever(collection_name: str) -> Retriever:
    """Provide retriever for a collection."""
    return Retriever(collection_name=collection_name)
