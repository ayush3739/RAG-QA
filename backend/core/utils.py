# backend/core/utils.py
import re
import os
from typing import Tuple, Optional

from qdrant_client import QdrantClient


def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def vector_db_health_check(url: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """Check Qdrant vector DB health.

    Returns (ok: bool, error_message: Optional[str]).
    Does not raise; callers can log or surface the error.
    """
    qdrant_url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        # simple call to validate connectivity
        client.get_collections()
        return True, None
    except Exception as e:
        return False, str(e)